from typing import Dict, Tuple
from torch.nn import functional as F
import torch
from argparse import ArgumentParser
# from pytorch_lightning.utilities.distributed import sync_ddp_if_available  # wrapper around torch.distributed.all_reduce
import torch.distributed as dist
from neptune.types import File
from ._ssl_base_model import SslModelBase
from ._resnet_backbone import (
    make_vae_decoder_backbone_from_scratch,
    make_vae_decoder_backbone_from_resnet,
    make_vae_encoder_backbone_from_scratch,
    make_vae_encoder_backbone_from_resnet,
)
from tissuemosaic.models._optim_scheduler import LARS, linear_warmup_and_cosine_protocol
from tissuemosaic.plots.plot_images import show_batch


class ConvolutionalVae(torch.nn.Module):
    def __init__(self,
                 backbone_type: str,
                 in_size: int,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: Tuple[int] = (32, 64, 128, 256, 512),
                 decoder_activation: torch.nn.Module = torch.nn.Identity(),
                 ) -> None:
        super(ConvolutionalVae, self).__init__()

        assert (in_size % 32) == 0, "The input size must be a multiple of 32. Received {0}".format(in_size)

        assert backbone_type in ('vanilla', 'resnet18', 'resnet34', 'resnet50'), \
            "Invalid vae_type. Received {0}".format(backbone_type)
        x_fake = torch.zeros((2, in_channels, in_size, in_size))

        # encoder
        print("making encoder", backbone_type)
        self.latent_dim = latent_dim
        if backbone_type == 'vanilla':
            self.encoder_backbone = make_vae_encoder_backbone_from_scratch(
                in_channels=in_channels,
                hidden_dims=hidden_dims
            )
        elif backbone_type.startswith("resnet"):
            self.encoder_backbone = make_vae_encoder_backbone_from_resnet(
                in_channels=in_channels,
                resnet_type=backbone_type
            )
        else:
            raise Exception("Invalid vae_type. Received {0}".format(backbone_type))

        x_latent = self.encoder_backbone(x_fake)
        small_ch = x_latent.shape[-3]
        self.small_size = x_latent.shape[-1]
        self.fc_mu = torch.nn.Linear(small_ch * self.small_size * self.small_size, latent_dim)
        self.fc_var = torch.nn.Linear(small_ch * self.small_size * self.small_size, latent_dim)

        # Decoder
        self.decoder_input = torch.nn.Linear(latent_dim, small_ch * self.small_size * self.small_size)

        z_to_decode = torch.zeros((2, small_ch, self.small_size, self.small_size))
        if backbone_type == 'vanilla':
            tmp_list = list(hidden_dims)
            tmp_list.reverse()
            reverse_hidden_dims = tuple(tmp_list)
            self.decoder_backbone = make_vae_decoder_backbone_from_scratch(hidden_dims=reverse_hidden_dims)
        elif backbone_type.startswith("resnet"):
            self.decoder_backbone = make_vae_decoder_backbone_from_resnet(resnet_type=backbone_type)
        else:
            raise Exception("Invalid vae_type. Received {0}".format(backbone_type))

        x_tmp = self.decoder_backbone(z_to_decode)
        ch_tmp = x_tmp.shape[-3]
        last_hidden_ch = min(ch_tmp, 64)

        self.final_layer = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=ch_tmp,
                out_channels=last_hidden_ch,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1,
                output_padding=1),
            torch.nn.BatchNorm2d(last_hidden_ch),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=last_hidden_ch,
                out_channels=in_channels,
                kernel_size=(3, 3),
                padding=1),
            decoder_activation,
        )

        # make sure the VAE reproduce the correct shape
        dict_vae = self.forward(x_fake)
        assert dict_vae['x_rec'].shape == x_fake.shape

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        Args:
            x: (Tensor) [B x C x H x W]
            verbose: bool

        Returns:
            mu, log_var (Tensors) [B x latent_dim]
        """
        result = self.encoder_backbone(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return {'mu': mu, 'log_var': log_var}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        Args:
            z: (Tensor) [B x D]
            verbose: bool

        Returns:
            (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(z.shape[0], -1, self.small_size, self.small_size)
        result = self.decoder_backbone(result)
        x_rec = self.final_layer(result)
        return x_rec

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        dict_encoder = self.encode(x)
        z = self.reparameterize(mu=dict_encoder['mu'], logvar=dict_encoder['log_var'])
        x_rec = self.decode(z)
        return {'x_rec': x_rec, 'x_in': x, 'mu': dict_encoder['mu'], 'log_var': dict_encoder['log_var']}


class VaeModel(SslModelBase):
    """
    Convolutional Variational Auto Encoders (VAE) with dynamically adjusted hyper-parameter :math:`\\beta`.

    The loss function is a weighted sum of the reconstruction (MSE) and regularization (KL):

    :math:`\\text{loss} = \\beta \\times \\text{KL} + (1-\\beta) \\times \\text{MSE}`

    We view this problem as a Multi-Objective Optimization (minimizing MSE *and* KL) and we dynamically
    adjust :math:`\\beta \\in (.0, 1.0)` taking inspiration from the ideas in
    `Multi-Task Learning as Multi-Objective Optimization
    <https://proceedings.neurips.cc/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf>`_

    Note:
        Depending on the data pre-processing step, the input images might be mostly zeros with few "spots" in them.
        In this case, the VAE might collapse to a local minimum in the loss function corresponding to identically
        zero reconstruction. You might try to solve this collapse by using a larger latent dimension or by implementing
        a KL-free variation of the VAE, such as `VQ-VAE <https://arxiv.org/pdf/1711.00937v2.pdf>`_.
        However, spotty images are inherently difficult for
        (unstructured) VAE which are required to retain (in their latent embedding) precise information about the
        location of each individual spots.
    """
    def __init__(
            self,
            # architecture
            backbone_type: str,
            global_size: int,
            image_in_ch: int,
            latent_dim: int,
            encoder_hidden_dims: Tuple[int],
            decoder_output_activation: str,

            # optimizer
            optimizer_type: str,
            beta_vae_init: float,
            momentum_beta_vae: float,

            # scheduler
            warm_up_epochs: int,
            warm_down_epochs: int,
            max_epochs: int,
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,

            # gradient clipping (these parameters are defined)
            gradient_clip_val: float = 0.0,
            gradient_clip_algorithm: str = 'value',

            # validation
            val_iomin_threshold: float = 0.0,
            run_classify_regress: bool = False,
            **kwargs,
            ):
        """
        Args:
            backbone_type: Either 'vanilla', 'resnet18', 'resnet34' or 'resnet50'
            global_size: Size in pixel of the input image. Must be a multiple of 32.
            image_in_ch: number of channels in the input images, used to adjust the first
                convolution filter in the backbone
            latent_dim: number of latent dimensions of the embeddings
            encoder_hidden_dims: Dimension of the hidden layers. Used only in :attr:`backbone_type` == 'vanilla'.
            decoder_output_activation: The non-linearity used to produce the reconstructed image.
                In most cases "identity" (default) will work just fine. This is true even when he pixel values are
                strictly positive and a "softplus" or "sigmoid" activations could be used.
            optimizer_type: Either 'adamw', 'sgd', 'adam' or 'rmsprop'.
            beta_vae_init: Initial value for :math:`\\beta` (the coefficient multiplying the KL divergence in the loss)
                It should be in (0.0, 1.0). The reconstruction error in the loss is multiplied by :math:`(1-\\beta)`.
            momentum_beta_vae: momentum for the Exponential Moving Average which updates the value of :math:`\\beta`.
                It should be in (0.0, 1.0).
            warm_up_epochs: epochs during which to linearly increase learning rate (at the beginning of training)
            warm_down_epochs: epochs during which to anneal learning rate with cosine protocoll (at the end of training)
            max_epochs: total number of epochs
            min_learning_rate: minimum learning rate (at the very beginning and end of training)
            max_learning_rate: maximum learning rate (after linear ramp)
            min_weight_decay: minimum weight decay (during the entirety of the linear ramp)
            max_weight_decay: maximum weight decay (reached at the end of training)
            gradient_clip_algorithm: Either "norm" or "value". The algorithm to use for gradient clipping.
            gradient_clip_val: Clip the gradients to this value. If 0 no clipping
            val_iomin_threshold: during validation, only patches with Intersection Over MinArea < IoMin_threshold
                are used. Should be in [0.0, 1.0). If 0 only strictly non-overlapping patches are allowed.
        """
        super(VaeModel, self).__init__(val_iomin_threshold=val_iomin_threshold, run_classify_regress=run_classify_regress)

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # Next two lines will make checkpointing much simpler. Always keep them as-is
        self.save_hyperparameters()  # all hyperparameters are saved to the checkpoint
        self.neptune_run_id = None  # if from scratch neptune_experiment_is is None

        # to make sure that you load the input images only once
        self.already_loaded_input_val_images = False

        # architecture
        if decoder_output_activation == 'identity':
            output_activation = torch.nn.Identity()
        elif decoder_output_activation == 'relu':
            output_activation = torch.nn.ReLU()
        elif decoder_output_activation == 'tanh':
            output_activation = torch.nn.Tanh()
        elif decoder_output_activation == 'softplus':
            output_activation = torch.nn.Softplus()
        elif decoder_output_activation == "sigmoid":
            output_activation = torch.nn.Sigmoid()
        else:
            raise Exception("invalid decoder_output_activation. Received {0}".format(decoder_output_activation))

        self.image_size = global_size
        self.vae = ConvolutionalVae(
            backbone_type=backbone_type,
            in_size=self.image_size,
            in_channels=image_in_ch,
            latent_dim=latent_dim,
            hidden_dims=tuple(encoder_hidden_dims),
            decoder_activation=output_activation
        )

        # stuff to do gradient clipping internally
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm

        # stuff to keep the gradients and adjust beta_vae
        self.loss_type = None
        self.grad_due_to_kl = None
        self.grad_due_to_mse = None
        self.grad_old = None
        assert 0.0 < beta_vae_init < 1.0, \
            "Error. beta_vae_init should be in (0,1). Received {0}".format(beta_vae_init)
        self.register_buffer("beta_vae", float(beta_vae_init) * torch.ones(1, requires_grad=False).float())
        self.momentum_beta_vae = momentum_beta_vae

        # optimizer
        self.optimizer_type = optimizer_type

        # scheduler
        assert warm_up_epochs + warm_down_epochs <= max_epochs
        self.learning_rate_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_learning_rate, max_learning_rate, min_learning_rate),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))
        self.weight_decay_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_weight_decay, min_weight_decay, max_weight_decay),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))

    @classmethod
    def add_specific_args(cls, parent_parser):
        """
        Utility functions which add parameters to argparse to simplify setting up a CLI

        Example:
            >>> import sys
            >>> import argparse
            >>> parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')
            >>> parser = VaeModel.add_specific_args(parser)
            >>> args = parser.parse_args(sys.argv[1:])
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')

        # validation
        parser.add_argument("--val_iomin_threshold", type=float, default=0.0,
                            help="during validation, only patches with IoMinArea < IoMin_threshold are used "
                                 "in the kn-classifier and kn-regressor.")

        # this model has manual optimization therefore it has to handle clipping internally.
        parser.add_argument("--gradient_clip_val", type=float, default=0.5,
                            help="Clip the gradients to this value. If 0 no clipping")
        parser.add_argument("--gradient_clip_algorithm", type=str, default="value", choices=["norm", "value"],
                            help="Algorithm to use for gradient clipping.")

        # architecture
        parser.add_argument("--backbone_type", type=str, default="resnet18",
                            choices=["vanilla", "resnet18", "resnet34", "resnet50"],
                            help="The backbone architecture of the VAE")
        parser.add_argument("--global_size", type=int, default=64,
                            help="size in pixel of the input image. Must be a multiple of 32")
        parser.add_argument("--image_in_ch", type=int, default=3, help="number of channels of the input image")
        parser.add_argument("--latent_dim", type=int, default=128, help="number of latent dimensions")
        parser.add_argument("--encoder_hidden_dims", type=int, nargs='*', default=[32, 64, 128, 256, 512],
                            help="dimension of the hidden layers. Used only in backbone_type='vanilla'.")
        parser.add_argument("--decoder_output_activation", type=str, default="identity",
                            choices=["sigmoid", "identity", "tanh", "softplus", "relu"],
                            help="The non-linearity used to produce the reconstructed image.")

        # optimizer
        parser.add_argument("--optimizer_type", type=str, default='adam', help="optimizer type",
                            choices=['adamw', 'sgd', 'adam', 'rmsprop'])

        # Parameters to update the beta (i.e. the balancing between MSE and KL)
        parser.add_argument('--beta_vae_init', type=float, default=0.1,
                            help="Initial value for beta (coefficient in front of KL). Should be in (0.0, 1.0)")
        parser.add_argument('--momentum_beta_vae', type=float, default=0.999,
                            help="momentum for the EMA which updates the value of beta")

        # scheduler
        parser.add_argument("--warm_up_epochs", default=10, type=int,
                            help="Number of epochs for the linear learning-rate warm up.")
        parser.add_argument("--warm_down_epochs", default=100, type=int,
                            help="Number of epochs for the cosine decay.")
        parser.add_argument("--max_epochs", type=int, default=300, help="maximum number of training epochs")

        parser.add_argument('--min_learning_rate', type=float, default=1e-5,
                            help="Target LR at the end of cosine protocol (smallest LR used during training).")
        parser.add_argument("--max_learning_rate", type=float, default=5e-4,
                            help="learning rate at the end of linear ramp (largest LR used during training).")

        parser.add_argument('--min_weight_decay', type=float, default=0.0,
                            help="Minimum value of the weight decay. It is used during the linear ramp.")
        parser.add_argument('--max_weight_decay', type=float, default=0.0,
                            help="Maximum Value of the weight decay. It is reached at the end of cosine protocol.")

        return parser

    @classmethod
    def get_default_params(cls) -> dict:
        """
        Get the default configuration parameters for this model

        Example:
            >>> config = VaeModel.get_default_params()
            >>> my_barlow = VaeModel(**config)
        """
        parser = ArgumentParser()
        parser = cls.add_specific_args(parser)
        args = parser.parse_args(args=[])
        return args.__dict__

    @staticmethod
    def compute_losses(x_in, x_rec, mu, log_var):
        # compute both kl and derivative of kl w.r.t. mu and log_var
        assert len(mu.shape) == 2
        batch_size = mu.shape[0]
        # kl_loss = 0.5 * (mu ** 2 + log_var.exp() - log_var - 1.0).sum() / batch_size ## this should be equivalent to the next line, but grad_fn is different
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        ## weight by M/N, where m is # of latent dims and n is # of pixels
        kl_weight = mu.shape[-1] / (x_in.shape[-1] * x_in.shape[-2])
        kl_loss = kl_weight * kl_loss

        mse_loss = F.mse_loss(x_in, x_rec, reduction='mean')

        return {
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
        }

    def head_and_backbone_embeddings_step(self, x):
        # this generates both head and backbone embeddings
        # return mu twice so that it is interpreted as backbone and head features
        mu = self(x)
        return mu, mu

    def forward(self, x) -> torch.Tensor:
        # this is the stuff that returns the embeddings "
        dict_encoder = self.vae.encode(x)
        return dict_encoder['mu']

    def training_step(self, batch, batch_idx):

        with torch.no_grad():
            # Update the optimizer parameters
            opt: torch.optim.Optimizer = self.optimizers()
            lr = self.learning_rate_fn(self.current_epoch)
            wd = self.weight_decay_fn(self.current_epoch)
            for i, param_group in enumerate(opt.param_groups):
                param_group["lr"] = lr
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd
                else:
                    param_group["weight_decay"] = 0.0

            # this is data augmentation
            list_imgs = batch[0]
            batch_size = len(list_imgs)
            img_in = self.trsfm_train_global(list_imgs)
            assert img_in.shape[-1] == self.image_size, \
                "img.shape {0} vs image_size {1}".format(img_in.shape[-1], self.image_size)

        # does the encoding-decoding
        dict_vae = self.vae(img_in)
        loss_dict = self.compute_losses(
            x_in=dict_vae['x_in'],
            x_rec=dict_vae['x_rec'],
            mu=dict_vae['mu'],
            log_var=dict_vae['log_var']
        )
        assert torch.all(torch.isfinite(dict_vae['mu'])), "In training step. mu in NOT finite"
        
        ## debugging
        if not torch.all(torch.isfinite(dict_vae['x_rec'])):
            print("debugging x_rec not finite")
            print(dict_vae['x_rec'])
            
            
        assert torch.all(torch.isfinite(dict_vae['x_rec'])), "In training step. x_rec is NOT finite"

        # Manual optimization
        opt.zero_grad()
        loss_kl = self.beta_vae * loss_dict["kl_loss"]
        loss_mse = (1.0 - self.beta_vae) * loss_dict["mse_loss"]

        if batch_idx == 0:
            # two backward passes to collect the two gradients separately
            self.manual_backward(loss_kl, retain_graph=True)
            grad_due_to_kl_tmp = self.__get_grad_from_last_layer_of_encoder__()
            self.manual_backward(loss_mse, retain_graph=False)
            grad_tot_tmp = self.__get_grad_from_last_layer_of_encoder__()
            grad_due_to_mse_tmp = grad_tot_tmp - grad_due_to_kl_tmp
        else:
            # a single backward pass
            self.manual_backward(loss_kl + loss_mse)
            grad_due_to_kl_tmp, grad_due_to_mse_tmp = None, None

        self.clip_gradients(
            opt,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm
        )
        opt.step()
        # end manual optimization

        with torch.no_grad():
            self.log('weight_decay', wd,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('learning_rate', lr,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)

            # Use the 75% quantile, i.e. we are requiring that 75% of the pixel are reconstructed better than rec_target
            mse_for_constraint = torch.quantile(
                input=(dict_vae['x_in'] - dict_vae['x_rec']).pow(2).sum(dim=-3),
                q=0.75
            )

            self.log('train_loss', loss_kl + loss_mse,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=batch_size, sync_dist=True)
            self.log('train_mse_loss', loss_dict['mse_loss'],
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=batch_size, sync_dist=True)
            self.log('train_kl_loss', loss_dict['kl_loss'],
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=batch_size, sync_dist=True)
            self.log('mse_for_constraint', mse_for_constraint,
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=batch_size, sync_dist=True)

            # batch_size
            self.log('batch_size_per_gpu_train', float(len(list_imgs)),
                     on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)

            # update the beta_vae if necessary
            if grad_due_to_mse_tmp is not None and grad_due_to_kl_tmp is not None:
                # grad_due_to_mse = sync_ddp_if_available(grad_due_to_mse_tmp, group=None, reduce_op='mean')
                # grad_due_to_kl = sync_ddp_if_available(grad_due_to_kl_tmp, group=None, reduce_op='mean')

                ## update to lightning 2.0; need to initialize the process group for this to work
                # grad_due_to_mse = dist.all_reduce(grad_due_to_mse_tmp, op=dist.ReduceOp.AVG)
                # grad_due_to_kl = dist.all_reduce(grad_due_to_kl_tmp, op=dist.ReduceOp.AVG)

                # TODO: add support for multi-gpu training
                grad_due_to_mse = grad_due_to_mse_tmp
                grad_due_to_kl = grad_due_to_kl_tmp

                c11 = torch.dot(grad_due_to_kl, grad_due_to_kl) / self.beta_vae**2
                c22 = torch.dot(grad_due_to_mse, grad_due_to_mse) / (1.0 - self.beta_vae)**2
                c12 = torch.dot(grad_due_to_kl, grad_due_to_mse) / (self.beta_vae * (1.0 - self.beta_vae))

                method = 0
                if method == 0:
                    # find beta in (0,1) which minimizes: || beta * grad_kl + (1-beta) * grad_mse ||^2
                    # see paper: "Multi-Task Learning as Multi-Objective Optimization"
                    # This is the close form solution
                    ideal_beta_vae = ((c22 - c12) / (c11 + c22 - 2 * c12)).clamp(min=0.0, max=1.0)
                elif method == 1:
                    # find beta in (0,1) which makes the two gradient equal size, i.e.:
                    # set: beta * sqrt(c11) = (1 - beta) * sqrt(c22)
                    # leads to: beta = sqrt(c22) / (sqrt(c11) + sqrt(c22))
                    ideal_beta_vae = (c22.sqrt() / (c11.sqrt() + c22.sqrt())).clamp(min=0.0, max=1.0)
                else:
                    raise Exception("Method can only be 0 or 1. Received {0}".format(method))

                # update beta using a slow Exponential Moving Average (EMA)
                self.__update_beta_vae__(ideal_beta=ideal_beta_vae, beta_momentum=self.momentum_beta_vae)

                self.log('beta/c11', c11, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1, sync_dist=True)
                self.log('beta/c12', c12, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1, sync_dist=True)
                self.log('beta/c22', c22, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1, sync_dist=True)
                self.log('beta/beta_vae', self.beta_vae,
                         on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1, sync_dist=True)
                self.log('beta/ideal_beta_vae', ideal_beta_vae,
                         on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1, sync_dist=True)

    def __get_grad_from_last_layer_of_encoder__(self) -> torch.Tensor:
        grad = torch.cat((
            self.vae.fc_mu.bias.grad.detach().clone().flatten(),
            self.vae.fc_mu.weight.grad.detach().clone().flatten(),
            self.vae.fc_var.bias.grad.detach().clone().flatten(),
            self.vae.fc_var.weight.grad.detach().clone().flatten()
        ), dim=0)
        return grad

    def __update_beta_vae__(self, ideal_beta, beta_momentum):
        # update only if the suggested beta is finite
        if ideal_beta.isfinite():
            tmp = beta_momentum * self.beta_vae + (1.0 - beta_momentum) * ideal_beta
            self.beta_vae = tmp.clamp(min=1.0E-5, max=1.0 - 1.0E-5)

    def validation_step(self, batch, batch_idx, dataloader_idx: int = -1):

        # Log an example of the reconstructed images
        if self.global_rank == 0 and batch_idx == 0:
            list_imgs = batch[0]
            img_in = self.trsfm_test(list_imgs)
            dict_vae = self.vae(img_in)

            img_out = dict_vae['x_rec'].clone().detach().float()  # make sure this is in full precision for plotting
            one_ch_tmp_out_plot = show_batch(img_out[0].unsqueeze(dim=-3), n_col=5,
                                             title="output, epoch={0}".format(self.current_epoch))
            self.logger.run["rec/output_imgs/one_ch"].log(File.as_image(one_ch_tmp_out_plot))
            all_ch_tmp_out_plot = show_batch(img_out[:10], n_col=5,
                                             title="output, epoch={0}".format(self.current_epoch))
            self.logger.run["rec/output_imgs/all_ch"].log(File.as_image(all_ch_tmp_out_plot))

            if not self.already_loaded_input_val_images:
                img_in_tmp = img_in.clone().detach().float()  # make sure this is in full precision for plotting
                one_ch_tmp_in_plot = show_batch(img_in_tmp[0].unsqueeze(dim=-3), n_col=5,
                                                title="input, epoch={0}".format(self.current_epoch))
                self.logger.run["rec/input_imgs/one_ch"].log(File.as_image(one_ch_tmp_in_plot))
                all_ch_tmp_in_plot = show_batch(img_in_tmp[:10], n_col=5,
                                                title="input, epoch={0}".format(self.current_epoch))
                self.logger.run["rec/input_imgs/all_ch"].log(File.as_image(all_ch_tmp_in_plot))
                self.already_loaded_input_val_images = True

        # call the super.validation_step
        return super(VaeModel, self).validation_step(batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        # the learning_rate and weight_decay are very large. They are just placeholder.
        # The real value will be set by the scheduler
        regularized = []
        not_regularized = []
        for name, param in self.vae.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        arg_for_optimizer = [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]

        # The real lr will be set in the training step
        # The weight_decay for the regularized group will be set in the training step
        if self.optimizer_type == 'adam':
            return torch.optim.Adam(arg_for_optimizer, betas=(0.9, 0.999), lr=0.0)
        elif self.optimizer_type == 'sgd':
            return torch.optim.SGD(arg_for_optimizer, momentum=0.9, lr=0.0)
        elif self.optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(arg_for_optimizer, alpha=0.99, lr=0.0)
        elif self.optimizer_type == 'lars':
            # for convnet with large batch_size
            return LARS(arg_for_optimizer, momentum=0.9, lr=0.0)
        else:
            # do adamw
            raise Exception("optimizer is misspecified")

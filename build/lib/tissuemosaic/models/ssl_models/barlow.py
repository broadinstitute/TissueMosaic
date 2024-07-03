from typing import List
import torch
from argparse import ArgumentParser
from pytorch_lightning.utilities.distributed import sync_ddp_if_available  # wrapper around torch.distributed.all_reduce
from neptune.new.types import File
from ._resnet_backbone import make_resnet_backbone
from ._ssl_base_model import SslModelBase
from tissuemosaic.plots.plot_misc import show_corr_matrix
from tissuemosaic.models._optim_scheduler import LARS, linear_warmup_and_cosine_protocol


class BarlowModel(SslModelBase):
    """
    Barlow self supervised learning model.
    Inspired by the `BarlowTwin official implementation <https://github.com/facebookresearch/barlowtwins>`_ and this
    `Barlow pytorch lightning reimplementation <https://github.com/IgorSusmelj/barlowtwins/blob/main/main.py>`_.
    """
    def __init__(
            self,
            # architecture
            backbone_type: str,
            image_in_ch: int,
            head_hidden_chs: List[int],
            head_out_ch: int,
            # loss
            lambda_off_diagonal: float,
            # optimizer
            optimizer_type: str,
            # scheduler
            warm_up_epochs: int,
            warm_down_epochs: int,
            max_epochs: int,
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,
            # validation
            val_iomin_threshold: float = 0.0,
            run_classify_regress: bool=False, 
            **kwargs,
            ):
        """
        Args:
            backbone_type: Either 'resnet18', 'resnet34' or 'resnet50'
            image_in_ch: number of channels in the input images, used to adjust the first
                convolution filter in the backbone
            head_hidden_chs: List of integers with the size of the hidden layers of the projection head
            head_out_ch: output dimension of the projection head
            lambda_off_diagonal: multiplicative factor for the off diagonal elements in the loss. Usually << 1.
            optimizer_type: Either 'adamw', 'lars', 'sgd', 'adam' or 'rmsprop'
            warm_up_epochs: epochs during which to linearly increase learning rate (at the beginning of training)
            warm_down_epochs: epochs during which to anneal learning rate with cosine protocoll (at the end of training)
            max_epochs: total number of epochs
            min_learning_rate: minimum learning rate (at the very beginning and end of training)
            max_learning_rate: maximum learning rate (after linear ramp)
            min_weight_decay: minimum weight decay (during the entirety of the linear ramp)
            max_weight_decay: maximum weight decay (reached at the end of training)
            val_iomin_threshold: during validation, only patches with Intersection Over MinArea < IoMin_threshold
                are used. Should be in [0.0, 1.0). If 0 only strictly non-overlapping patches are allowed.
            run_classify_regress: If True, runs classification/regression validation
        """

        super(BarlowModel, self).__init__(val_iomin_threshold=val_iomin_threshold, run_classify_regress=run_classify_regress)

        # Next two lines will make checkpointing much simpler. Always keep them as-is
        self.save_hyperparameters()  # all hyperparameters are saved to the checkpoint
        self.neptune_run_id = None  # if from scratch neptune_experiment_is is None

        # architecture
        self.backbone = make_resnet_backbone(
            backbone_in_ch=image_in_ch,
            backbone_type=backbone_type)

        tmp_in = torch.zeros((1, image_in_ch, 64, 64))
        tmp_out = self.backbone(tmp_in)
        backbone_ch_out = tmp_out.shape[1]

        self.projection = self.init_projection(
            ch_in=backbone_ch_out,
            ch_hidden=head_hidden_chs,
            ch_out=head_out_ch)
        # Use bn to center and scale the feature across multiple-gpus
        self.bn_final = torch.nn.BatchNorm1d(head_out_ch, affine=False)

        # loss
        self.lambda_off_diagonal = lambda_off_diagonal
        self.cross_corr = None

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

    @staticmethod
    def init_projection(
            ch_in: int,
            ch_out: int,
            ch_hidden: List[int] = None):

        sizes = [ch_in] + ch_hidden + [ch_out]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(torch.nn.BatchNorm1d(sizes[i + 1]))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
        return torch.nn.Sequential(*layers)

    @classmethod
    def add_specific_args(cls, parent_parser):
        """
        Utility functions which add parameters to argparse to simplify setting up a CLI

        Example:
            >>> import sys
            >>> import argparse
            >>> parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')
            >>> parser = BarlowModel.add_specific_args(parser)
            >>> args = parser.parse_args(sys.argv[1:])
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')

        # validation
        parser.add_argument("--val_iomin_threshold", type=float, default=0.0,
                            help="during validation, only patches with IoMinArea < IoMin_threshold are used "
                                 "in the kn-classifier and kn-regressor.")

        # architecture
        parser.add_argument("--image_in_ch", type=int, default=3, help="number of channels in the input images")
        parser.add_argument("--backbone_type", type=str, default="resnet34", help="backbone type",
                            choices=['resnet18', 'resnet34', 'resnet50'])
        parser.add_argument("--head_hidden_chs", type=int, nargs='+', default=[2048, 2048],
                            help="List of integers. Hidden channels in projection head.")
        parser.add_argument("--head_out_ch", type=int, default=2048, help="head output channels")

        # loss
        parser.add_argument("--lambda_off_diagonal", type=float, default=5e-3,
                            help="lambda multiplying off diagonal elements in the loss")

        # optimizer
        parser.add_argument("--optimizer_type", type=str, default='adam', help="optimizer type",
                            choices=['adamw', 'lars', 'sgd', 'adam', 'rmsprop'])

        # scheduler
        parser.add_argument("--max_epochs", default=1000, type=int,
                            help="Total number of epochs in training.")
        parser.add_argument("--warm_up_epochs", default=100, type=int,
                            help="Number of epochs for the linear learning-rate warm up.")
        parser.add_argument("--warm_down_epochs", default=500, type=int,
                            help="Number of epochs for the cosine decay.")
        parser.add_argument('--min_learning_rate', type=float, default=1e-5,
                            help="Target LR at the end of cosine protocol (smallest LR used during training).")
        parser.add_argument("--max_learning_rate", type=float, default=5e-4,
                            help="learning rate at the end of linear ramp (largest LR used during training).")
        parser.add_argument('--min_weight_decay', type=float, default=0.04,
                            help="Minimum value of the weight decay. It is used during the linear ramp.")
        parser.add_argument('--max_weight_decay', type=float, default=0.4,
                            help="Maximum Value of the weight decay. It is reached at the end of cosine protocol.")
        return parser

    @classmethod
    def get_default_params(cls) -> dict:
        """
        Get the default configuration parameters for this model

        Example:
            >>> config = BarlowModel.get_default_params()
            >>> my_barlow = BarlowModel(**config)
        """
        parser = ArgumentParser()
        parser = BarlowModel.add_specific_args(parser)
        args = parser.parse_args(args=[])
        return args.__dict__

    def on_validation_epoch_start(self) -> None:
        if self.global_rank == 0 and self.cross_corr is not None:
            corr_matrix_plot = show_corr_matrix(data=self.cross_corr.detach().clone(),
                                                show_colorbar=True,
                                                sup_title="epoch {0}".format(self.current_epoch))
            self.logger.run["corr_matrix"].log(File.as_image(corr_matrix_plot))

    def forward(self, x) -> torch.Tensor:
        # this is the stuff that will generate the backbone embeddings
        y = self.backbone(x)  # shape (batch, ch)
        return y

    def head_and_backbone_embeddings_step(self, x):
        # this generates both head and backbone embeddings
        y = self(x)  # shape: (batch, ch)
        z = self.projection(y)  # shape: (batch, latent)
        return z, y

    def training_step(self, batch, batch_idx) -> dict:
        with torch.no_grad():
            list_imgs, list_labels, list_metadata = batch
            x1 = self.trsfm_train_global(list_imgs)
            x2 = self.trsfm_train_global(list_imgs)

        # forward is inside the no-grad context
        z1, y1 = self.head_and_backbone_embeddings_step(x1)
        z2, y2 = self.head_and_backbone_embeddings_step(x2)

        # empirical cross-correlation matrix
        # note that batch-norm are syncronized therefore mean and std are computed across all devices
        corr_tmp = self.bn_final(z1).T @ self.bn_final(z2)  # shape: (latent, latent)
        batch_size_per_gpu = float(z1.shape[0])
        batch_size_total = sync_ddp_if_available(torch.tensor(z1.shape[0], device=z1.device, dtype=torch.float),
                                                 group=None,
                                                 reduce_op='sum')  # sum across devices
        corr_sum = sync_ddp_if_available(corr_tmp,
                                         group=None,
                                         reduce_op='sum')  # sum across devices
        corr = corr_sum / batch_size_total  # divide by total batch size
        self.cross_corr = corr.detach()  # this is for logging

        # compute the loss
        mask_diag = torch.eye(corr.shape[-1], dtype=torch.bool, device=corr.device)
        on_diag = corr[mask_diag].add_(-1.0).pow_(2).sum()
        off_diag = corr[~mask_diag].pow_(2).sum()
        loss = on_diag + self.lambda_off_diagonal * off_diag

        # Update the optimizer parameters and log stuff
        with torch.no_grad():
            lr = self.learning_rate_fn(self.current_epoch)
            wd = self.weight_decay_fn(self.current_epoch)
            for i, pg in enumerate(self.optimizers().param_groups):
                pg["lr"] = lr
                if i == 0:  # only the first group is regularized
                    pg["weight_decay"] = wd
                else:
                    pg["weight_decay"] = 0.0

            # Finally I log interesting stuff
            self.log('train_loss', loss, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('weight_decay', wd, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('learning_rate', lr, on_step=False, on_epoch=True, rank_zero_only=True, batch_size=1)
            self.log('batch_size_per_gpu_train', batch_size_per_gpu, on_step=False, on_epoch=True, rank_zero_only=True)
            self.log('batch_size_total_train', batch_size_total, on_step=False, on_epoch=True, rank_zero_only=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        regularized = []
        not_regularized = []
        for name, param in list(self.backbone.named_parameters()) + list(self.projection.named_parameters()):
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

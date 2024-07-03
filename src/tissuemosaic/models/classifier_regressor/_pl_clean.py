from pytorch_lightning import LightningModule
from typing import List, Callable, Tuple
import torch
from ._mlp import make_mlp_torch
from .._optim_scheduler import linear_warmup_and_cosine_protocol


class PlMlpClean(LightningModule):
    """ Simple LightningModule which process the input through a MLP net """
    def __init__(
            self,
            # loss
            criterium: Callable,
            # architecture
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            hidden_activation: torch.nn.Module,
            output_activation: torch.nn.Module,
            # optimizer
            solver: str,
            betas: Tuple[float, float],
            momentum: float,
            alpha: float,
            # protocoll
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,
            warm_up_epochs: int,
            warm_down_epochs: int,
            max_epochs: int
    ):
        super().__init__()
        # loss
        self.criterium = criterium
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim

        # architecture
        self.net = make_mlp_torch(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

        # optimizers
        self.solver = solver
        self.betas = betas
        self.momentum = momentum
        self.alpha = alpha

        # protocoll
        assert warm_up_epochs + warm_down_epochs <= max_epochs
        self.learning_rate_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_learning_rate, max_learning_rate, min_learning_rate),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))
        self.weight_decay_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_weight_decay, min_weight_decay, max_weight_decay),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))

        # hidden quantities
        self.loss_ = None
        self.loss_curve_ = []
        self.loss_accumulation_value_ = None
        self.loss_accumulation_counter_ = None

    @property
    def input_dim(self):
        return self.input_dim_

    @property
    def output_dim(self):
        return self.output_dim_

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        # Manually update the optimizer parameters
        with torch.no_grad():
            opt = self.optimizers()
            assert isinstance(opt, torch.optim.Optimizer)
            lr = self.learning_rate_fn(self.current_epoch)
            wd = self.weight_decay_fn(self.current_epoch)
            for i, param_group in enumerate(opt.param_groups):
                param_group["lr"] = lr
                param_group["weight_decay"] = wd
            # print("lr, wd", lr, wd)

        x, y, index = batch
        y_hat = self(x)
        loss = self.criterium(y_hat, y)

        # Compute few things of interests
        with torch.no_grad():
            self.loss_ = loss.detach().cpu().item()  # current value of loss
            self.loss_accumulation_value_ += self.loss_
            self.loss_accumulation_counter_ += 1

        return loss

    def on_train_epoch_start(self) -> None:
        self.loss_accumulation_value_ = 0.0
        self.loss_accumulation_counter_ = 0

    def on_train_epoch_end(self, unused=None) -> None:
        tmp = (self.loss_accumulation_value_ / self.loss_accumulation_counter_)
        self.loss_curve_.append(tmp)

    def configure_optimizers(self):
        if self.solver == 'adam':
            return torch.optim.Adam(
                self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                betas=self.betas)
        elif self.solver == 'sgd':
            return torch.optim.SGD(
                self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                momentum=self.momentum)
        elif self.solver == 'rmsprop':
            return torch.optim.RMSprop(
                self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                alpha=self.alpha)

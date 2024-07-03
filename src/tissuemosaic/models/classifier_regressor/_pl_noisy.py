import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional
from typing import List, Tuple
from ._mlp import make_mlp_torch
from .._optim_scheduler import linear_warmup_and_cosine_protocol


class PlMlpNoisy(LightningModule):
    """ This class will be used in PlClassifier (has-a relationship).

    pytorch Lightning implementation of 'Unsupervised Label Noise Modeling and Loss Correction'
    see: https://github.com/PaulAlbert31/LabelNoiseCorrection

    New labels are computed as:
    soft_labels = (1.0-w) * labels_one_hot + w * prob_one_hot  # if hard bootstrapping
    soft_labels = (1.0-w) * labels_one_hot + w * prob          # if soft bootstrapping
    where prob is what the NN predicts, and w is the probability of label being incorrect.
    It is computed by computing the assignment probability of a 2-component Mixture Model.
    It is based on the idea that correct/incorrect labels will lead to small/large losses.

    These is an additional regularization term so that corrected labels do not collapse to a single class
    empirical_class_distribution = torch.mean(prob, dim=-2)  # mean over batch. Shape (classes)
    loss_reg = self.kl_between_class_distributions(empirical_class_distribution)

    # Loss function is sum of CE w.r.t. the corrected labels plus: lambda_reg * loss_reg
    """
    def __init__(
            self,
            # architecture
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            hidden_activation: torch.nn.Module,
            # optimizer
            solver: str,
            betas: Tuple[float, float],
            momentum: float,
            alpha: float,
            # loss
            bootstrap_epoch_start: int,
            lambda_reg: float,
            hard_bootstrapping: bool,
            # protocoll
            warm_up_epochs: int,
            warm_down_epochs: int,
            max_epochs: int,
            min_learning_rate: float,
            max_learning_rate: float,
            min_weight_decay: float,
            max_weight_decay: float,

            # loss
    ):
        """
        Args:
            input_dim: input channels
            output_dim: output channels, i.e. number of classes
            hidden_dims: size of hidden layers
            hidden_activation: activation to apply to the hidden layers
            solver: one among 'adam' or 'sgd' or 'rmsprop'
            betas: parameters for the adam optimizer (used only if :attr:'solver' is 'adam')
            momentum: parameter for sgd optimizer (used only if :attr:'solver' is 'sgd')
            alpha: parameters for the rmsprop optimizer (used only if :attr:'solver' is 'rmsprop')
            bootstrap_epoch_start: bootstrapping, i.e. correcting the labels will start after this many epochs.
                The idea is that the model learns a bit before the loss are meaningful to identify incorrect labels.
            lambda_reg: strength of regularization to avoid that the corrected labels collapse to a single class.
            hard_bootstrapping: if True the correct labels are a weighted sum of two delta-functions. If False
                the corrected labels are a weighted sum of a delta-function and the network probability.
            warm_up_epochs: epochs during which to linearly increase learning rate (at the beginning of training)
            warm_down_epochs: epochs during which to anneal learning rate with cosine protocoll (at the end of training)
            max_epochs: total number of epochs
            min_learning_rate: minimum learning rate (at the very beginning and end of training)
            max_learning_rate: maximum learning rate (after linear ramp)
            min_weight_decay: minimum weight decay (during the entirety of the linear ramp)
            max_weight_decay: maximum weight decay (reached at the end of training)
        """
        super().__init__()
        # loss
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim

        # make a mlp
        self.net = make_mlp_torch(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            hidden_activation=hidden_activation,
            output_activation=torch.nn.Identity(),
        )

        # loss
        self.bootstrap_epoch_start = bootstrap_epoch_start
        self.lambda_reg = lambda_reg
        self.hard_bootstrapping = hard_bootstrapping

        # optimizers
        self.solver = solver
        self.betas = betas
        self.momentum = momentum
        self.alpha = alpha

        # protocol
        assert warm_up_epochs + warm_down_epochs <= max_epochs
        self.learning_rate_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_learning_rate, max_learning_rate, min_learning_rate),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))
        self.weight_decay_fn = linear_warmup_and_cosine_protocol(
            f_values=(min_weight_decay, min_weight_decay, max_weight_decay),
            x_milestones=(0, warm_up_epochs, max_epochs - warm_down_epochs, max_epochs))

        # BetaMixture business
        self.bmm_model = BetaMixture1D(max_iters=50)
        self.bmm_model_maxLoss = None
        self.bmm_model_minLoss = None
        self.track_loss_hard_ce = None

        # hidden quantities
        self.bmm_fitted_alpha0_curve_ = []
        self.bmm_fitted_alpha1_curve_ = []
        self.bmm_fitted_beta0_curve_ = []
        self.bmm_fitted_beta1_curve_ = []
        self.bmm_fitted_mean0_curve_ = []
        self.bmm_fitted_mean1_curve_ = []
        self.learning_rate_curve_ = []
        self.weight_decay_curve_ = []

        # for compatibility with Scikit-learn, keep track of loss and loss_curve
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
        # this is what will be called when asking for a prediction
        return self.net(x)

    def compute_assignment_prob_bmm_model(self, log_prob, soft_targets):
        assert log_prob.shape == soft_targets.shape  # shape (N, C)
        batch_losses = - (soft_targets * log_prob).sum(dim=-1)  # shape N
        batch_losses = (batch_losses - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss)
        batch_losses.clamp_(min=1E-4, max=1.0-1E-4)
        assignment_prob = self.bmm_model.look_lookup(batch_losses)  # shape N with values in (0, 1)
        return torch.from_numpy(assignment_prob).to(device=log_prob.device).float()  # shape N

    @staticmethod
    def kl_between_class_distributions(empirical_hist):
        assert len(empirical_hist.shape) == 1  # shape (Classes)
        prior_hist = torch.ones_like(empirical_hist)
        prior = prior_hist / prior_hist.sum()
        post = empirical_hist
        tmp = prior * (prior.log() - post.log())
        kl = torch.where(torch.isfinite(tmp), tmp, torch.zeros_like(tmp))
        return kl.sum(dim=-1)  # shape -> scalar

    @staticmethod
    def soft_ce_loss(logits, soft_targets):
        assert len(logits.shape) == 2
        assert logits.shape == soft_targets.shape  # shape (N, C)
        loss = - (soft_targets * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)  # shape N
        assert loss.shape == torch.Size([logits.shape[0]])
        return loss  # shape (N)

    @staticmethod
    def hard_ce_loss(logits, targets):
        assert len(logits.shape) == 2  # shape (N,C)
        assert len(targets.shape) == 1  # shape (N)
        assert logits.shape[0] == targets.shape[0]
        loss_ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')  # shape (N)
        assert loss_ce.shape[0] == targets.shape[0]
        return loss_ce  # shape (N)

    def train_bootstrapping_beta(self, logits, targets, lambda_reg: float = 1.0, hard_bootstrapping: bool = False):
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        labels_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.output_dim)

        # look up the assignment probability
        with torch.no_grad():
            if self.current_epoch < 1:
                w = 0.5 * torch.ones_like(targets).float()
            else:
                w = self.compute_assignment_prob_bmm_model(log_prob, labels_one_hot)
                w.clamp_(min=1E-4, max=1.0-1E-4)
            w = w.unsqueeze(dim=-1)  # add a singleton for the class_dimension

        # compute loss w.r.t. the corrected labels
        if hard_bootstrapping:
            prob_one_hot = torch.nn.functional.one_hot(torch.argmax(prob, dim=-1), num_classes=self.output_dim)
            soft_labels = (1.0-w) * labels_one_hot + w * prob_one_hot
        else:
            soft_labels = (1.0-w) * labels_one_hot + w * prob
        loss_ce_soft = self.soft_ce_loss(logits=logits, soft_targets=soft_labels)

        # add a regularization term so that corrected labels do not collapse to a single class
        empirical_class_distribution = torch.mean(prob, dim=-2)  # mean over batch. Shape (classes)
        loss_reg = self.kl_between_class_distributions(empirical_class_distribution)

        # put all the losses together
        loss_training = (loss_ce_soft + lambda_reg * loss_reg).mean()
        loss_hard_ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        return {"loss_training": loss_training, "loss_hard_ce": loss_hard_ce.detach()}

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

        # compute the loss
        x, y, index = batch
        logits = self.net(x)
        if self.current_epoch < self.bootstrap_epoch_start:
            loss_ce_hard = self.hard_ce_loss(logits=logits, targets=y)
            loss_dict = {"loss_training": loss_ce_hard.mean(), "loss_hard_ce": loss_ce_hard.detach()}
        else:
            loss_dict = self.train_bootstrapping_beta(logits=logits,
                                                      targets=y,
                                                      lambda_reg=self.lambda_reg,
                                                      hard_bootstrapping=self.hard_bootstrapping)

        # track the training losses
        with torch.no_grad():
            self.track_loss_hard_ce.append(loss_dict["loss_hard_ce"])

            # This block compute the current loss and the average over one epoch
            self.loss_ = loss_dict["loss_training"].detach().cpu().item()  # current value of loss
            self.loss_accumulation_value_ += self.loss_
            self.loss_accumulation_counter_ += 1

        return loss_dict["loss_training"]

    def on_train_epoch_start(self) -> None:
        # stuff for compatibility with scikit learn
        self.loss_accumulation_value_ = 0.0
        self.loss_accumulation_counter_ = 0

        # Track stuff for BMM model
        self.track_loss_hard_ce = []

    def on_train_epoch_end(self, unused=None) -> None:
        # stuff for compatibility with scikit learn
        tmp = (self.loss_accumulation_value_ / self.loss_accumulation_counter_)
        self.loss_curve_.append(tmp)

        # Track stuff for BMM model
        all_losses = torch.cat(self.track_loss_hard_ce, dim=0)
        max_perc = torch.quantile(all_losses, q=0.95)
        min_perc = torch.quantile(all_losses, q=0.05)

        # disregard the outliers
        all_losses = all_losses[(all_losses <= max_perc) & (all_losses >= min_perc)]

        # update the BMM model
        self.bmm_model_minLoss = min_perc
        self.bmm_model_maxLoss = max_perc

        # rescale the losses in (eps, 1-eps)
        all_losses = (all_losses - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss)
        all_losses.clamp_(min=1E-4, max=1.0-1E-4)

        # fit the bmm_model based on the rescaled loss
        self.bmm_model.fit(all_losses)
        self.bmm_model.create_lookup(1)

        # keep track of the values fitted by the BetaMixtureModel
        alphas = self.bmm_model.alphas
        betas = self.bmm_model.betas
        self.bmm_fitted_alpha0_curve_.append(alphas[0])
        self.bmm_fitted_alpha1_curve_.append(alphas[1])
        self.bmm_fitted_beta0_curve_.append(betas[0])
        self.bmm_fitted_beta1_curve_.append(betas[1])

        # rescale those values from dimensionless to dimensionfull
        delta_loss = (max_perc - min_perc).cpu().numpy()
        min_loss = min_perc.cpu().numpy()
        fitted_means = min_loss + delta_loss * alphas / (alphas + betas)
        self.bmm_fitted_mean0_curve_.append(fitted_means[0])
        self.bmm_fitted_mean1_curve_.append(fitted_means[1])

        # keep track of learning rate and weight decay
        opt = self.optimizers()
        param_group = opt.param_groups[0]
        lr = param_group["lr"]
        wd = param_group["weight_decay"]
        self.learning_rate_curve_.append(lr)
        self.weight_decay_curve_.append(wd)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.solver == 'adam':
            return torch.optim.Adam(
                params=self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                betas=self.betas)
        elif self.solver == 'sgd':
            return torch.optim.SGD(
                params=self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                momentum=self.momentum)
        elif self.solver == 'rmsprop':
            return torch.optim.RMSprop(
                params=self.net.parameters(),
                lr=0.0,
                weight_decay=0.0,
                alpha=self.alpha)
        else:
            raise Exception("unrecognized Optimizer. Received {0}".format(self.solver))


# This block of code is taken from:
# https://github.com/PaulAlbert31/LabelNoiseCorrection
# It fits the CrossEntropyLoss of a classifier with a two components Beta Distribution using the EM algorithm.
# Low losses correspond to correct labels, High losses to incorrect labels.
# For each instance you get the assignment probability, w_i, to the correct and incorrect component. 
# The number of classes in the classifier can be LARGER than 2. 
# The corrected labels are computed as:
# new_label = (1-w_i) * y_i + w_i * z_i
# where:
# a. y_i is the (possibly incorrect) hard label
# b. z_i is the label generated by the classifier (can be either one-hot or soft)
# c. w_i is the assignment probability to the high-loss component (i.e. probability of label being incorrect).


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(
            self, 
            max_iters=10,
            alphas_init=(2, 5),
            betas_init=(5, 2),
            weights_init=(0.5, 0.5)):
        # save the initial values
        self.alphas_init = alphas_init
        self.betas_init = betas_init
        self.weights_init = weights_init
        # other parameters
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12
        # for debug
        self.empirical_loss = None

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def reset_param(self):
        self.alphas = np.array(self.alphas_init, dtype=np.float64)
        self.betas = np.array(self.betas_init, dtype=np.float64)
        self.weight = np.array(self.weights_init, dtype=np.float64)

    def switch_param(self):
        tmp_alphas = np.copy(self.alphas[::-1])
        tmp_betas = np.copy(self.betas[::-1])
        tmp_weight = np.copy(self.weight[::-1])
        self.alphas = tmp_alphas
        self.betas = tmp_betas
        self.weight = tmp_weight

    def fit(self, x, reset: bool = False):
        # set the parameters to their initial value to start the EM algorithm
        if reset:
            self.reset_param()

        x = x.cpu().numpy()
        x = np.copy(x)
        self.empirical_loss = x  # store the value which have been used to fit the model

        # EM on beta distributions un-usable with x == 0 or x==1
        eps = 1e-4
        x = np.clip(x, a_min=eps, a_max=1-eps)

        # EM algorithm
        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

            # on rare occasion the two components can be switched
            # (i.e. 0 -> high value and 1 -> low value).
            # If this is the case, switch them back
            fitted_means_tmp = self.alphas / (self.alphas + self.betas)
            if fitted_means_tmp[0] > fitted_means_tmp[1]:
                self.switch_param()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l  # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.hist(self.empirical_loss, bins=50, density=True)
        plt.legend()

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

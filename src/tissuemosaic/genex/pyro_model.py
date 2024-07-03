from typing import Tuple, Union, List
import torch
import numpy
import pyro
import time
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.ops.special import get_quad_rule
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property
from pyro.infer import SVI, Trace_ELBO
import pandas as pd
import pyro.poutine
import pyro.optim
import matplotlib.pyplot as plt
from .gene_utils import GeneDataset


class LogNormalPoisson(TorchDistribution):
    """
    A Poisson distribution with rate: :math:`r = N \\times \\exp\\left[ \\log \\mu + \\epsilon \\right]`
    where noise is normally distributed with mean zero and variance sigma, i.e. :math:`\\epsilon \\sim N(0, \\sigma)`.

    See `Mingyuan <http://people.ee.duke.edu/~lcarin/Mingyuan_ICML_2012.pdf>`_ for discussion
    of the nice properties of the LogNormalPoisson model.
    """

    arg_constraints = {
        "n_trials": constraints.positive,
        "log_rate": constraints.real,
        "noise_scale": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(
            self,
            n_trials: torch.Tensor,
            log_rate: torch.Tensor,
            noise_scale: torch.Tensor,
            *,
            num_quad_points=8,
            validate_args=None, ):
        """
        Args:
            n_trials: non-negative number of Poisson trials, i.e. `N`.
            log_rate: the log_rate of a single trial, i.e. :math:`\\log \\mu`.
            noise_scale: controls the level of the injected noise, i.e. :math:`\\sigma`.
            num_quad_points: number of quadrature points used to compute the (approximate) `log_prob`. Defaults to 8.
        """

        if num_quad_points < 1:
            raise ValueError("num_quad_points must be positive.")

#         if not torch.all(torch.isfinite(log_rate)):
#             raise ValueError("log_rate must be finite.")
#
#         if torch.any(noise_scale < 0):
#             raise ValueError("Noise_scale must be positive.")
#
#         if not torch.all(n_trials > 0):
#             raise ValueError("n_trials must be positive")

        n_trials, log_rate, noise_scale = broadcast_all(
            n_trials, log_rate, noise_scale
        )

        self.quad_points, self.log_weights = get_quad_rule(num_quad_points, log_rate)
        quad_log_rate = (
                log_rate.unsqueeze(-1)
                + noise_scale.unsqueeze(-1) * self.quad_points
        )
        quad_rate = quad_log_rate.exp()

#         assert torch.all(torch.isfinite(quad_rate)), \
#             "Quad_Rate is not finite. quad_log_rate={}, quad_rate={}".format(quad_log_rate, quad_rate)
#         assert n_trials.device == quad_rate.device, "Got {0} and {1}".format(n_trials.device, quad_rate.device)

        self.poi_dist = dist.Poisson(rate=n_trials.unsqueeze(-1) * quad_rate)

        self.n_trials = n_trials
        self.log_rate = log_rate
        self.noise_scale = noise_scale
        self.num_quad_points = num_quad_points

        batch_shape = broadcast_shape(
            noise_scale.shape, self.poi_dist.batch_shape[:-1]
        )
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        poi_log_prob = self.poi_dist.log_prob(value.unsqueeze(-1))
        return torch.logsumexp(self.log_weights + poi_log_prob, axis=-1)

    def sample(self, sample_shape=torch.Size()):
        eps = dist.Normal(loc=0, scale=self.noise_scale).sample(sample_shape=sample_shape)
        return dist.Poisson(rate=self.n_trials * torch.exp(self.log_rate + eps)).sample()

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)

        n_trials = self.n_trials.expand(batch_shape)
        log_rate = self.log_rate.expand(batch_shape)
        noise_scale = self.noise_scale.expand(batch_shape)
        LogNormalPoisson.__init__(
            new,
            n_trials,
            log_rate,
            noise_scale,
            num_quad_points=self.num_quad_points,
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def mean(self):
        return self.n_trials * torch.exp(
            self.log_rate
            + 0.5 * self.noise_scale.pow(2.0)
        )

    @lazy_property
    def variance(self):
        kappa = torch.exp(self.noise_scale.pow(2.0))
        return self.mean + self.mean.pow(2.0) * (kappa - 1.0)


class GeneRegression:
    """
    Given the cell-type labels and some covariates the model predicts the gene expression.
    The counts are modelled as a LogNormalPoisson process. See documentation for more details.
    """

    def __init__(self):
        self._optimizer = None
        self._optimizer_initial_state = None
        self._loss_history = []
        self._train_kargs = None

    def _get_gene_list(self) -> List[str]:
        return self._train_kargs["gene_names"]

    def _get_cell_type_mapping(self) -> dict:
        return self._train_kargs["cell_type_mapping"]

    def _get_inverse_cell_type_mapping(self) -> dict:
        cell_type_mapping = self._get_cell_type_mapping()

        # Invert the cell_type_mapping (of the form: "cell_type" -> integer code)
        # to inverse_cell_type_mapping (of the form: integer_code -> "cell_types")
        # Note that multiple cell_types can be assigned to the same integer codes thefefor the inversion need
        # to keep track of possible name collisions
        inverse_cell_type_mapping = dict()
        for cell_type_name, code in cell_type_mapping.items():
            try:
                existing = inverse_cell_type_mapping[code]
                inverse_cell_type_mapping[code] = existing + "_AND_" + str(cell_type_name)
            except KeyError:
                inverse_cell_type_mapping[code] = str(cell_type_name)
        return inverse_cell_type_mapping

    def _get_cell_type_names_kg(self, g) -> numpy.ndarray:
        """ Return a numpy.array of shape k_cell_type by g with the cell_type_names """
        inverse_cell_type_mapping = self._get_inverse_cell_type_mapping()
        k_cell_types = len(inverse_cell_type_mapping.keys())
        cell_types_codes = torch.arange(k_cell_types).view(-1, 1).expand(k_cell_types, g)
        cell_types_names_kg = numpy.array(list(inverse_cell_type_mapping.values()))[cell_types_codes.cpu().numpy()]
        return cell_types_names_kg

    def _get_gene_names_kg(self, k: int) -> numpy.ndarray:
        """ Return a numpy.array of shape k by genes with the gene_names """
        gene_names_list = self._get_gene_list()
        len_genes = len(gene_names_list)
        gene_codes = torch.arange(len_genes).view(1, -1).expand(k, len_genes)
        gene_names_kg = numpy.array(gene_names_list)[gene_codes.cpu().numpy()]
        return gene_names_kg

    def _model(self,
               n_cells: int,
               g_genes: int,
               l_cov: int,
               k_cell_types: int,
               use_covariates: bool,
               counts_ng: torch.Tensor,
               total_umi_n: torch.Tensor,
               covariates_nl: torch.Tensor,
               cell_type_ids_n: torch.Tensor,
               beta0_g_init: torch.Tensor,
               eps_range: Tuple[float, float],
               l1_regularization_strength: float,
               l2_regularization_strength: float,
               subsample_size_cells: int,
               subsample_size_genes: int,
               **kargs):

        # Define the right device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        one = torch.ones(1, device=device)

        # Define the plates (i.e. conditional independence). It make sense to subsample only gene and cells.
        #cell_plate = pyro.plate("cells", size=n_cells, dim=-3, device=device, subsample_size=subsample_size_cells)
        #cell_types_plate = pyro.plate("cell_types", size=k_cell_types, dim=-3, device=device)
        gene_plate = pyro.plate("genes", size=g_genes, dim=-1, device=device, subsample_size=subsample_size_genes)
        
        ## put cell types as dim = -2
        cell_plate = pyro.plate("cells", size=n_cells, dim=-2, device=device, subsample_size=subsample_size_cells)
        cell_types_plate = pyro.plate("cell_types", size=k_cell_types, dim=-2, device=device)

        eps_kg1 = pyro.param("eps",
                             0.5 * (eps_range[0] + eps_range[1]) * torch.ones((k_cell_types, g_genes, 1),
                                                                              device=device),
                             constraint=constraints.interval(lower_bound=eps_range[0],
                                                             upper_bound=eps_range[1]))

        # Figure out a reasonable initialization for beta0_kg
        beta0_kg1 = pyro.param("beta0", beta0_g_init[None, None].expand_as(eps_kg1).to(device))

        with gene_plate:
            with cell_types_plate:
                if l1_regularization_strength is not None:
                    # l1 prior
                    mydist = dist.Laplace(loc=0, scale=one / l1_regularization_strength)
                elif l2_regularization_strength is not None:
                    # l2 prior
                    mydist = dist.Normal(loc=0, scale=one / l2_regularization_strength)
                else:
                    # no prior (note the mask statement. This will always give log_prob and kl_divergence =0)
                    mydist = dist.Normal(loc=0, scale=0.1).mask(False)

                #beta_klg = pyro.sample("beta_cov", mydist.expand([k_cell_types, l_cov, g_genes]))
                beta_kgl = pyro.sample("beta_cov", mydist.expand([l_cov]).to_event(1)) ## will expand in context of plates automatically
                assert beta_kgl.shape == torch.Size([k_cell_types, g_genes, l_cov]), \
                    "Received {}".format(beta_klg.shape)

        with cell_plate as ind_n:
            cell_ids_sub_n = cell_type_ids_n[ind_n].to(device)
            beta0_ng1 = beta0_kg1[cell_ids_sub_n]
            eps_ng1 = eps_kg1[cell_ids_sub_n]
            beta_ngl = beta_kgl[cell_ids_sub_n]
            total_umi_n11 = total_umi_n[ind_n, None, None].to(device)
            if use_covariates:
                covariate_sub_n1l = covariates_nl[cell_ids_sub_n].unsqueeze(dim=-2).to(device)

            with gene_plate as ind_g:
                eps_sub_ng1 = eps_ng1[None, ind_g, None] ## FIX THIS 
                if use_covariates:
                    log_mu_ng1 = beta0_ng1[None, ind_g, None] + \
                                 torch.sum(covariate_sub_n1l * beta_ngl[None, ind_g, None], dim=-1, keepdim=True)
                else:
                    log_mu_ng1 = beta0_n1g[None, ind_g, None]

                pyro.sample("counts",
                            LogNormalPoisson(n_trials=total_umi_n11,
                                             log_rate=log_mu_n1g,
                                             noise_scale=eps_sub_n1g,
                                             num_quad_points=8),
                            obs=counts_ng[ind_n.cpu(), None].index_select(dim=-2, index=ind_g.cpu()).to(device))
                ## make sure dimensions are corresponding to the cell plate and gene plate dimensions
                ## go over LogNormalPoisson distribution, make sure handling dimensions correctly 

    def _guide(self,
               g_genes: int,
               l_cov: int,
               k_cell_types: int,
               use_covariates: bool,
               subsample_size_genes: int,
               **kargs):

        # Define the right device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Define the gene and cell plates. It make sense to subsample only gene and cells.
        cell_types_plate = pyro.plate("cell_types", size=k_cell_types, dim=-2, device=device) 
        gene_plate = pyro.plate("genes", size=g_genes, dim=-1, device=device, subsample_size=subsample_size_genes)

        beta_param_loc_kgl = pyro.param("beta", 0.1 * torch.randn((k_cell_types, g_genes, l_cov), device=device))

        with gene_plate as ind_g:
            with cell_types_plate:
                if use_covariates:
                    beta_loc_tmp = beta_param_loc_kgl[None, ind_g, None]
                else:
                    beta_loc_tmp = torch.zeros_like(beta_param_loc_kgl[None, ind_g, None])
                pyro.sample("beta_cov", dist.Delta(v=beta_loc_tmp))

    @property
    def optimizer(self) -> pyro.optim.PyroOptim:
        """ The optimizer associated with this model. """
        assert self._optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
        return self._optimizer

    def configure_optimizer(self,
                            optimizer_type: str = 'adam',
                            lr: float = 5E-3,
                            betas: Tuple[float, float] = (0.9, 0.999),
                            momentum: float = 0.9,
                            alpha: float = 0.99):
        """
        Configure the optimizer to use.

        Args:
            optimizer_type: Either 'adam' (default), 'sgd' or 'rmsprop'
            lr: learning rate
            betas: betas for 'adam' optimizer. Ignored if :attr:`optimizer_type` is not 'adam'.
            momentum: momentum for 'sgd' optimizer. Ignored if :attr:`optimizer_type` is not 'sgd'.
            alpha: alpha for 'rmsprop' optimizer. Ignored if :attr:`optimizer_type` is not 'rmsprop'.
        """
        if optimizer_type == 'adam':
            self._optimizer = pyro.optim.Adam({"lr": lr, "betas": betas})
        elif optimizer_type == 'sgd':
            self._optimizer = pyro.optim.SGD({"lr": lr, "momentum": momentum})
        elif optimizer_type == 'rmsprop':
            self._optimizer = pyro.optim.RMSprop({"lr": lr, "alpha": alpha})
        else:
            raise ValueError("optimizer_type should be either 'adam', 'sgd' or 'rmsprop'. \
                              Received {}".format(optimizer_type))
        self._optimizer_initial_state = self._optimizer.get_state()

    def save_ckpt(self, filename: str):
        """
        Save the full state of the model and optimizer to disk.
        Use it in pair with :meth:`load_ckpt`.

        Note:
            Pyro saves unconstrained parameters and the constrain transformation.
            This means that if you manually "look inside" the ckpt you will see strange values.
            To get the actual value of the fitted parameter use the :meth:`get_params` method.
        """
        ckpt = {
            "param_store": pyro.get_param_store().get_state(),
            "optimizer": self._optimizer,
            "optimizer_state": self._optimizer.get_state(),
            "optimizer_initial_state": self._optimizer_initial_state,
            "loss_history": self._loss_history,
            "train_kargs": self._train_kargs
        }

        with open(filename, "wb") as output_file:
            torch.save(ckpt, output_file)

    def load_ckpt(self, filename: str, map_location=None):
        """
        Load the full state of the model and optimizer from disk.
        Use it in pair with :meth:`save_ckpt`.
        """

        with open(filename, "rb") as input_file:
            ckpt = torch.load(input_file, map_location)

        pyro.clear_param_store()
        pyro.get_param_store().set_state(ckpt["param_store"])
        self._optimizer = ckpt["optimizer"]
        self._optimizer.set_state(ckpt["optimizer_state"])
        self._optimizer_initial_state = ckpt["optimizer_initial_state"]
        self._loss_history = ckpt["loss_history"]
        self._train_kargs = ckpt["train_kargs"]

    @staticmethod
    def remove_params(beta0: bool = False, beta: bool = False, eps: bool = False):
        """
        Selectively remove parameters from param_store.

        Args:
            beta0: If True (defaults is False) remove :attr:`beta0` of shape :math:`(N, G)` from the param_store
            beta: If True (defaults is False) remove :attr:`beta` of shape :math:`(N, L, G)` from the param_store
            eps: If True (defaults is False) remove :attr:`eps` of shape :math:`(N, G)` from the param_store

        Note:
            This is useful in combination with :meth:`load_ckpt` and :meth:`train`.
            For example you might have fitted a model with `l1` covariate and wanting to try a different
            model with `l2` covariate. You can load the previous ckpt and remove :attr:`beta`
            while keeping :attr:`beta0` and :attr:`eps` (which do not depend on the number of covariate)

        Example:
             >>> gr.load_ckpt("ckpt_with_l1_covariate.pt")
             >>> gr.remove_from_param_store(beta=True)
             >>> gr.train(dataset=dataset_with_l2_covariate, initialization_type="pretrained")
        """
        if (not beta0) and (not beta) and (not eps):
            raise ValueError("At least one attributes should be true otherwise there is nothing to do")

        if beta0:
            pyro.get_param_store().__delitem__("beta0")
        if beta:
            pyro.get_param_store().__delitem__("beta")
        if eps:
            pyro.get_param_store().__delitem__("eps")

    def get_params(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Returns:
            df: dataframe with the fitted parameters.

        Note:
            This method can be used in combination with :math:`load_ckpt` to inspect
            the fitted parameters of a previous run.

        Examples:
            >>> gr = GeneRegression()
            >>> gr.load_ckpt(filename="my_old_ckpt.pt")
            >>> df_beta0, df_beta, df_eps = gr.get_params()
            >>> df_beta0.head()
        """

        # get all the fitted parameters
        mydict = dict()
        for k, v in pyro.get_param_store().items():
            mydict[k] = v.detach().cpu()

        assert set(mydict.keys()).issuperset({"beta0", "eps"}), \
            "Error. Unexpected parameter names {}".format(mydict.keys())

        k_cell_types = mydict["beta0"].shape[0]
        len_genes = mydict["eps"].shape[-2]
        cell_types_names_kg = self._get_cell_type_names_kg(g=len_genes)
        gene_names_kg = self._get_gene_names_kg(k=k_cell_types)

        # check shapes
        # eps.shape = (cell_type, 1, genes)
        assert mydict["eps"].shape == torch.Size([k_cell_types, len_genes, 1]), \
            "Unexpected shape for eps {}".format(mydict["eps"].shape)

        # beta0.shape = (cell_types, 1, genes)
        assert mydict["beta0"].shape == torch.Size([k_cell_types, len_genes, 1]), \
            "Unexpected shape for beta0 {}".format(mydict["beta0"].shape)

        # Create dataframe with beta_0 and beta (if present).
        # Beta might not be there if you used partial_load_ckpt
        if "beta" in set(mydict.keys()):
            # beta.shape = (cell_types, covariates, genes)
            tmp_a, tmp_b, tmp_c = mydict["beta"].shape
            assert tmp_a == k_cell_types and tmp_c == len_genes, \
                "Unexpected shape for beta {}".format(mydict["beta"].shape)

            beta = mydict["beta"].permute(0, 2, 1)  # shape: (cell_types, genes, covariates)
            columns = ["beta_{}".format(i+1) for i in range(beta.shape[-1])]
            df = pd.DataFrame(beta.flatten(end_dim=-2).cpu().numpy(), columns=columns)
            df["beta_0"] = mydict["beta0"].squeeze(dim=-2).flatten().cpu().numpy()
        else:
            df = pd.DataFrame(mydict["beta0"].squeeze(dim=-2).flatten().cpu().numpy(), columns=["beta_0"])

        # add all the rest to the dataframe
        df["cell_type"] = cell_types_names_kg.flatten()
        df["gene"] = gene_names_kg.flatten()
        df["eps"] = mydict["eps"].squeeze(dim=-2).flatten().cpu().numpy()

        return df

    def show_loss(self, figsize: Tuple[float, float] = (4, 4), logx: bool = False, logy: bool = False, ax=None):
        """
        Show the loss history. Useful for checking if the training has converged.

        Args:
            figsize: the size of the image. Used only if ax=None
            logx: if True the x_axis is shown in logarithmic scale
            logy: if True the x_axis is shown in logarithmic scale
            ax: The axes object to draw the plot onto. If None (defaults) creates a new figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self._loss_history)

        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")

        if logy:
            ax.set_yscale("log")
        else:
            ax.set_xscale("linear")

        if ax is None:
            fig.tight_layout()
            plt.close(fig)
            return fig

    def train(self,
              dataset: GeneDataset,
              n_steps: int = 2500,
              print_frequency: int = 50,
              use_covariates: bool = True,
              l1_regularization_strength: float = 0.1,
              l2_regularization_strength: float = None,
              eps_range: Tuple[float, float] = (1.0E-3, 1.0),
              subsample_size_cells: int = None,
              subsample_size_genes: int = None,
              initialization_type: str = "scratch",
              **kargs
              ):
        """
        Train the model. The trained parameter are stored in the pyro.param_store and
        can be accessed via :meth:`get_params`.

        Args:
            dataset: Dataset to train the model on
            n_steps: number of training step
            print_frequency: how frequently to print loss to screen
            use_covariates: if true, use covariates, if false use cell type information only
            l1_regularization_strength: controls the strength of the L1 regularization on the regression coefficients.
                If None there is no L1 regularization.
            l2_regularization_strength: controls the strength of the L2 regularization on the regression coefficients.
                If None there is no L2 regularization.
            eps_range: range of the possible values of the gene-specific noise. Must the a strictly positive range.
            subsample_size_genes: for large dataset, the minibatch can be created using a subset of genes.
            subsample_size_cells: for large dataset, the minibatch can be created using a subset of cells.
            initialization_type: Either "scratch", "pretrained" or "resume".
                If "resume" both the model and optimizer state are kept and training restart from where it was left off.
                If "pretrained" the model state is kept but the optimizer state is erased.
                If "scratch" (default) both the model and optimizer state are erased (i.e. simulation start from
                scratch).
            kargs: unused parameters

        Note:
            If you get an out-of-memory error try to tune the :attr:`subsample_size_cells`
            and :attr:`subsample_size_genes`.
        """

        if initialization_type == "scratch":
            print("training from scratch")
            pyro.clear_param_store()
            self._loss_history = []
            assert self.optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
            self.optimizer.set_state(self._optimizer_initial_state)
        elif initialization_type == "pretrained":
            print("training from pretrained model")
            self._loss_history = []
            assert self.optimizer is not None, "Optimizer is not specified. Call configure_optimizer first."
            self.optimizer.set_state(self._optimizer_initial_state)
        elif initialization_type == "resume":
            print("extending previous training")
        else:
            raise ValueError("Expected 'scratch' or 'pretrained' or 'resume'. Received {}".format(initialization_type))
        steps_completed = len(self._loss_history)

        # check validity
        assert l1_regularization_strength is None or l1_regularization_strength > 0.0
        assert l2_regularization_strength is None or l2_regularization_strength > 0.0
        assert not (l1_regularization_strength is not None and l2_regularization_strength is not None), \
            "You can NOT define both l1_regularization_strength and l2_regularization_strength."
        assert eps_range[1] > eps_range[0] > 0

        # prepare train kargs dict
        train_kargs = {
            'use_covariates': use_covariates,
            'l1_regularization_strength': l1_regularization_strength,
            'l2_regularization_strength': l2_regularization_strength,
            'eps_range': eps_range,
            'subsample_size_genes': subsample_size_genes,
            'subsample_size_cells': subsample_size_cells,
            'cell_type_mapping': dataset.cell_type_mapping,
            'gene_names': dataset.gene_names,
        }

        # make a copy so that can edit train_kargs without changing _train_kargs
        self._train_kargs = train_kargs.copy()

        # Unpack the dataset
        counts_ng = dataset.counts.long()
        cell_type_ids = dataset.cell_type_ids.long()
        total_umi_n = counts_ng.sum(dim=-1)

        # Figure out a good initialization for beta0 based on: counts = total_umi * beta0.exp()
        fraction_ng = counts_ng / total_umi_n.view(-1, 1)
        tmp_g = fraction_ng.mean(dim=0).log()
        beta0_g_init = torch.where(torch.isfinite(tmp_g), tmp_g, torch.zeros_like(tmp_g))  # remove nan if Any

        # Prepare arguments for training
        train_kargs["n_cells"] = counts_ng.shape[0]
        train_kargs["g_genes"] = counts_ng.shape[1]
        train_kargs["l_cov"] = dataset.covariates.shape[-1]
        train_kargs["k_cell_types"] = dataset.k_cell_types
        train_kargs["counts_ng"] = counts_ng.cpu()
        train_kargs["total_umi_n"] = total_umi_n.cpu()
        train_kargs["covariates_nl"] = dataset.covariates.float().cpu()
        train_kargs["cell_type_ids_n"] = cell_type_ids
        train_kargs["beta0_g_init"] = beta0_g_init.cpu()

        start_time = time.time()
        svi = SVI(self._model, self._guide, self.optimizer, loss=Trace_ELBO())
        for i in range(steps_completed+1, steps_completed + n_steps + 1):
            loss = svi.step(**train_kargs)
            self._loss_history.append(loss)
            if (i % print_frequency == 0) or (i == steps_completed+1):
                print('[iter {}]  loss: {:.4f}'.format(i, loss))
        print("Training completed in {} seconds".format(time.time()-start_time))

    @torch.no_grad()
    def predict(self,
                dataset: GeneDataset,
                num_samples: int = 10,
                subsample_size_cells: int = None,
                subsample_size_genes: int = None) -> (pd.DataFrame, pd.DataFrame):
        """
        Use the parameters currently in the param_store to run the prediction and report some metrics.
        If you want to run the prediction based on a different set of parameters you need
        to call :meth:`load_ckpt` first.

        The Q metric is :math:`Q = E\\left[|X_{i,g} - Y_{i,g}|\\right]`
        where `X` is the (observed) data and `Y` is a sample from the predicted posterior and `(i,g)`
        indicates cell and genes respectively.

        The log_score metric is :math:`\\text{log_score} = \\log p_\\text{posterior}\\left(X_\\text{data}\\right)`

        Args:
            dataset: the dataset to run the prediction on
            num_samples: how many random samples to draw from the predictive distribution
            subsample_size_cells: if not None (defaults) the prediction are made in chunks to avoid memory issue
            subsample_size_genes: if not None (defaults) the prediction are made in chunks to avoid memory issue

        Returns:
            df_metric: For each cell_type and gene we report the Q and log_score metrics
            df_counts: For each cell and gene we report the observed counts and a single sample from the posterior
        """
        n, g = dataset.counts.shape[:2]
        n, l = dataset.covariates.shape[:2]
        k = dataset.k_cell_types

        # params
        eps_kg1 = pyro.get_param_store().get_param("eps").float().cpu()
        beta0_kg1 = pyro.get_param_store().get_param("beta0").float().cpu()
        beta_kgl = pyro.get_param_store().get_param("beta").float().cpu()

        # dataset
        counts_ng = dataset.counts.long().cpu()
        cell_type_ids = dataset.cell_type_ids.long().cpu()
        covariates_n1l = dataset.covariates.unsqueeze(dim=-2).float().cpu()

        assert eps_kg1.shape == torch.Size([k, g, 1]), \
            "Got {0}. Are you predicting on the right dataset?".format(eps_k1g.shape)
        assert beta0_kg1.shape == torch.Size([k, g, 1]), \
            "Got {0}. Are you predicting on the right dataset?".format(beta0_k1g.shape)
        assert beta_kgl.shape == torch.Size([k, g, l]), \
            "Got {0}. Are you predicting on the right dataset?".format(beta_klg.shape)

        # prepare storage
        device_calculation = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        q_ng = torch.zeros((n, g), dtype=torch.float, device=torch.device("cpu"))
        mad_ng = torch.zeros((n, g), dtype=torch.float, device=torch.device("cpu"))
        pred_counts_ng = torch.zeros((n, g), dtype=torch.long, device=torch.device("cpu"))
        log_score_ng = torch.zeros((n, g), dtype=torch.float, device=torch.device("cpu"))

        # Loop to fill the predictions for all cell and genes
        subsample_size_cells = n if subsample_size_cells is None else subsample_size_cells
        subsample_size_genes = g if subsample_size_genes is None else subsample_size_genes

        for n_left in range(0, n, subsample_size_cells):
            n_right = min(n_left + subsample_size_cells, n)

            subn_cell_ids = cell_type_ids[n_left:n_right]
            subn_counts_ng = counts_ng[n_left:n_right]
            subn_covariates_n1l = covariates_n1l[n_left:n_right]
            subn_total_umi_n1 = subn_counts_ng.sum(dim=-1, keepdim=True)

            for g_left in range(0, g, subsample_size_genes):
                g_right = min(g_left + subsample_size_genes, g)

                eps_ng1 = eps_kg1[None,g_left:g_right,None][subn_cell_ids]
                beta0_ng1 = beta0_kg1[None,g_left:g_right,None][subn_cell_ids]
                log_rate_ng1 = beta0_ng1 + torch.sum(subn_covariates_n1l *
                                                     beta_kgl[None, g_left:g_right,None][subn_cell_ids],
                                                     dim=-1, keepdim=True)

                assert subn_total_umi_n1.shape == torch.Size([n_right-n_left, 1])
                assert log_rate_ng1.shape == torch.Size([n_right-n_left, g_right-g_left, 1])
                assert beta0_ng1.shape == torch.Size([n_right-n_left, g_right-g_left, 1])
                assert eps_ng1.shape == torch.Size([n_right-n_left, g_right-g_left, 1])

                mydist = LogNormalPoisson(
                    n_trials=subn_total_umi_n1.to(device_calculation),
                    log_rate=log_rate_ng1.squeeze(dim=-2).to(device_calculation),
                    noise_scale=eps_ng1.squeeze(dim=-2).to(device_calculation),
                    num_quad_points=8)

                subn_subg_counts_ng = subn_counts_ng[..., g_left:g_right].to(device_calculation)
                log_score_ng[n_left:n_right, g_left:g_right] = mydist.log_prob(subn_subg_counts_ng).cpu()

                # compute the Q metric, i.e. |x_obs - x_pred| averaged over the multiple posterior samples
                pred_counts_tmp_bng = mydist.sample(sample_shape=torch.Size([num_samples]))
                q_ng_tmp = (pred_counts_tmp_bng - subn_subg_counts_ng).abs().float().mean(dim=-3)
                q_ng[n_left:n_right, g_left:g_right] = q_ng_tmp.cpu()
                pred_counts_ng[n_left:n_right, g_left:g_right] = pred_counts_tmp_bng[0].long().cpu()
                
                # compute the MAD metric i.e. E_pred[|x_obs - x_pred|]
                lambda_ng = mydist.n_trials * mydist.log_rate
                mad_ng_tmp = (lambda_ng - subn_subg_counts_ng).abs().float().mean(dim=-3)
                mad_ng[n_left:n_right, g_left:g_right] = mad_ng_tmp.cpu()
                
                sample_1 = np.random.choice(subn_subg_counts_ng, size=100)
                sample_2 = np.random.choice(subn_subg_counts_ng, size=100)

                mad_ng_sample = np.abs(sample_1 - sample_2)
                mad_ng_mu = np.mean(mad_sample)
                mad_ng_std = np.std(mad_sample)

                mad_z = (mad_gn - mad_mu)/mad_std
                
                

        # average by cell_type to obtain q_prediction
        unique_cell_types = torch.unique(cell_type_ids)
        q_kg = torch.zeros((unique_cell_types.shape[0], g), dtype=torch.float, device=torch.device("cpu"))
        log_score_kg = torch.zeros((unique_cell_types.shape[0], g), dtype=torch.float, device=torch.device("cpu"))
        for k, cell_type in enumerate(unique_cell_types):
            mask = (cell_type_ids == cell_type)
            log_score_kg[k] = log_score_ng[mask].mean(dim=0)
            q_kg[k] = q_ng[mask].mean(dim=0)

        # Compute df_metric_kg
        # combine: gene_names_kg, cell_types_names_kg, q_kg, q_data_kg, log_score_kg into a dataframe
        cell_types_names_kg = self._get_cell_type_names_kg(g=len(dataset.gene_names))
        k_cell_types, len_genes = cell_types_names_kg.shape
        gene_names_kg = self._get_gene_names_kg(k=k_cell_types)
        assert gene_names_kg.shape == cell_types_names_kg.shape == q_kg.shape == log_score_kg.shape, \
            "Shape mismatch {0} vs {1} vs {2} vs {3}".format(gene_names_kg.shape,
                                                             cell_types_names_kg.shape,
                                                             q_kg.shape,
                                                             log_score_kg.shape)

        df_metric_kg = pd.DataFrame(cell_types_names_kg.flatten(), columns=["cell_type"])
        df_metric_kg["gene"] = gene_names_kg.flatten()
        df_metric_kg["q_dist"] = q_kg.flatten().cpu().numpy()
        df_metric_kg["log_score"] = log_score_kg.flatten().cpu().numpy()

        # Compute df_counts_ng
        cell_type_ids_ng = cell_type_ids.view(-1, 1).expand(n, g)
        cell_names_ng = numpy.array(list(self._get_inverse_cell_type_mapping().values()))[cell_type_ids_ng.cpu().numpy()]
        gene_names_ng = self._get_gene_names_kg(k=cell_names_ng.shape[0])
        assert cell_names_ng.shape == dataset.counts.shape == pred_counts_ng.shape == cell_names_ng.shape, \
            "Shape mismatch {0} vs {1} vs {2} vs {3}".format(cell_names_ng.shape,
                                                             dataset.counts.shape,
                                                             pred_counts_ng.shape,
                                                             cell_names_ng.shape)

        df_counts_ng = pd.DataFrame(pred_counts_ng.flatten().cpu().numpy(), columns=["counts_pred"])
        df_counts_ng["counts_obs"] = dataset.counts.flatten().cpu().numpy()
        df_counts_ng["cell_type"] = cell_names_ng.flatten()
        df_counts_ng["gene"] = gene_names_ng.flatten()

        # return
        return df_metric_kg, df_counts_ng
        

    def extend_train(
            self,
            dataset: GeneDataset,
            n_steps: int = 2500,
            print_frequency: int = 50):
        """
        Utility methods which calls :meth:`train` with the same parameter just used effectively extending the training.

        Args:
            dataset: Dataset to train the model on
            n_steps: number of training step
            print_frequency: how frequently to print loss to screen
        """

        self.train(
            dataset=dataset,
            n_steps=n_steps,
            print_frequency=print_frequency,
            initialization_type="resume",
            **self._train_kargs)

    def train_and_test(
            self,
            train_dataset: GeneDataset,
            test_dataset: GeneDataset,
            test_num_samples: int = 10,
            train_steps: int = 2500,
            train_print_frequency: int = 50,
            use_covariates: bool = True,
            l1_regularization_strength: float = 0.1,
            l2_regularization_strength: float = None,
            eps_range: Tuple[float, float] = (1.0E-3, 1.0),
            subsample_size_cells: int = None,
            subsample_size_genes: int = None,
            initialization_type: str = "scratch") -> (pd.DataFrame, pd.DataFrame):
        """
        Utility method which sequentially calls the methods :meth:`train` and :meth:`predict`.

        Args:
            train_dataset: Dataset to train the model on
            test_dataset: Dataset to run the prediction on
            test_num_samples: how many random samples to draw from the predictive distribution
            train_steps: number of training step
            train_print_frequency: how frequently to print loss to screen during training
            use_covariates: if true, use covariates, if false use cell type information only
            l1_regularization_strength: controls the strength of the L1 regularization on the regression coefficients.
                If None there is no L1 regularization.
            l2_regularization_strength: controls the strength of the L2 regularization on the regression coefficients.
                If None there is no L2 regularization.
            eps_range: range of the possible values of the gene-specific noise. Must the a strictly positive range.
            subsample_size_genes: for large dataset, the minibatch can be created using a subset of genes.
            subsample_size_cells: for large dataset, the minibatch can be created using a subset of cells.
            initialization_type: Either "scratch", "pretrained" or "resume".
                If "resume" both the model and optimizer state are kept and training restart from where it was left off.
                If "pretrained" the model state is kept but the optimizer state is erased.
                If "scratch" (default) both the model and optimizer state are erased (i.e. simulation start from
                scratch).

        Returns:
            metrics: See :meth:`predict`.
        """

        self.train(
            dataset=train_dataset,
            n_steps=train_steps,
            print_frequency=train_print_frequency,
            use_covariates=use_covariates,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            eps_range=eps_range,
            subsample_size_cells=subsample_size_cells,
            subsample_size_genes=subsample_size_genes,
            initialization_type=initialization_type)

        print("training completed")

        return self.predict(
            dataset=test_dataset,
            num_samples=test_num_samples,
            subsample_size_cells=subsample_size_cells,
            subsample_size_genes=subsample_size_genes)

    def extend_train_and_test(
            self,
            train_dataset: GeneDataset,
            test_dataset: GeneDataset,
            test_num_samples: int = 10,
            train_steps: int = 2500,
            train_print_frequency: int = 50) -> (pd.DataFrame, pd.DataFrame):
        """
        Utility method which sequentially calls the methods :meth:`extend_train` and :meth:`predict`.

        Args:
            train_dataset: Dataset to train the model on
            test_dataset: Dataset to run the prediction on
            test_num_samples: how many random samples to draw from the predictive distribution
            train_steps: number of training step
            train_print_frequency: how frequently to print loss to screen during training

        Returns:
            metrics: See :meth:`predict`.
        """

        self.extend_train(
            dataset=train_dataset,
            n_steps=train_steps,
            print_frequency=train_print_frequency)

        print("training completed")

        return self.predict(
            dataset=test_dataset,
            num_samples=test_num_samples,
            subsample_size_cells=self._train_kargs["subsample_size_cells"],
            subsample_size_genes=self._train_kargs["subsample_size_genes"])

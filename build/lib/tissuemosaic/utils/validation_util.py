import numpy
from typing import Union, Tuple
from umap.umap_ import UMAP
import leidenalg
import igraph as ig
import torch
import torch.nn.functional
import scipy
import scipy.sparse


def get_percentile(data: Union[torch.Tensor, numpy.ndarray], dim: int) -> Union[torch.Tensor, numpy.ndarray]:
    """
    Takes some data and convert it into a percentile (in [0.0, 1.0]) along a specified dimension.
    Useful to convert a tensor into the range [0.0, 1.0] for visualization.

    Args:
        data: input data to convert to percentile in [0,1].
        dim: the dimension along which to compute the quantiles

    Returns:
        percentile: torch.tensor or numpy.array (depending on the input type)
            with the same shape as the input with the percentile values.
            A percentile of 0.9 means that 90% of the input values were smaller.
    """
    assert isinstance(data, torch.Tensor) or isinstance(data, numpy.ndarray) or isinstance(data, list), \
        "Error. Input must be either a numpy.array or torch.Tensor or list. Received {0}".format(type(data))

    def _to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, numpy.ndarray):
            return torch.from_numpy(x)

    data_torch = _to_torch(data)

    old_dims = list(data_torch.shape)
    new_dims = [1] * len(old_dims)
    new_dims[dim] = old_dims[dim]

    indices = torch.argsort(data_torch, dim=dim, descending=False)
    src = torch.linspace(0.0, 1.0, new_dims[dim]).view(new_dims).expand_as(indices).clone()
    percentile = torch.zeros_like(data_torch)
    percentile.scatter_(dim=dim, index=indices, src=src)

    if isinstance(data, numpy.ndarray):
        return percentile.cpu().numpy()
    else:
        return percentile


def inverse_one_hot(image_in, bg_label: int = -1, dim: int = -3, threshold: float = 0.1):
    """
    Takes float tensor and compute the argmax and max_value along the specified dimension.
    Returns a integer tensor of the same shape as the input_tensor but with the :attr:`dim` removed.
    If the max_value is less than the threshold the bg_label is assigned.

    Note:
        It can take an image of size :math:`(C, W, H)` and generate an integer mask
        of size :math:`(W, H)`.
        This operation can be thought as the inverse of the one-hot operation which takes an integer tensor of size (n)
        and returns a float tensor with an extra dimension, for example (n, num_classes).

    Args:
        image_in: any float tensor
        bg_label: integer, the value assigned to the entries of which are smaller than the threshold
        dim: int, the dimension along which to compute the max.
            For images this is usually the channel dimension, i.e. -3.
        threshold: float, the value of the threshold. Value smaller than this are set assigned to the background

    Returns:
        out: An integer mask with the same size of the input tensor but with the dim removed.
    """
    assert isinstance(bg_label, int), "Error. bg_label must be an integer. Received {0}".format(bg_label)

    _values, _indices = torch.max(image_in, dim=dim)
    _mask_bg = (_values < threshold)
    _indices[_mask_bg] = bg_label
    return _indices.long()


def get_z_score(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standardize vector by removing the mean and scaling to unit variance

    Args:
        x: torch.Tensor
        dim: the dimension along which to compute the mean and std

    Return:
        The z-score, i.e. z = (x - mean) / std
    """
    std, mean = torch.std_mean(x, dim=dim, unbiased=True, keepdim=True)
    return (x-mean)/std


def compute_distance_embedding(ref_embeddings: torch.Tensor,
                               other_embeddings: torch.Tensor,
                               metric: str,
                               temperature: float = 0.5) -> torch.Tensor:
    """ Compute distance between embeddings

        Args:
            ref_embeddings: torch.Tensor of shape :math:`(*, k)` where `k`
                is the dimension of the embedding
            other_embeddings: torch.Tensor of shape :math:`(n, k)`
            temperature: float, the temperature used to compute contrastive distance
            metric: Can be either 'contrastive' or 'euclidean'

        Returns:
            dist: distance of shape :math:`(*, n)`
    """
    assert ref_embeddings.shape[-1] == other_embeddings.shape[-1]
    assert len(other_embeddings.shape) == 2
    assert len(ref_embeddings.shape) == 1 or len(ref_embeddings.shape) == 2

    if metric == "contrastive":
        # this is similar to the contrastive loss used to train SimClr model
        ref_embeddings_norm = torch.nn.functional.normalize(ref_embeddings, dim=-1)  # shape: (*, k)
        other_embeddings_norm = torch.nn.functional.normalize(other_embeddings, dim=-1)  # shape: (n, k)
        logits = torch.einsum('...c,nc->...n', ref_embeddings_norm, other_embeddings_norm)
        distance = torch.exp(-logits / temperature)  # Note: If similarity is high, distance is low, shape: (*, n)
        return distance

    elif metric == "euclidean":
        # ref_embeddings  # shape: (*, k)
        # other_embeddings  # shape: (n, k)
        delta = ref_embeddings.unsqueeze(dim=-2) - other_embeddings  # shape (*, n, k)
        distance2 = torch.einsum('...k, ...k -> ...', delta, delta)  # shape (*, n)
        return distance2.float().sqrt()

    elif metric == 'cosine':
        ref_embeddings_norm = torch.nn.functional.normalize(ref_embeddings, dim=-1)  # shape: (*, k)
        other_embeddings_norm = torch.nn.functional.normalize(other_embeddings, dim=-1)  # shape: (n, k)
        cosine = torch.einsum('...c,nc->...n', ref_embeddings_norm, other_embeddings_norm)
        return 1.0 + 1E-4 - cosine
    else:
        raise Exception("Wrong metric. Received {0}".format(metric))


class SmartUmap(UMAP):
    """ Wrapper around standard UMAP with :meth:`get_graph` exposed. """

    def __init__(self,
                 preprocess_strategy: str,
                 compute_all_pairwise_distances: bool = False,
                 **kargs):
        """
        Args:
            preprocess_strategy: str, can be 'center', 'z_score', 'raw'. This is the operation to perform before UMAP
            compute_all_pairwise_distances: bool, it True (default is False) compute all pairwise distances
            **kargs: All the arguments that standard UMAP can accept
        """

        assert preprocess_strategy == 'z_score' or \
               preprocess_strategy == 'center' or \
               preprocess_strategy == 'raw', "Preprocessing \
                must be either 'center', 'z_score' or 'raw'. Received {0}".format(preprocess_strategy)

        # overwrites some UMAP default values
        kargs['n_neighbors'] = kargs.get('n_neighbors', 25)
        kargs['min_dist'] = kargs.get('min_dist', 0.5)

        # inputs
        self.preprocess_strategy = preprocess_strategy
        self.compute_all_pairwise_distances = compute_all_pairwise_distances

        # thing computed during fit
        self._distances = None
        self._mean = None
        self._std = None
        self._fitted = False
        super(SmartUmap, self).__init__(random_state=0, **kargs)

    def get_graph(self) -> scipy.sparse.coo_matrix:
        """ Returns the symmetric (sparse) matrix with the SIMILARITIES between elements """
        return self.graph_

    def get_distances(self) -> torch.Tensor:
        """ Returns the symmetric (dense) matrix with the DISTANCES between elements """
        return self._distances

    def _compute_std_mean(self, data) -> (torch.Tensor, torch.Tensor):
        if self.preprocess_strategy == 'z_score':
            std, mean = torch.std_mean(data, dim=-2, unbiased=True, keepdim=True)
            # because Relu activation it is possible that std=0
            mask = (std == 0.0)
            std[mask] = 1.0
        elif self.preprocess_strategy == 'center':
            mean = torch.mean(data, dim=-2, keepdim=True)
            std = torch.ones_like(mean)
        else:
            # this is the case self.preprocess_strategy == 'raw'
            mean = torch.zeros_like(data[0, :])
            std = torch.ones_like(mean)

        return std, mean

    def fit(self, data, y=None) -> "SmartUmap":
        """
        Fit the Umap given the data

        Args:
            data: array of shape :math:`(n, p)` where `n` are the points and `p` the features
        """
        assert y is None
        if isinstance(data, numpy.ndarray):
            data_new = torch.tensor(data).clone().float()
        else:
            data_new = data.clone().float()

        std, mean = self._compute_std_mean(data_new)
        self._mean = mean
        self._std = std
        self._fitted = True
        data_new = (data_new - mean) / std
        return super(SmartUmap, self).fit(data_new.detach().cpu().numpy(), y)

    def transform(self, data) -> numpy.ndarray:
        """
        Use previously fitted model (including mean and std for centering and scaling the data).
        to transform the embeddings.

        Args:
            data: array of shape :math:`(n, p)` to transfrom

        Returns:
            embeddings: numpy.tensor of shape (n_sample, n_components)
        """

        assert self._fitted, "UMAP is not fitted. Cal 'fit' or 'fit_transform' first."

        if isinstance(data, numpy.ndarray):
            data_new = torch.tensor(data).clone().float()
        else:
            data_new = data.clone().float()

        data_new = (data_new - self._mean.to(data_new.device)) / self._std.to(data_new.device)
        embeddings = super(SmartUmap, self).transform(data_new.detach().cpu().numpy())

        if self.compute_all_pairwise_distances:
            self._distances = compute_distance_embedding(
                ref_embeddings=data_new,
                other_embeddings=data_new,
                metric=self.metric).cpu().numpy()

        return embeddings

    def fit_transform(self, data, y=None) -> numpy.ndarray:
        """ Utility method which internally calls :meth:`fit` and :meth:`transform` """
        self.fit(data)
        return self.transform(data)


class SmartLeiden:
    """
    Wrapper around standard Leiden algorithm.
    It can be initialized using the output of the :class:`SmartUmap.get_graph()`
    """

    def __init__(self, graph: "coo_matrix", directed: bool = True):
        """
        Args:
            graph: Usually a sparse matrix with the similarities among nodes describing the graph
            directed: if True (default) builds a directed graph.

        Note:
            The matrix obtained by the UMAP algorithm is symmetric, in that case directed should be set to True
        """
        sources, targets = graph.nonzero()
        weights = graph[sources, targets]
        if isinstance(weights, numpy.matrix):
            weights = weights.A1
        self.ig_graph = ig.Graph(directed=directed)
        self.ig_graph.add_vertices(graph.shape[0])  # this adds adjacency.shape[0] vertices
        self.ig_graph.add_edges(list(zip(sources, targets)))
        self.ig_graph.es['weight'] = weights

    def cluster(self,
                resolution: float = 1.0,
                use_weights: bool = True,
                random_state: int = 0,
                n_iterations: int = -1,
                partition_type: str = 'RBC') -> numpy.ndarray:
        """
        Find the clusters in the data

        Args:
            resolution: resolution parameter controlling (indirectly) the number of clusters
            use_weights: if True (defaults) the graph is weighted, i.e. the edges have different strengths
            random_state: control the random state. For reproducibility
            n_iterations: how many iterations of the greedy algorithm to perform.
                If -1 (defaults) it iterates till convergence.
            partition_type: The metric to optimize to find clusters. Either 'CPM' or 'RBC'. :

        Returns:
            labels: the integer cluster labels
        """

        partition_kwargs = {
            'resolution_parameter': resolution,
            'n_iterations': n_iterations,
            'seed': random_state,
        }

        if use_weights:
            partition_kwargs['weights'] = numpy.array(self.ig_graph.es['weight']).astype(numpy.float64)

        if partition_type == 'CPM':
            partition_type = leidenalg.CPMVertexPartition
        elif partition_type == 'RBC':
            partition_type = leidenalg.RBConfigurationVertexPartition
        else:
            raise Exception("Partitin type not recognized. Expected CPM or RBC. Received {0}".format(partition_type))

        part = leidenalg.find_partition(self.ig_graph, partition_type, **partition_kwargs)

        return numpy.array(part._membership).astype(int)


class SmartPca:
    """ Return the PCA embeddings.  """

    def __init__(self,
                 preprocess_strategy: str):
        """
        Args:
            preprocess_strategy: str, can be 'center', 'z_score', 'raw'.
                This is the operation to perform before PCA
        """

        assert preprocess_strategy == 'z_score' or \
               preprocess_strategy == 'center' or \
               preprocess_strategy == 'raw', "Preprocessing \
        must be either 'center', 'z_score' or 'raw'. Received {0}".format(preprocess_strategy)

        self.preprocess_strategy = preprocess_strategy
        self._fitted = False
        self._n_components = None
        self._V = None  # matrix of shape (p,q) i.e. (features, reduced_dimension)
        self._eigen_cov_matrix = None
        self._mean = None
        self._std = None

    @property
    def explained_variance_(self):
        """ For compatibility with scikit_learn """
        return self._eigen_cov_matrix

    @property
    def explained_variance_ratio_(self):
        """ For compatibility with scikit_learn """
        tmp = self.explained_variance_.cumsum(dim=-1)
        return tmp / tmp[-1]

    def _compute_std_mean(self, data) -> (torch.Tensor, torch.Tensor):
        if self.preprocess_strategy == 'z_score':
            std, mean = torch.std_mean(data, dim=-2, unbiased=True, keepdim=True)
            # because Relu activation it is possible that std=0
            mask = (std == 0.0)
            std[mask] = 1.0
        elif self.preprocess_strategy == 'center':
            mean = torch.mean(data, dim=-2, keepdim=True)
            std = torch.ones_like(mean)
        else:
            # this is the case self.preprocess_strategy == 'raw'
            mean = torch.zeros_like(data[0, :])
            std = torch.ones_like(mean)

        return std, mean

    def _apply_scaling(self, data):
        self._mean = self._mean.to(device=data.device, dtype=data.dtype)
        self._std = self._std.to(device=data.device, dtype=data.dtype)
        return (data - self._mean) / self._std

    def _get_q(self, n_components: Union[int, float], p: int) -> int:
        if isinstance(n_components, int) and (0 < n_components <= p):
            return n_components
        elif isinstance(n_components, float) and (0.0 < n_components <= 1.0):
            indicator = (self.explained_variance_ratio_ > n_components)
            values, counts = torch.unique_consecutive(indicator, return_counts=True)
            return counts[0]
        else:
            raise Exception("n_components needs to be an integer in (0, {0}] or a float in (0.0, 1.0). \
            Received {1}".format(p, n_components))

    def fit(self, data) -> "SmartPca":
        """
        Fit the PCA given the data.
        It automatically select the algorithm based on the number of features.

        Args:
            data: array of shape :math:`(n, p)` where `n` are the points and `p` the features
        """
        if isinstance(data, numpy.ndarray):
            data_new = torch.tensor(data).clone().float()
        else:
            data_new = data.clone().float()  # upgrade to full precision in case you are at Half

        std, mean = self._compute_std_mean(data_new)
        self._std = std
        self._mean = mean
        data_new = self._apply_scaling(data_new)

        n, p = data_new.shape
        # In case some of the inputs are not finite. This should never happen
        mask = torch.isfinite(data_new)
        data_new[~mask] = 0.0

        q = min(p, n)
        if p <= 2500:
            # Use the covariance method, i.e. p x p matrix
            cov = torch.einsum('np,nq -> pq', data_new, data_new) / (n - 1)  # (p x p) covariance matrix
            # add a small diagonal term to make sure that the covariance matrix is not singular
            eps = 1E-4 * torch.randn(p, dtype=cov.dtype, device=cov.device)
            cov += torch.diag(eps)
            assert cov.shape == torch.Size([p, p])

            try:
                U, S, _ = torch.svd(cov)
            except:   # torch.svd may have convergence issues for GPU and CPU.
                U_np, S_np, _ = scipy.linalg.svd(cov.detach().cpu().numpy(), lapack_driver="gesdd")  # works
                U = torch.from_numpy(U_np).to(dtype=cov.dtype, device=cov.device)
                S = torch.from_numpy(S_np).to(dtype=cov.dtype, device=cov.device)
            self._eigen_cov_matrix = S[:q]  # shape (q)
            self._V = U[:, :q]  # shape (p, q)
        else:
            # Use the approximate random method
            U, S, V = torch.pca_lowrank(data_new, center=False, niter=2, q=q)
            assert U.shape == torch.Size([n, q]), "U.shape {0}".format(U.shape)
            assert S.shape == torch.Size([q]), "S.shape {0}".format(S.shape)
            assert V.shape == torch.Size([p, q]), "V.shape {0}".format(V.shape)
            # dist = torch.dist(data_new, U @ torch.diag(S) @ V.permute(1, 0))
            # print("{0} check the low_rank_PCA -> {1}".format(n, dist))
            eigen_cov_matrix = S.pow(2) / (n-1)
            self._eigen_cov_matrix = eigen_cov_matrix  # shape (q)
            self._V = V  # shape (p, q)

        self._fitted = True
        return self

    def transform(self, data, n_components: Union[int, float] = None) -> numpy.ndarray:
        """
        Use a previously fitted model to transform the data.

        Args:
            data: tensor of shape :math:`(n, p)` where `n` is the number of points and `p` are the features
            n_components: If integer specifies the dimensionality of the data after PCA. If float in (0, 1)
                it auto selects the dimensionality so that the explained variance is at least that value.
                If none it uses the previously used value.
        """
        assert self._fitted, "PCA is not fitted. Cal 'fit' or 'fit_transform' first."
        if isinstance(data, numpy.ndarray):
            data_new = torch.tensor(data).clone().float()
        else:
            data_new = data.clone().float()

        data_new = self._apply_scaling(data_new)
        if n_components is None and self._n_components is not None:
            q = self._n_components  # use the previously defined value
        elif n_components is None and self._n_components is None:
            raise Exception("n_components has never been specified. Expected n_components = Union[ont, float]")
        else:
            q = self._get_q(n_components, p=data_new.shape[-1])
            self._n_components = q

        self._V = self._V.to(device=data_new.device, dtype=data_new.dtype)
        return torch.einsum('np,pq -> nq', data_new, self._V[:, :q]).cpu().numpy()

    def fit_transform(self, data, n_components: Union[int, float] = None) -> numpy.ndarray:
        """
        Utility method which internally calls :meth:`fit` and :meth:`transform`.

        Args:
            data: tensor of shape :math:`(n, p)`
            n_components: If integer specifies the dimensionality of the data after PCA. If float in (0, 1)
                it auto selects the dimensionality so that the explained variance is at least that value.
                If none (defaults) uses the value previously used.

        Returns:
            data_transformed: array of shape :math:`(n, q)`
        """
        self.fit(data)
        return self.transform(data, n_components)


class SmartScaler:
    """
    Scale the values using the median and quantiles (with are robust version of mean and variance).
    :math:`data = (data - median) / scale`

    If clamp=True, each feature is clamped to the quantile range before applying the transformation.
    This is a simple way to deal with the outliers.

    It does not deal with the situation in which outliers are inside the "box" of acceptable range
    but far from the reduced manifold. 
    # See situation shown below:
    #              x  x
    #         x x
    #    x x         o
    # x x
    """

    def __init__(self, quantiles: Tuple[float, float], clamp: bool):
        """
        Args:
            quantiles: The lowest and largest quantile used to scale the data. Must be in (0.0, 1.0)
            clamp: If True, the data is clamped into q_low, q_high before scaling.
        """

        # Inputs
        self._q_low = quantiles[0]
        self._q_high = quantiles[1]
        self._clamp = clamp

        # Things computed during fit
        self._fitted = False
        self._median = None
        self._low = None
        self._high = None

    def _compute_median_quantile(self, data) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        median = torch.median(data, dim=-2)[0]
        q_tensor = torch.tensor([self._q_low, self._q_high], dtype=data.dtype, device=data.device)
        qs = torch.quantile(data, q=q_tensor, dim=-2)
        return qs[0], qs[1], median

    def fit(self, data) -> "SmartScaler":
        """ Fit the data (i.e. computes quantiles and median) """
        if isinstance(data, numpy.ndarray):
            data = torch.tensor(data).clone().float()
        else:
            data = data.clone().float()

        low: torch.Tensor
        high: torch.Tensor
        median: torch.Tensor
        low, high, median = self._compute_median_quantile(data)
        test: torch.Tensor = (low < high)
        assert torch.all(test).item()

        self._high = high
        self._low = low
        self._median = median
        self._fitted = True
        return self

    def _apply_scaling(self, data):
        scale = (self._high - self._low).to(device=data.device, dtype=data.dtype)
        median = self._median.to(device=data.device, dtype=data.dtype)
        return (data - median) / scale

    def transform(self, data) -> numpy.ndarray:
        """
        Transform the data

        Args:
            data: tensor of shape :math:`(n, p)`

        Returns:
            out: tensor of the same shape as :attr:`data` with the scaled values.
        """
        assert self._fitted, "Scaler is not fitted. Cal 'fit' or 'fit_transform' first."

        if isinstance(data, numpy.ndarray):
            data_new = torch.tensor(data).clone().float()
        else:
            data_new = data.clone().float()

        if self._clamp:
            data_new = data.clamp(min=self._low.to(data_new.device),
                                  max=self._high.to(data_new.device))
        return self._apply_scaling(data_new).cpu().numpy()

    def fit_transform(self, data) -> numpy.ndarray:
        """ Utility method which internally calls :meth:`fit` and :meth:`transform` """
        self.fit(data)
        return self.transform(data)

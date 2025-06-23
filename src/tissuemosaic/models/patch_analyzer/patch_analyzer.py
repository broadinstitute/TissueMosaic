from __future__ import annotations
from typing import Union, List, Optional, Tuple, TYPE_CHECKING
import numpy
import torch
from sklearn.neighbors import NearestNeighbors

# trick to avoid circular imports
if TYPE_CHECKING:
    from tissuemosaic.data.sparse_image import SparseImage


# class SpatialAutocorrelation:
#     """
#     Compute the Moran's I or Geary's C score of a sparse torch tensor. If the sparse tensor has `ch` channels
#     it will produce `ch` scores, each indicating how each channel is dispersed in all the others.

#     Note:
#         The results of a Moran's I and Geary's C tests depend on the choice of the weight matrix
#         (see `Moran <https://en.wikipedia.org/wiki/Moran%27s_I>`_ and
#         `Geary <https://en.wikipedia.org/wiki/Geary%27s_C>`_ for details).
#         The input parameters determine the strategy used for the construction of the weight matrix.
#     """

#     def __init__(self,
#                  modality: str = 'moran',
#                  n_neighbours: Optional[int] = None,
#                  radius: Optional[float] = None,
#                  neigh_correct: bool = True):

#         """
#         Args:
#             modality: string, either "moran" or "geary"
#             n_neighbours: if specified the weight matrix will be 1 for the first n_neighbours and zero otherwise.
#                 A value often used is 6.
#             radius: if specified the weight matrix will be 1 for distance < radius and zero otherwise
#             neigh_correct: if True, only the neighbours with distance less than 1.3 median distance are considered.
#         """
#         mylist = [n_neighbours is not None, radius is not None]
#         assert sum(mylist) == 1, "One and only one among n_neighbours, radius need to be specified"
#         assert modality == "geary" or modality == "moran"

#         self.modality = modality
#         self.n_neighbours = n_neighbours
#         self.radius = radius
#         self.neigh_correct = neigh_correct

#     @torch.no_grad()
#     def __call__(self, data: Union[SparseImage, List[SparseImage], torch.sparse.Tensor, List[torch.sparse.Tensor]]):
#         """
#         Args:
#             data: A (list of) sparse tensor with `C` channels

#         Returns:
#             score: A (list of) torch.tensor of size `C` with the score (either moran or gready)
#                 indicating how each channel is dispersed in all the others.
#         """

#         result = []
#         mylist = data if isinstance(data, List) else [data]

#         for n, sparse_tensor in enumerate(mylist):
#             # if torch.cuda.is_available():
#             #     sparse_tensor = sparse_tensor.cuda()

#             n_category = sparse_tensor.shape[0]

#             # print("sparse tensor shape:")
#             # print(sparse_tensor.shape)
            
# #             if sparse_tensor.values().shape[0] > self.n_neighbours:
# #                 category, coords = self._extract_category_and_coords(sparse_tensor=sparse_tensor)
# #                 adj = self._build_connectivity(coords=coords, remove_diagonal=True)

# #                 score = self._compute_autocorrelation(category=category, adj=adj, n_category=n_category)
            
# #             else:
# #                 #print(sparse_tensor.values().shape)
# #                 score = torch.zeros(n_category)#.device('cuda:0') ## dummy score
# #                 print("reached")
            
#             ## why is category shape 1 in some cases
#             #if category.shape[0] >= self.n_neighbours:
                
#             ## assert that n_element_min > n_neighbours?
#             category, coords = self._extract_category_and_coords(sparse_tensor=sparse_tensor)
#             adj = self._build_connectivity(coords=coords, remove_diagonal=True)
#             score = self._compute_autocorrelation(category=category, adj=adj, n_category=n_category)
#             result.append(score)

#         return result if len(result) > 1 else result[0]

#     @staticmethod
#     def _extract_category_and_coords(sparse_tensor: torch.sparse.Tensor):
#         """ Extract the category and the coordinates. If there are more that one object at the same x,y location
#             I need to add a little bit of noise to its coordinates """
#         original_device = sparse_tensor.device
#         if torch.cuda.is_available():
#             sparse_tensor = sparse_tensor.cuda()

#         #print(sparse_tensor.indices())
#         category_tmp, x_pixel, y_pixel = sparse_tensor.indices()
#         #print(category_tmp) ## take cell type with highest proportion as cell-type label for computing moran's
#         values = sparse_tensor.values()

#         values = values.type(torch.int64) ## delete this?
        
# #         print("dim:")
# #         print(category_tmp.shape)
# #         print(x_pixel.shape)
# #         print(y_pixel.shape)
        
# #         print("values:")
# #         print(values.shape)
# #         #print(values.max())
#         #print(sparse_tensor)
#         #print(values.max().item())
        
#         # if values.max().item() > 1.0:
#         #     import pdb; pdb.set_trace()
#         #print(values)
        
#         # print("values shape:")
#         # print(values.shape)
        
        
#         ## fig out how to deal for pcm where values are < 1?
#         if values.max().item() <= 1:
#             coords = torch.stack((x_pixel, y_pixel), dim=-1)
#             category = category_tmp
#         else:
#             #  If I have more that one object at the same x,y location I add a little bit of noise to its coordinates <--?
#             x_list, y_list, category_list = [], [], []
#             for i in range(1, values.max().item() + 1):
#                 mask = (values == i)

#                 if i == 1:
#                     x_list.append(x_pixel[mask])
#                     y_list.append(y_pixel[mask])
#                     category_list.append(category_tmp[mask])
#                 else:
#                     noise_x = 1E-3 * torch.rand(size=[i], dtype=torch.float, device=x_pixel.device).view(i, 1)
#                     noise_y = 1E-3 * torch.rand(size=[i], dtype=torch.float, device=x_pixel.device).view(i, 1)
#                     x_tmp = x_pixel[mask] + noise_x  # shape: (i, n_tmp)
#                     y_tmp = y_pixel[mask] + noise_y  # shape: (i, n_tmp)
#                     cat_tmp = category_tmp[mask].view(1, -1).expand_as(x_tmp)  # shape: (i, n_tmp)

#                     x_list.append(x_tmp.flatten())
#                     y_list.append(y_tmp.flatten())
#                     category_list.append(cat_tmp.flatten())

#             x_tmp = torch.cat(x_list, dim=0)
#             y_tmp = torch.cat(y_list, dim=0)
#             coords = torch.stack((x_tmp, y_tmp), dim=1)
#             category = torch.cat(category_list, dim=0)

#         return category.to(device=original_device), coords.to(device=original_device)

#     def _compute_autocorrelation(self, category, adj, n_category: int):
#         assert category.shape[0] == adj.shape[0] == adj.shape[1]
#         original_device = category.device

#         if torch.cuda.is_available():
#             category = category.cuda()
#             adj = adj.cuda()

#         N = category.shape[0]
#         col_index, row_index = adj.indices()
#         w = adj.values()
#         W = w.sum().item()

#         ncount = torch.bincount(category, minlength=n_category)
#         scores = torch.zeros(size=[n_category], dtype=torch.float, device=category.device)
#         for i in range(ncount.shape[0]):

#             x = (category == i).float()  # either 0 or 1
#             z = x - x.mean()
#             denominator = (z * z).sum()

#             if self.modality == 'moran':
#                 numerator = (w * z[col_index] * z[row_index]).sum()
#                 scores[i] = (N * numerator) / (W * denominator)
#             else:
#                 dz = z[col_index] - z[row_index]
#                 numerator = (w * dz * dz).sum()
#                 scores[i] = ((N - 1) * numerator) / (2 * W * denominator)

#             # this calculation gives Nan if all x are of the same category (b/c division by zero).
#             # Here I deal with this edge case
#             if denominator == 0.0:
#                 scores[i] = 0.0
        
#         return scores.to(device=original_device)

#     def _build_connectivity(self, coords: torch.Tensor, remove_diagonal: bool):
#         # TODO: Improve _build_connectivity by keeping into account that some channels might be much more
#         #  sparse than others and therefore should have a longer range connectivity
#         original_device = coords.device
#         N = coords.shape[0]
#         tree = NearestNeighbors(n_neighbors=self.n_neighbours or 6, radius=self.radius or 1, metric="euclidean")
#         tree.fit(coords.cpu().numpy())

#         if self.radius is not None:
#             results = tree.radius_neighbors()
#             dists = numpy.concatenate(results[0])
#             row_indices = numpy.concatenate(results[1])
#             lengths = [len(x) for x in results[1]]
#             col_indices = numpy.repeat(numpy.arange(N), lengths)
#         else:
#             results = tree.kneighbors()
#             dists, row_indices = (result.reshape(-1) for result in results)
#             col_indices = numpy.repeat(numpy.arange(N), self.n_neighbours or 6)
#             if self.neigh_correct:
#                 dist_cutoff = numpy.median(dists) * 1.3  # There's a small amount of sway
#                 mask = dists < dist_cutoff
#                 row_indices, col_indices = row_indices[mask], col_indices[mask]
#                 dists = dists[mask]

#         device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         torch_row_indices = torch.tensor(row_indices, dtype=torch.long, device=device)
#         torch_col_indices = torch.tensor(col_indices, dtype=torch.long, device=device)

#         if remove_diagonal:
#             mask_not_diagonal = (torch_row_indices != torch_col_indices)
#             torch_row_indices = torch_row_indices[mask_not_diagonal]
#             torch_col_indices = torch_col_indices[mask_not_diagonal]

#         adj = torch.sparse_coo_tensor(
#             indices=torch.stack((torch_row_indices, torch_col_indices), dim=0),
#             values=torch.ones_like(torch_row_indices).int(),
#             size=(N, N),
#             requires_grad=False,
#             device=device).coalesce()

#         # symmetrize the adjency matrix since adj created using knn does NOT need to be symmetric
#         return 0.5 * (adj + adj.transpose(dim0=-1, dim1=-2)).coalesce().to(original_device)


class SpatialAutocorrelation:
    """
    Compute the Moran's I or Geary's C score of a sparse torch tensor. If the sparse tensor has `ch` channels
    it will produce `ch` scores, each indicating how each channel is dispersed in all the others.

    Note:
        The results of a Moran's I and Geary's C tests depend on the choice of the weight matrix
        (see `Moran <https://en.wikipedia.org/wiki/Moran%27s_I>`_ and
        `Geary <https://en.wikipedia.org/wiki/Geary%27s_C>`_ for details).
        The input parameters determine the strategy used for the construction of the weight matrix.
    """

    def __init__(self,
                 modality: str = 'moran',
                 n_neighbours: Optional[int] = None,
                 radius: Optional[float] = None,
                 neigh_correct: bool = True):

        """
        Args:
            modality: string, either "moran" or "geary"
            n_neighbours: if specified the weight matrix will be 1 for the first n_neighbours and zero otherwise.
                A value often used is 6.
            radius: if specified the weight matrix will be 1 for distance < radius and zero otherwise
            neigh_correct: if True, only the neighbours with distance less than 1.3 median distance are considered.
        """
        mylist = [n_neighbours is not None, radius is not None]
        assert sum(mylist) == 1, "One and only one among n_neighbours, radius need to be specified"
        assert modality == "geary" or modality == "moran"

        self.modality = modality
        self.n_neighbours = n_neighbours
        self.radius = radius
        self.neigh_correct = neigh_correct

    @torch.no_grad()
    def __call__(self, data: Union[SparseImage, List[SparseImage], torch.sparse.Tensor, List[torch.sparse.Tensor]]):
        """
        Args:
            data: A (list of) sparse tensor with `C` channels

        Returns:
            score: A (list of) torch.tensor of size `C` with the score (either moran or gready)
                indicating how each channel is dispersed in all the others.
        """

        result = []
        mylist = data if isinstance(data, List) else [data]

        for n, sparse_tensor in enumerate(mylist):
            # if torch.cuda.is_available():
            #     sparse_tensor = sparse_tensor.cuda()

            n_category = sparse_tensor.shape[0]
                
            ## assert that n_element_min > n_neighbours?
            category, coords = self._extract_category_and_coords(sparse_tensor=sparse_tensor)
            adj = self._build_connectivity(coords=coords, remove_diagonal=True)
            score = self._compute_autocorrelation(category=category, adj=adj, n_category=n_category)
            result.append(score)

        return result if len(result) > 1 else result[0]

    @staticmethod
    def _extract_category_and_coords(sparse_tensor: torch.sparse.Tensor):
        """ Extract the category and the coordinates. If there are more that one object at the same x,y location
            I need to add a little bit of noise to its coordinates """
        original_device = sparse_tensor.device
        if torch.cuda.is_available():
            sparse_tensor = sparse_tensor.cuda()

        #print(sparse_tensor.indices())
        category_tmp, x_pixel, y_pixel = sparse_tensor.indices()
        #print(category_tmp) ## take cell type with highest proportion as cell-type label for computing moran's
        values = sparse_tensor.values()

        # print(values)

        ## better way to deal for pcm where values are < 1?
        values[values > 0.5] = 1

        values = values.type(torch.int64)

        # print(values) 
        
        
        
        if values.max().item() <= 1:
            coords = torch.stack((x_pixel, y_pixel), dim=-1)
            category = category_tmp
        else:
            #  If I have more that one object at the same x,y location I add a little bit of noise to its coordinates <--?
            x_list, y_list, category_list = [], [], []
            for i in range(1, values.max().item() + 1):
                mask = (values == i)

                if i == 1:
                    x_list.append(x_pixel[mask])
                    y_list.append(y_pixel[mask])
                    category_list.append(category_tmp[mask])
                else:
                    noise_x = 1E-3 * torch.rand(size=[i], dtype=torch.float, device=x_pixel.device).view(i, 1)
                    noise_y = 1E-3 * torch.rand(size=[i], dtype=torch.float, device=x_pixel.device).view(i, 1)
                    x_tmp = x_pixel[mask] + noise_x  # shape: (i, n_tmp)
                    y_tmp = y_pixel[mask] + noise_y  # shape: (i, n_tmp)
                    cat_tmp = category_tmp[mask].view(1, -1).expand_as(x_tmp)  # shape: (i, n_tmp)

                    x_list.append(x_tmp.flatten())
                    y_list.append(y_tmp.flatten())
                    category_list.append(cat_tmp.flatten())

            x_tmp = torch.cat(x_list, dim=0)
            y_tmp = torch.cat(y_list, dim=0)
            coords = torch.stack((x_tmp, y_tmp), dim=1)
            category = torch.cat(category_list, dim=0)

        return category.to(device=original_device), coords.to(device=original_device)

    def _compute_autocorrelation(self, category, adj, n_category: int):
        assert category.shape[0] == adj.shape[0] == adj.shape[1]
        original_device = category.device

        if torch.cuda.is_available():
            category = category.cuda()
            adj = adj.cuda()

        N = category.shape[0]
        col_index, row_index = adj.indices()
        w = adj.values()
        W = w.sum().item()

        ncount = torch.bincount(category, minlength=n_category)
        scores = torch.zeros(size=[n_category], dtype=torch.float, device=category.device)
        for i in range(ncount.shape[0]):

            x = (category == i).float()  # either 0 or 1
            z = x - x.mean()
            denominator = (z * z).sum()

            if self.modality == 'moran':
                numerator = (w * z[col_index] * z[row_index]).sum()
                scores[i] = (N * numerator) / (W * denominator)
            else:
                dz = z[col_index] - z[row_index]
                numerator = (w * dz * dz).sum()
                scores[i] = ((N - 1) * numerator) / (2 * W * denominator)

            # this calculation gives Nan if all x are of the same category (b/c division by zero).
            # Here I deal with this edge case
            if denominator == 0.0:
                scores[i] = 0.0
        
        return scores.to(device=original_device)

    def _build_connectivity(self, coords: torch.Tensor, remove_diagonal: bool):
        # TODO: Improve _build_connectivity by keeping into account that some channels might be much more
        #  sparse than others and therefore should have a longer range connectivity
        original_device = coords.device
        N = coords.shape[0]
        tree = NearestNeighbors(n_neighbors=self.n_neighbours or 6, radius=self.radius or 1, metric="euclidean")
        tree.fit(coords.cpu().numpy())

        if self.radius is not None:
            results = tree.radius_neighbors()
            dists = numpy.concatenate(results[0])
            row_indices = numpy.concatenate(results[1])
            lengths = [len(x) for x in results[1]]
            col_indices = numpy.repeat(numpy.arange(N), lengths)
        else:
            results = tree.kneighbors()
            dists, row_indices = (result.reshape(-1) for result in results)
            col_indices = numpy.repeat(numpy.arange(N), self.n_neighbours or 6)
            if self.neigh_correct:
                dist_cutoff = numpy.median(dists) * 1.3  # There's a small amount of sway
                mask = dists < dist_cutoff
                row_indices, col_indices = row_indices[mask], col_indices[mask]
                dists = dists[mask]

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch_row_indices = torch.tensor(row_indices, dtype=torch.long, device=device)
        torch_col_indices = torch.tensor(col_indices, dtype=torch.long, device=device)

        if remove_diagonal:
            mask_not_diagonal = (torch_row_indices != torch_col_indices)
            torch_row_indices = torch_row_indices[mask_not_diagonal]
            torch_col_indices = torch_col_indices[mask_not_diagonal]

        adj = torch.sparse_coo_tensor(
            indices=torch.stack((torch_row_indices, torch_col_indices), dim=0),
            values=torch.ones_like(torch_row_indices).int(),
            size=(N, N),
            requires_grad=False,
            device=device).coalesce()

        # symmetrize the adjency matrix since adj created using knn does NOT need to be symmetric
        return 0.5 * (adj + adj.transpose(dim0=-1, dim1=-2)).coalesce().to(original_device)


class Composition:
    """ Counts the number of elements in every channel and return their raw values or their normalized frequencies. """

    def __init__(self, return_fraction: bool = True):
        """
        Args:
            return_fraction: if True (defaults) it returns the normalized frequency instead of the raw values.
        """
        self.return_fraction = return_fraction

    def __call__(
            self,
            data: Union[torch.Tensor, torch.sparse.Tensor, SparseImage,
                        List[torch.sparse.Tensor], List[SparseImage]],
            windows: Tuple[float, float, float, float] = None) -> torch.Tensor:
        """
        Count the intensity for each channel in a 2D window.

        Args:
            data: torch.Tensor or torch.sparse.Tensor or SparseImage (or list thereof)
                corresponding to a spatial data of shape :math:`(C, W, H)`
            windows: tuple with (min_row, min_col, max_row, max_col). If None (default) the entire image is considered.

        Returns:
            composition: A vector of size `C` with the count for each channel (or list thereof).
        """
        if not isinstance(data, list):
            data = [data]

        result = []

        for tmp_data in data:
            original_device = tmp_data.device
            if torch.cuda.is_available():
                tmp_data = tmp_data.cuda()

            if tmp_data.is_sparse:
                codes, ix, iy = tmp_data.indices()
                values = tmp_data.values()
                n_channels = tmp_data.shape[-3]
                counter = torch.zeros(n_channels, dtype=values.dtype,
                                      device=values.device)  # from 0 to max_channel included

                if windows is not None:
                    min_row, min_col, max_row, max_col = windows
                    mask_reduce = (ix >= min_row) * (ix < max_row) * (iy >= min_col) * (iy < max_col)
                    codes_reduced = codes[mask_reduce]
                    values_reduced = values[mask_reduce]
                else:
                    codes_reduced = codes
                    values_reduced = values

                for n in range(n_channels):
                    mask = (codes_reduced == n)
                    counter[n] = values_reduced[mask].sum()

            else:
                assert len(tmp_data.shape) >= 3
                if windows is not None:
                    min_row, min_col, max_row, max_col = windows
                    w, h = tmp_data.shape[-2]
                    x_grid, y_grid = torch.meshgrid(
                        torch.arange(w, device=tmp_data.device),
                        torch.arange(h, device=tmp_data.device)
                    )
                    mask = (x_grid >= min_row) * (x_grid < max_row) * (y_grid >= min_col) * (y_grid < max_col)
                    counter = (tmp_data * mask).flatten(start_dim=-2).sum(dim=-1)
                else:
                    counter = tmp_data.flatten(start_dim=-2).sum(dim=-1)

            # in all cases (sparse and dense tensor) do this
            if self.return_fraction:
                counter = counter / counter.sum(dim=-1, keepdim=True)
            result.append(counter.to(original_device))

        return result[0] if len(result) == 1 else result

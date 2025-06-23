from __future__ import annotations
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import numpy
import numpy as np
import pandas
import copy
import tqdm
import torch
from tissuemosaic.models.patch_analyzer import SpatialAutocorrelation
from tissuemosaic.data.dataset import CropperSparseTensor
from tissuemosaic.utils.validation_util import SmartPca, SmartUmap, SmartLeiden
from tissuemosaic.utils.nms_util import NonMaxSuppression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from scanpy import AnnData
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import torch.nn.functional as F

# trick to avoid circular imports
if TYPE_CHECKING:
    from .datamodule import AnndataFolderDM

class SparseImage:
    """
    Sparse torch tensor containing the spatial data
    (for example spatial gene expression or spatial cell types).

    It has 3 dictionaries with spot, patch and image properties.
    """

    def __init__(
        self,
        spot_properties_dict: dict,
        x_key: str,
        y_key: str,
        category_key: str,
        categories_to_codes: dict,
        pixel_size: float,
        padding: int = 10,
        sample_status: int = 0,
        patch_properties_dict: dict = None,
        image_properties_dict: dict = None,
        anndata: AnnData = None):
        """
        The user can initialize a SparseImage using this constructor or the :meth:`from_anndata`.

        Args:
            spot_properties_dict: the dictionary with the spot properties (at the minimum x,y,category)
            x_key: str, the key where the x_coordinates are stored in the spot_properties_dict
            y_key: str, the key where the y_coordinates are stored in the spot_properties_dict
            category_key: str, the key where the categories are stored in the spot_properties_dict
            categories_to_codes: dictionary with the mapping from categories (keys) to codes (values).
                The codes must be integers starting from zero. For example {"macrophage" : 0, "t-cell": 1}.
            pixel_size: float, size of the pixel. It used in the conversion
                between raw coordinates and pixel coordinates.
            padding: int, padding of the image (must be >= 1)
            patch_properties_dict: the dictionary with the patch properties.
                If None (defaults) an empty dict is generated.
            image_properties_dict: the dictionary with the image properties.
                If None (defaults) an empty dict is generated.
            anndata: the anndata object with other information (such as count_matrix etc)
        """
        # These variable are passed in
        self._padding = max(1, int(padding))
        self._pixel_size = pixel_size
        self._x_key = x_key
        self._y_key = y_key
        self._cat_key = category_key
        self._categories_to_codes = categories_to_codes
        self._spot_properties_dict = spot_properties_dict
        self._patch_properties_dict = {} if patch_properties_dict is None else patch_properties_dict
        self._image_properties_dict = {} if image_properties_dict is None else image_properties_dict
        self._anndata = anndata
        self._sample_status = sample_status

        # These variables are built
        self.origin = None

        self.data = self._create_torch_sparse_image(padding=self._padding, pixel_size=self._pixel_size)
        print("The dense shape of the image is ->", self.data.size())
        # print("Occupacy (zero, single, double, ...) of voxels in 3D sparse array ->",
        #       torch.bincount(self.data.values()).cpu().numpy())
        # tmp_sp = torch.sparse.sum(self.data, dim=-3)  # sum over categories
        # print("Occupacy (zero, single, double, ...) of voxels  in 2D sparse array (summed over category) ->",
        #       torch.bincount(tmp_sp.values()).cpu().numpy())

    def _create_torch_sparse_image(self, padding: int, pixel_size: float) -> torch.sparse.Tensor:

        # Check all vectors are 1D and of the same length
        x_raw = torch.from_numpy(self.x_raw).float()
        y_raw = torch.from_numpy(self.y_raw).float()
        cat_raw = self.cat_raw ## cat raw represents original dataframe of cell type proportions


        assert x_raw.shape[0] == y_raw.shape[0] == cat_raw.shape[0] and len(x_raw.shape) == 1, \
            "Error. x_raw, y_raw, cat_raw must be 1D array of the same length"

        # Check all category keys are included in categories_to_code
        assert set(cat_raw.columns) == set(self._categories_to_codes.keys()), \
            "Error. Some categories are NOT present in the categories_to_codes dictionary or vice versa"

        ## convert cell type proportions dataframe to tensor 'codes_vals'
        codes = torch.tensor([cat_raw[cat] for cat in cat_raw]).long() 
        codes_vals = torch.tensor(cat_raw.to_numpy()) 
        
        # Use GPU if available
        if torch.cuda.is_available():
            x_raw = x_raw.cuda().float()
            y_raw = y_raw.cuda().float()
            codes = codes.cuda().long()

        assert x_raw.shape[0] == y_raw.shape[0] == codes.shape[1]
        assert len(codes.shape) == 2
        print("number of elements --->", x_raw.shape[0]) 
        mean_spacing, median_spacing = self._check_mean_median_spacing(x_raw, y_raw)
        print("mean and median spacing {0}, {1}".format(mean_spacing, median_spacing))

        
        n_codes = numpy.max(list(self._categories_to_codes.values()))

        # Check that all codes are used at least once
        # code_usage = torch.bincount(codes, minlength=n_codes+1)
        # if not torch.prod(code_usage > 0):
        #     print("WARNING: some codes are not used! \
        #     This might be OK if some codes correspond to very rare genes or cell_types")

        # Define the raw coordinates of the origin
        x_raw_min = torch.min(x_raw).item()
        y_raw_min = torch.min(y_raw).item()
        self.origin = (x_raw_min - padding * pixel_size, y_raw_min - padding * pixel_size)

        # Convert the coordinates and round to the closest integer
        x_pixel, y_pixel = self.raw_to_pixel(x_raw, y_raw)
        ix = torch.round(x_pixel).long()
        iy = torch.round(y_pixel).long()

        # Create a sparse array with cell type proportion in the correct channel and x,y location.
        # The coalesce make sure that if in case of a collision the values are summed.
        dense_shape = (
            n_codes + 1,
            torch.max(ix).item() + 1 + 2 * padding,
            torch.max(iy).item() + 1 + 2 * padding,
        )
        
        # Obtain cell types as indices
        codes = torch.arange(start=0,end=n_codes+1).repeat(ix.shape[0])
        codes_1d = torch.flatten(codes)

        # Remove any nan (todo: check why nan exists in the first place)
        codes_vals = torch.flatten(torch.nan_to_num(codes_vals)) 
        
        # Remove 0 values as don't need to store
        keep_ind = torch.nonzero(codes_vals).squeeze()
        codes_vals = codes_vals[keep_ind]
        codes_1d = codes_1d[keep_ind]
        
        # Repeat ix and iy values based on number of cell_types at that spot
        ix_rep = torch.repeat_interleave(ix, n_codes+1)
        ix_rep = ix_rep[keep_ind]
        
        iy_rep = torch.repeat_interleave(iy, n_codes+1)
        iy_rep = iy_rep[keep_ind]
        
        return torch.sparse_coo_tensor(
            indices=torch.stack((codes_1d.cuda(), ix_rep, iy_rep), dim=0),
            values=codes_vals,
            size=dense_shape,
            device=codes.device,
            requires_grad=False,
        ).coalesce()
    

    def read_from_spot_dictionary(self, key: str) -> torch.Tensor:
        """
        Helper function to read from the spot dictionary.

        Args:
            key: the key corresponding to the information to read

        Returns:
            values: array of shape :math:`(n_\\text{spot}, *)` with the spot-level information
        """
        return self._to_torch(self._spot_properties_dict[key])

    def read_from_patch_dictionary(self, key: str) -> (torch.Tensor, torch.Tensor):
        """
        Helper function to read from the patch dictionary.

        Args:
            key: the key corresponding to the information to read

        Returns:
            values: array of shape :math:`(n_\\text{patches}, *, *, *)` with the patch-level information
            patches_xywh: array of shape :math:`(n_\\text{patches}, 4)` with the coordinates of the patches
        """
        patch_quantity = self._to_torch(self._patch_properties_dict[key]).to(device=self.device,
                                                                             dtype=torch.float)
        patches_xywh = self._to_torch(self._patch_properties_dict[key+"_patch_xywh"]).to(device=self.device,
                                                                                         dtype=torch.long)

        assert len(patches_xywh.shape) == 2 and patches_xywh.shape[-1] == 4, \
            "Error. Received {0}".format(patches_xywh.shape)

        assert patch_quantity.shape[0] == patches_xywh.shape[0], \
            "Shape mismatched in patch_dictionary for {0}. Received {1} and {2}".format(key,
                                                                                        patch_quantity.shape,
                                                                                        patches_xywh.shape)
        return patch_quantity, patches_xywh

    def read_from_image_dictionary(self, key: str) -> torch.Tensor:
        """
        Helper function to read from the patch dictionary.

        Args:
            key: the key corresponding to the information to read

        Returns:
            values: array of shape :math:`(ch, w, h)` with the image-level information
        """
        return self._to_torch(self._image_properties_dict[key])

    def write_to_spot_dictionary(self, key: str, values: Union[torch.Tensor, numpy.ndarray], overwrite: bool = False):
        """
        Helper function to write info to the spot dictionary.

        Args:
            key: the key corresponding to the information to write
            values: array of shape :math:`(n, *)` with the spot-level information
            overwrite: If True (default is False) overwrite the value if already present
        """
        assert len(values.shape) == 2 or len(values.shape) == 1, \
            "Error. values must be a 1D or 2D array. Received {0}".format(values.shape)
        assert values.shape[0] == self.n_spots
        if overwrite or key not in set(self._spot_properties_dict.keys()):
            self._spot_properties_dict[key] = self._to_numpy(values)
        else:
            print("Key {0} already present in spot dictionary. Set overwrite to True to overwrite".format(key))

    def write_to_patch_dictionary(self,
                                  key: str,
                                  values: Union[torch.Tensor, numpy.ndarray],
                                  patches_xywh: Union[torch.Tensor, numpy.ndarray],
                                  overwrite: bool = False):
        """
        Helper function to write info to the patch dictionary.

        Args:
            key: the name under which to save the information
            values: the patch-level information of shape :math:`(n_\\text{patches}, *, *, *)`
            patches_xywh: the location from where the patches were taken of shape :math:`(n_\\text{patches}, 4)`
            overwrite: If True (default is False) overwrite the value if already present
        """
        assert values.shape[0] == patches_xywh.shape[0]
        assert len(patches_xywh.shape) == 2 and patches_xywh.shape[-1] == 4, \
            "Error. Received {0}".format(patches_xywh.shape)

        if overwrite or key not in set(self._patch_properties_dict.keys()):
            self._patch_properties_dict[key] = self._to_numpy(values)
            self._patch_properties_dict[key + "_patch_xywh"] = self._to_numpy(patches_xywh)
        else:
            print("Key {0} already present in patch dictionary. Set overwrite to True to overwrite".format(key))

    def write_to_image_dictionary(self, key, values: Union[torch.Tensor, numpy.ndarray], overwrite: bool = False):
        """
        Helper function to write information to the image dictionary.

        Args:
            key: the key corresponding to the information to write
            values: array of shape :math:`(*, w, h)` with the image-level information
            overwrite: If True (default is False) overwrite the value if already present
        """
        assert len(values.shape) == 2 or len(values.shape) == 3, \
            "Error. values must be a 2D or 3D array. Received {0}".format(values.shape)
        assert values.shape[-2:] == self.shape[-2:]
        if overwrite or key not in set(self._image_properties_dict.keys()):
            self._image_properties_dict[key] = self._to_numpy(values)
        else:
            print("Key {0} already present in image dictionary. Set overwrite to True to overwrite".format(key))

    def clear_dicts(self, patch_dict: bool = True, image_dict: bool = True):
        """
        Clear the patch_properties_dict and image_properties_dict in their entirety.
        Useful to restart the analysis from scratch.
        It will never modify the spot_properties_dict.

        Args:
            patch_dict: If True (defaults) the patch_properties_dictionary is cleared
            image_dict: If True (defaults) the image_properties_dictionary is cleared
        """
        if patch_dict:
            self._patch_properties_dict = {}
        if image_dict:
            self._image_properties_dict = {}

    def trim_spot_dictionary(self, keys: List[str]):
        """
        Clear selective entries in the spot_properties_dictionary.

        Args:
            keys: the list of keys to remove from the spot dictionary
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            _ = self._spot_properties_dict.pop(key, None)

    def trim_patch_dictionary(self, keys: List[str]):
        """
        Clear selective entries in the patch_properties_dictionary.

        Args:
            keys: the list of keys to remove from the patch dictionary
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            _ = self._patch_properties_dict.pop(key, None)

    def trim_image_dictionary(self, keys: List[str]):
        """
        Clear selective entries in the image_properties_dictionary.

        Args:
            keys: the list of keys to remove from the image dictionary
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            _ = self._image_properties_dict.pop(key, None)
            
    @property
    def spot_properties_dict(self):
        """
        Return spot_properties_dictionary.

        """
        return self._spot_properties_dict
    
    @property
    def patch_properties_dict(self):
        """
        Return patch_properties_dictionary.

        """
        return self._patch_properties_dict

    @property
    def image_properties_dict(self):
        """
        Return image_properties_dictionary.

        """
        return self._image_properties_dict
    
    @property
    def pixel_size(self):
        """
        Return pixel size
        """
        return self._pixel_size
        
    def pixel_to_raw(
            self,
            x_pixel: torch.Tensor,
            y_pixel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Utility to convert the pixel coordinates to the raw_coordinates.
        This is a simple scale and shift transformation.

        Args:
            x_pixel: tensor of arbitrary shape with the x_index of the pixels
            y_pixel: tensor of arbitrary shape with the x_index of the pixels

        Returns:
            x_raw: tensor with the x_coordinates in raw unit. It has the same shape as input.
            y_raw: tensor with the y_coordinates in raw unit. It has the same shape as input.
        """
        x_raw = x_pixel * self._pixel_size + self.origin[0]
        y_raw = y_pixel * self._pixel_size + self.origin[1]
        return x_raw, y_raw

    def raw_to_pixel(
            self,
            x_raw: torch.Tensor,
            y_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Utility to convert the raw coordinates to pixel coordinates.
        This is a simple scale and shift transformation.

        Args:
            x_raw: tensor of arbitrary shape with the x_raw coordinates
            y_raw: tensor of arbitrary shape with the y_raw coordinates

        Returns:
            x_pixel: tensor with the x_coordinates in pixel unit. It has the same shape as input.
            y_pixel: tensor with the y_coordinates in pixel unit. It has the same shape as input.
        """
        x_pixel = (x_raw - self.origin[0]) / self._pixel_size
        y_pixel = (y_raw - self.origin[1]) / self._pixel_size
        return x_pixel, y_pixel

    @property
    def category_to_channels(self) -> dict:
        """ The mapping between categories and channels in the image. """
        return self._categories_to_codes

    @property
    def channels_to_category(self) -> numpy.ndarray:
        """
        The mapping between channels in the image and categories.
        Note that a channel can represent more than one category.
        For example CD+ and CD- cells can both be shown in channel 7.
        In that case channels_to_category[7] -> "CD+_CD-".
        """
        list_tmp = [None]*self.shape[-3]
        for cat, ch in self._categories_to_codes.items():
            if list_tmp[ch] is None:
                list_tmp[ch] = str(cat)
            else:
                list_tmp[ch] = "{}_{}".format(list_tmp[ch], str(cat))
        return numpy.array(list_tmp)

    @property
    def anndata(self) -> AnnData:
        """ The anndata object used to create the image. Might be None. """
        return self._anndata

    @property
    def x_raw(self) -> numpy.ndarray:
        """ The x_coordinates (in raw units) of the spots used to create the sparse image """
        return numpy.asarray(self._spot_properties_dict[self._x_key])

    @property
    def y_raw(self) -> numpy.ndarray:
        """ The y_coordinates (in raw units) of the spots used to create the sparse image """
        return numpy.asarray(self._spot_properties_dict[self._y_key])

    @property
    def cat_raw(self) -> pandas.DataFrame:
        """ The categorical labels (gene-identities or cell-identities) from the original data """

        cat_raw_dict = {}
        for key in self._spot_properties_dict.keys():
            ## if key is a cell type we are mapping
            if key in self._anndata.obsm[self._cat_key].columns:
                try: 
                    cat_raw_dict[key] = self._spot_properties_dict[key].tolist()
                except:
                    cat_raw_dict[key] = self._spot_properties_dict[key]
            
        cat_raw_df = pandas.DataFrame.from_dict(cat_raw_dict)  

        return cat_raw_df

    @property
    def n_spots(self) -> int:
        return self.x_raw.shape[0]

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def is_sparse(self) -> bool:
        return self.data.is_sparse

    def coalesce(self) -> "SparseImage":
        self.data = self.data.coalesce()
        return self

    def size(self) -> torch.Size:
        return self.data.size()

    def to(self, *args, **kwargs) -> "SparseImage":
        """ Move the data to a device or cast it to a different type """
        self.data = self.data.to(*args, **kwargs)
        return self

    def cuda(self) -> "SparseImage":
        self.data = self.data.cuda()
        return self

    def cpu(self) -> "SparseImage":
        self.data = self.data.cpu()
        return self

    def indices(self) -> torch.Tensor:
        return self.data.indices()

    def values(self) -> torch.Tensor:
        return self.data.values()

    def inspect(self):
        """ Describe the content of the spot, patch and image properties dictionaries """

        def _inspect_dict(d, prefix: str = ''):
            for k, v in d.items():
                if isinstance(v, list):
                    print(prefix, k, type(v), len(v))
                elif isinstance(v, torch.Tensor):
                    print(prefix, k, type(v), v.shape, v.device)
                elif isinstance(v, numpy.ndarray):
                    print(prefix, k, type(v), v.shape)
                elif isinstance(v, dict):
                    print(prefix, k, type(v))
                    _inspect_dict(v, prefix=prefix + "-->")
                else:
                    print(prefix, k, type(v))

        print("")
        print("-- spot_properties_dict --")
        _inspect_dict(self._spot_properties_dict)

        print("")
        print("-- patch_properties_dict --")
        _inspect_dict(self._patch_properties_dict)

        print("")
        print("-- image_properties_dict --")
        _inspect_dict(self._image_properties_dict)

    def to_dense(self) -> torch.Tensor:
        """
        Create a dense torch tensor of shape :math:`(C, W, H)`
        where the number of channels is equal to the number of categories of
        the underlying spatial data.

        Returns:
            dense_img: A dense representation of the sparse image

        Note:
            This will convert the sparse array into a dense array and might
            lead to a very large memory footprint.

        Note:
            It is useful for visualization of the data.
        """
        return self.data.to_dense()

    def to_rgb(self,
               spot_size: float = 1.0,
               cmap: matplotlib.colors.ListedColormap = None,
               figsize: Tuple = (8, 8),
               show_colorbar: bool = True,
               contrast: float = 1.0) -> (torch.Tensor, plt.Figure):
        """
        Make a 3 channel RGB image.

        Args:
            spot_size: size of sigma of gaussian kernel for rendering the spots
            cmap: the colormap to use
            figsize: the size of the figure
            show_colorbar: If True show the colorbar
            contrast: change to increase/decrease the contrast in the figure.
                It does not affect the returned tensor. It changes only the way to figure is displayed.

        Returns:
            dense_img: A torch.Tensor of size :math:`(3, W, H)` with the rgb rendering of the image
            fig: matplotlib figure.
        """

        def _make_kernel(_sigma: float):
            n = int(1 + 2 * numpy.ceil(4.0 * _sigma))
            dx_over_sigma = torch.linspace(-4.0, 4.0, 2 * n + 1).view(-1, 1)
            dy_over_sigma = dx_over_sigma.clone().permute(1, 0)
            d2_over_sigma2 = (dx_over_sigma.pow(2) + dy_over_sigma.pow(2)).float()
            kernel = torch.exp(-0.5 * d2_over_sigma2)
            return kernel

        def _get_color_tensor(_cmap, _ch):
            if _cmap is None:
                # cm = cc.cm.glasbey_bw_minc_20
                cm = plt.get_cmap('tab20', _ch)
                x = numpy.arange(_ch)
                colors_np = cm(x)
            else:
                cm = plt.get_cmap(_cmap, _ch)
                x = numpy.linspace(0.0, 1.0, _ch)
                colors_np = cm(x)

            color = torch.Tensor(colors_np)[:, :3]
            assert color.shape[0] == _ch
            return color

        dense_img = self.to_dense().unsqueeze(dim=0).float()  # shape: (1, ch, width, height)
        ch = dense_img.shape[-3]
        weight = _make_kernel(spot_size).expand(ch, 1, -1, -1)

        if torch.cuda.is_available():
            dense_img = dense_img.cuda()
            weight = weight.cuda()

        dense_rasterized_img = F.conv2d(
            input=dense_img,
            weight=weight,
            bias=None,
            stride=1,
            padding=(weight.shape[-1] - 1) // 2,
            dilation=1,
            groups=ch,
        ).squeeze(dim=0)

        colors = _get_color_tensor(cmap, ch).float().to(dense_rasterized_img.device)
        rgb_img = torch.einsum("cwh,cn -> nwh", dense_rasterized_img, colors)
        in_range_min, in_range_max = torch.min(rgb_img), torch.max(rgb_img)
        dist = in_range_max - in_range_min
        scale = 1.0 if dist == 0.0 else 1.0 / dist
        rgb_img.add_(other=in_range_min, alpha=-1.0).mul_(other=scale).clamp_(min=0.0, max=1.0)
        rgb_img = rgb_img.detach().cpu()

        # make the figure
        fig, ax = plt.subplots(figsize=figsize)
        _ = ax.imshow((rgb_img.permute(1, 2, 0)*contrast).clamp(min=0.0, max=1.0))

        if show_colorbar:
            discrete_cmp = matplotlib.colors.ListedColormap(colors.cpu().numpy())
            normalizer = matplotlib.colors.BoundaryNorm(
                boundaries=numpy.linspace(-0.5, ch - 0.5, ch + 1),
                ncolors=ch,
                clip=True)

            scalar_mappable = matplotlib.cm.ScalarMappable(norm=normalizer, cmap=discrete_cmp)
            cbar = fig.colorbar(scalar_mappable, ticks=numpy.arange(ch), ax=ax)
            legend_colorbar = list(self._categories_to_codes.keys())
            cbar.ax.set_yticklabels(legend_colorbar)
        plt.close()

        return rgb_img, fig
    
    ## TODO: deprecate this function
    def to_rgb_image_property(self,
               image_property_key: str = None,
               spot_size: float = 1.0,
               cmap: matplotlib.colors.ListedColormap = None,
               figsize: Tuple = (8, 8),
               show_colorbar: bool = True,
               contrast: float = 1.0) -> (torch.Tensor, plt.Figure):
        """
        Make a 3 channel RGB image from a property from image properties dict.

        Args:
            image_property_key: which key in image properties dict to visualize 
            spot_size: size of sigma of gaussian kernel for rendering the spots
            cmap: the colormap to use
            figsize: the size of the figure
            show_colorbar: If True show the colorbar
            contrast: change to increase/decrease the contrast in the figure.
                It does not affect the returned tensor. It changes only the way to figure is displayed.

        Returns:
            dense_img: A torch.Tensor of size :math:`(3, W, H)` with the rgb rendering of the image
            fig: matplotlib figure.
        """

        def _make_kernel(_sigma: float):
            n = int(1 + 2 * numpy.ceil(4.0 * _sigma))
            dx_over_sigma = torch.linspace(-4.0, 4.0, 2 * n + 1).view(-1, 1)
            dy_over_sigma = dx_over_sigma.clone().permute(1, 0)
            d2_over_sigma2 = (dx_over_sigma.pow(2) + dy_over_sigma.pow(2)).float()
            kernel = torch.exp(-0.5 * d2_over_sigma2)
            return kernel
 
        dense_img = torch.tensor(self._image_properties_dict[image_property_key]).unsqueeze(dim=0).unsqueeze(dim=0) # shape: (1, ch=1, width, height)

        ch = dense_img.shape[-3]
        weight = _make_kernel(spot_size).expand(ch, 1, -1, -1)

        if torch.cuda.is_available():
            dense_img = dense_img.cuda()
            weight = weight.cuda()

        ## TODO: test, remove later
        dense_img = dense_img.squeeze(0)
        
        dense_rasterized_img = F.conv2d(
            input=dense_img,
            weight=weight,
            bias=None,
            stride=1,
            padding=(weight.shape[-1] - 1) // 2,
            dilation=1,
            groups=ch,
        ).squeeze(dim=0)
        
        dense_rasterized_img = dense_rasterized_img.squeeze().cpu()
        
        return dense_img


    def crops(
            self,
            crop_size: int,
            strategy: str = 'random',
            n_crops: int = 10,
            n_element_min: int = 0,
            stride: int = 200,
            random_order: bool = True) -> Tuple[List[torch.sparse.Tensor], List[int], List[int]]:
        """
        Wrapper around :class:`tissuemosaic.dataset.CropperSparseTensor`.

        Returns:
            sp_img: list of crops represented as sparse torch Tensor
            x_list: list with the x coordinates of the bottom left corners of the crops
            y_list: list with the y coordinates of the bottom left corners of the crops
        """
        return CropperSparseTensor(strategy=strategy,
                                   n_crops=n_crops,
                                   n_element_min=n_element_min,
                                   crop_size=crop_size,
                                   stride=stride,
                                   random_order=random_order)(self.data)

    def moran(
            self,
            n_neighbours: Optional[int] = 6,
            radius: Optional[float] = None,
            neigh_correct: bool = False) -> torch.Tensor:
        """
        Wrapper around :class:`tissuemosaic.models.patch_analyzer.SpatialAutocorrelation`.

        Returns:
            score: array of shape `C` with the Moran's I score for each channels (i.e. how one channel is
                mixed with all the others)
        """
        return SpatialAutocorrelation(modality='moran',
                                      n_neighbours=n_neighbours,
                                      radius=radius,
                                      neigh_correct=neigh_correct)(self.data)

    def gready(
            self,
            n_neighbours: Optional[int] = 6,
            radius: Optional[float] = None,
            neigh_correct: bool = False) -> torch.Tensor:
        """
        Wrapper around :class:`tissuemosaic.models.patch_analyzer.SpatialAutocorrelation`.

        Returns:
            score: array of shape `C` with the Gready's score for each channels (i.e. how one channel is
                mixed with all the others)
        """
        return SpatialAutocorrelation(modality='gready',
                                      n_neighbours=n_neighbours,
                                      radius=radius,
                                      neigh_correct=neigh_correct)(self.data)

    def compute_ncv(self,
                    feature_name: str = None,
                    k: int = None,
                    r: float = None,
                    overwrite: bool = False):
        """
        Compute the neighborhood composition vectors (ncv) of every spot
        and store the results in the spot_properties_dictionary under :attr:`feature_name`.

        Args:
            feature_name: the key under which the results will be stored.
            k: if specified the k nearest neighbours are used to compute the ncv.
            r: if specified the neighbours at a distance less than r (in raw units) are used to compute the ncv.
            overwrite: if the :attr:`feature_name` is already present in the spot_properties_dict,
                this variable controls when to overwrite it.
        """

        assert (k is None and r is not None) or (k is not None and r is None), \
            "Exactly one between r and k must be defined."
        assert k is None or isinstance(k, int) and k > 0, "k is either None or a positive integer"
        assert r is None or r > 0, "r is either None or a positive value"

        
        metric_features = numpy.stack((self.x_raw, self.y_raw), axis=-1)
        chs = self.shape[-3]
        
        # get raw data (cell type identities)
        codes_vals = torch.tensor(self.cat_raw.to_numpy()).cpu()

        # convert probabilistic assignment to one hot encoded assignment for computing ncv
        cell_types_one_hot = numpy.zeros_like(codes_vals)
        max_assignment = numpy.argmax(codes_vals, axis = 1)
 
        cell_types_one_hot[numpy.arange(codes_vals.shape[0]), max_assignment] = 1
        cell_types_one_hot = torch.Tensor(cell_types_one_hot)

        # compute ncv
        if k is not None:
            # use a knn neighbours
            from sklearn.neighbors import KDTree
            kdtree = KDTree(metric_features)
            dist, ind = kdtree.query(metric_features, k=k)  # shapes (*, k), (*, k)
            ncv_tmp = cell_types_one_hot[ind]  # shape (*, k, ch)
            ncv = ncv_tmp.sum(dim=-2)  # shape (*, ch)
        else:
            # use radius neighbours
            from sklearn.neighbors import BallTree
            balltree = BallTree(metric_features)
            inds = balltree.query_radius(metric_features,
                                         r=r,
                                         return_distance=False,
                                         count_only=False,
                                         sort_results=False)
            ncv = torch.zeros_like(cell_types_one_hot)  # shape (*, ch)
            # inds is a list of array of different length. There is noway to avoid the for-loop
            n_neighbours_list = []
            for n, ind in enumerate(inds):
                ncv_tmp = cell_types_one_hot[ind].sum(dim=-2)  # shape (ch)
                ncv[n] = ncv_tmp
                n_neighbours_list.append(len(ind))

            # report the statistics for the number of neighbours at this cutoff radius
            n_neighbours_np = numpy.array(n_neighbours_list)
            print("# neighbours at this cutoff radius: min={0}, mean={1}, median={2}, max={3}".format(
                numpy.min(n_neighbours_np),
                numpy.mean(n_neighbours_np),
                numpy.median(n_neighbours_np),
                numpy.max(n_neighbours_np)))


        ncv = ncv.float() / ncv.sum(dim=-1, keepdim=True).clamp(min=1.0)  # transform to proportions
        self.write_to_spot_dictionary(key=feature_name, values=ncv, overwrite=overwrite)
        return ncv


    @torch.no_grad()
    def compute_patch_features(
            self,
            feature_name: str,
            datamodule: AnndataFolderDM,
            model: torch.nn.Module,
            apply_transform: bool = True,
            batch_size: int = 64,
            strategy: str='tiling',
            remove_overlap: bool = True, ## only used if strategy is 'random'
            val_io_min_threshold: float = 0.0,
            n_patches_max: int = 100,
            fraction_patch_overlap: float = 0.0,
            overwrite: bool = False,
            return_crops: bool = False,
            compute_ncv: bool = True,
            ncv_prefix: str = None) -> Union[torch.Tensor, None]:
        """
        Split the sparse image into (possibly overlapping) patches.
        Each patch is analyzed by the (pretrained) model.
        The features are stored in the patch_properties_dict under the :attr:`feature_name`.

        Args:
            feature_name: the key under which the results will be stored.
            datamodule: Datamodule used for training the model.
                Passing it here guarantees that the cropping strategy and
                the data augmentations are identical to the one used during training.
            model: the trained model will ingest the patch and produce the features.
            apply_transform: if True (defaults) the datamodule.test_trasform will be applied to the crops before
                feeding them into the model.
                If False no transformation is applied and the sparse tensors are fed into the model.
            batch_size: how many crops to process simultaneously (default = 64). Use to adjust the GPU memory footprint. Currently only supported for strategy 'random'
            strategy: either 'tiling' or 'random', determines cropper strategy for generating patches
            remove_overlap: if True, remove overlapping (random) patches
            val_iomin_threshold: threshold for the Intersection over Minimum for NonMaxSuppression. It must be in [0.0, 1.0). Only used if strategy is 'random'.
            n_patches_max: maximum number of patches generated to analyze the current picture (default = 100), only used if strategy is 'random'
            overwrite: if the :attr:'feature_names' are already present in the patch_properties_dict,
                this variable controls when to overwrite them.
            return_crops: if True the model returns a (batched) torch.Tensor of shape
                :math:`(\\text{n_patches_max}, c, w, h)` with all the crops which were fed to the model.
                Default is False.
            compute_ncv: if True, neighborhood composition of each patch is also computed and stored in the patch_properties_dict under the :attr:`ncv_prefix + _patch_ncv`.
            ncv_prefix: if provided, prepended to key under which patch ncv is saved in patch_properties_dict

        Returns:
            patches: If :attr:`return_crops` is True returns tensor of shape
                :math:`(N, C, W, H)` with all the crops which were cropped and analyzed.
                Else returns None.
        """

        if feature_name in self._patch_properties_dict.keys() and not overwrite:
            print("The key {0} is already present in patch_properties_dict.")
            print(" Set overwrite=True to overwrite its value. Nothing will be done.".format(feature_name))
            return
        elif feature_name in self._patch_properties_dict.keys() and overwrite:
            print("The key {0} is already present in patch_properties_dict and it will be overwritten".format(
                feature_name))

        # set the model into eval mode
        was_original_in_training_mode = model.training
        model.eval()

        all_patches, all_features, all_crops = [], [], []
        n_patches = 0
        patches_x, patches_y, patches_w, patches_h = [], [], [], []
        
        if strategy == 'random':
            
            while n_patches < n_patches_max:
                n_tmp = min(batch_size, n_patches_max - n_patches)
                crops, x_locs, y_locs = datamodule.cropper_test(self.data, strategy=strategy, n_crops=n_tmp)
                patches_x += x_locs
                patches_y += y_locs
                patches_w += [crop.shape[-2] for crop in crops]
                patches_h += [crop.shape[-1] for crop in crops]
                n_patches += len(x_locs)

                ## TODO: requires apply_transform to be set to true to work
                if apply_transform:
                    patches = datamodule.trsfm_test(crops)
                    all_patches.append(patches.detach().cpu())
                else:
                    patches = crops
                    all_patches.append(patches)
                

                features_tmp = model(patches)
                if isinstance(features_tmp, torch.Tensor):
                    all_features.append(features_tmp)
                elif isinstance(features_tmp, numpy.ndarray):
                    all_features.append(torch.from_numpy(features_tmp))
                elif isinstance(features_tmp, list):
                    all_features += features_tmp
                else:
                    raise NotImplementedError
                    
                all_crops += crops
                                
        elif strategy == 'tiling':
            ## TODO: allow batching with tiling strategy
            crops, x_locs, y_locs = datamodule.cropper_test(self.data, strategy=strategy, fraction_patch_overlap = fraction_patch_overlap)

            patches_x += x_locs
            patches_y += y_locs
            patches_w += [crop.shape[-2] for crop in crops]
            patches_h += [crop.shape[-1] for crop in crops]
            n_patches += len(x_locs)

            if apply_transform:
                patches = datamodule.trsfm_test(crops)
            else:
                patches = crops

            all_patches.append(patches.detach().cpu())

            features_tmp = model(patches)
            if isinstance(features_tmp, torch.Tensor):
                all_features.append(features_tmp)
            elif isinstance(features_tmp, numpy.ndarray):
                all_features.append(torch.from_numpy(features_tmp))
            elif isinstance(features_tmp, list):
                all_features += features_tmp
            else:
                raise NotImplementedError
                
            all_crops += crops
        

        # put back the model in the state it was original
        if was_original_in_training_mode:
            model.train()

        # make a single batched tensor of both patches coordinates
        x_torch = torch.tensor(patches_x, dtype=torch.int).cpu()
        y_torch = torch.tensor(patches_y, dtype=torch.int).cpu()
        w_torch = torch.tensor(patches_w, dtype=torch.int).cpu()
        h_torch = torch.tensor(patches_h, dtype=torch.int).cpu()
        patches_xywh = torch.stack((x_torch, y_torch, w_torch, h_torch), dim=-1).long()

        
        ## concatenate patches and features from batches
        
        if strategy == 'random':
            if len(all_features) == n_patches_max:
                features = torch.stack(all_features, dim=0).cpu()
            else:
                features = torch.cat(all_features, dim=0).cpu()
        elif strategy == 'tiling':
            features = torch.cat(all_features, dim=0).cpu()
            
            
        if strategy == 'random':
            if len(all_patches) == n_patches_max:
                patches = torch.stack(all_patches, dim=0).cpu()
            else:
                patches = torch.cat(all_patches, dim=0).cpu()
        elif strategy == 'tiling':
            patches = torch.cat(all_patches, dim=0).cpu()
            
            
        ## remove overlapping patches
        if strategy == 'random' and remove_overlap:
                
                print("remove overlap")

                initial_score = torch.rand_like(patches_xywh[:, 0].float())
                tissue_ids = torch.ones(patches_xywh.shape[0]) ## all crops coming from same sparse image
                nms_mask_n, overlap_nn = NonMaxSuppression.compute_nm_mask(
                    score=initial_score,
                    ids=tissue_ids,
                    patches_xywh=patches_xywh,
                    iom_threshold=val_io_min_threshold)
                binarized_overlap_nn = (overlap_nn > val_io_min_threshold).float()
                
                # create a dictionary with only non-overlapping patches to test kn-regressor/classifier
                nms_mask_n = NonMaxSuppression.perform_nms_selection(mask_overlap_nn=binarized_overlap_nn,
                                                                     score_n=torch.rand_like(initial_score),
                                                                     possible_n=torch.ones_like(initial_score).bool())
                
                patches = patches[nms_mask_n, :, :, :]
                features = features[nms_mask_n, :]
                
                mask_indices = [i for i, x in enumerate(nms_mask_n) if x]
                all_crops = [crop for crop, mask in zip(all_crops, nms_mask_n) if mask] 
                
                patches_xywh = patches_xywh[nms_mask_n, :]
        
        self.write_to_patch_dictionary(
            key=feature_name, values=features, patches_xywh=patches_xywh, overwrite=overwrite)
        
        if compute_ncv:
            codes = self._categories_to_codes.values()
            ## assert statement that values are integers or floats from 0-1
            
            patch_ncvs = []
            
            for crop in all_crops:
                
                ## get cell types in the patch
                cells = crop.indices().detach().clone()[[0]]
                ct_counts = []

                ## count number of each cell type
                for code in codes:
                    ct_counts.append(int(torch.count_nonzero(cells==code).detach().cpu()))

                ## normalize to form ncv
                ct_counts = np.array(ct_counts)
                ncv = ct_counts/np.sum(ct_counts)
                patch_ncvs.append(ncv)
                
            patch_ncvs = torch.tensor(np.array(patch_ncvs))
            
            if ncv_prefix is None:
                self.write_to_patch_dictionary(
                    key='patch_ncv', values=patch_ncvs, patches_xywh=patches_xywh, overwrite=overwrite)
            else:
                self.write_to_patch_dictionary(
                    key=ncv_prefix + '_patch_ncv', values=patch_ncvs, patches_xywh=patches_xywh, overwrite=overwrite)

        if return_crops:
            return patches
        
        
    @torch.no_grad()
    def compute_spot_features(self, feature_name: str,
            datamodule: AnndataFolderDM,
            model: torch.nn.Module,
            apply_transform: bool = True,
            batch_size: int = 64,
            overwrite: bool = True):
        """
        Draw a patch around each spot.
        Each (valid) patch is analyzed by the (pretrained) model.
        The features are stored in the patch_properties_dict under the :attr:`feature_name`.
        The validity of patches (boolean) is stored in the patch_properties_dict under the :attr:`feature_name_valid`.

        Args:
            feature_name: the key under which the results will be stored.
            datamodule: Datamodule used for training the model.
                Passing it here guarantees that the cropping strategy and
                the data augmentations are identical to the one used during training.
            model: the trained model will ingest the patch and produce the features.
            apply_transform: if True (defaults) the datamodule.test_trasform will be applied to the crops before
                feeding them into the model.
                If False no transformation is applied and the sparse tensors are fed into the model.
            batch_size: how many crops to process simultaneously (default = 64). Use to adjust the GPU memory footprint.
        """

        if feature_name in self._spot_properties_dict.keys() and not overwrite:
            print("The key {0} is already present in spot_properties_dict.")
            print(" Set overwrite=True to overwrite its value. Nothing will be done.".format(feature_name))
            return
        elif feature_name in self._spot_properties_dict.keys() and overwrite:
            print("The key {0} is already present in spot_properties_dict and it will be overwritten".format(
                feature_name))
        
        # set the model into eval mode
        was_original_in_training_mode = model.training
        model.eval()
        
        ## iterate over spots
        
        num_spots = self._spot_properties_dict["x_key"].shape[0]
        
        list_of_patch_xywh = []
        for spot_ind in range(num_spots):
            
            ## convert spot coordinate to image coordinate
            
            x_raw = torch.tensor(self.x_raw[spot_ind]).float()
            y_raw = torch.tensor(self.y_raw[spot_ind]).float()
            x_pixel, y_pixel = self.raw_to_pixel(x_raw=x_raw, y_raw=y_raw)
            
            # Convert the coordinates and round to the closest integer
            ix = torch.round(x_pixel).long()
            iy = torch.round(y_pixel).long()
        
            # get patch width and height from datamodule
            width = datamodule.global_size
            height = datamodule.global_size
            
            
            ## for each spot, define patch centered at that spot
            patch_xywh = torch.tensor([(ix - (width // 2)), (iy - (height // 2)), width, height]).unsqueeze(0)
            
            list_of_patch_xywh.append(patch_xywh)
            
        patches_xywh = torch.cat(list_of_patch_xywh, dim=0)

        ## copy sparse image to cpu before passing in if not already on cpu
        if self.data.device != 'cpu':
            data_cpu = self.data.clone().detach().cpu()
        
        crops, x_locs, y_locs = CropperSparseTensor.reapply_crops(data_cpu, patches_xywh)
        
        ## check if crops are valid
        
        list_of_n_elements = [len(crop.values()) for crop in crops]   

        valid_crops = np.array([n_elements >= datamodule.n_element_min_for_crop for n_elements in list_of_n_elements]) 
        
        ## compute features for crops
        
        all_features, all_crops = [], []
        n_patches = 0
        patches_x, patches_y, patches_w, patches_h = [], [], [], []
        n_patches_max = patches_xywh.shape[0]
            
        if torch.cuda.is_available():
            model = model.cuda()
            
        total_batches = n_patches_max // batch_size
        with tqdm.tqdm(total=total_batches) as pbar:
            while n_patches < n_patches_max:
                
                pbar.update(1)
                
                n_tmp = min(batch_size, n_patches_max - n_patches)

                crops_tmp = crops[n_patches:n_patches+n_tmp]
                
                patches_tmp = datamodule.trsfm_test(crops_tmp)
                
                if torch.cuda.is_available():
                    patches_tmp = patches_tmp.cuda()
                    
                features_tmp = model(patches_tmp)
                if isinstance(features_tmp, torch.Tensor):
                    all_features.append(features_tmp)
                elif isinstance(features_tmp, numpy.ndarray):
                    all_features.append(torch.from_numpy(features_tmp))
                elif isinstance(features_tmp, list):
                    all_features += features_tmp
                else:
                    raise NotImplementedError

                all_crops += crops
                
                n_patches = n_patches + n_tmp
            
        ## write to spot dictionary at the end 
        all_features = torch.cat(all_features,dim=0)
        self.write_to_spot_dictionary(key=feature_name, values=all_features)
        self.write_to_spot_dictionary(key=feature_name + "_valid", values=valid_crops)
        
        # put back the model in the state it was original
        if was_original_in_training_mode:
            model.train()
            
        print("Finished computing spot features.")
    
    
    def compute_patch_ncv_clusters(self, feature_xywh: str,
                                        res: int=0.4):
        """
            Utility function to compute leiden clusters with particular resolution on patch NCV vectors
            Args:
                feature_xywh: key in patch_properties_dict for xywh of model features
                res: resolution parameter for leiden clustering
        """
                    
        assert "patch_ncv" in self._patch_properties_dict.keys(), \
            "Compute Patch NCV first."

        ## Cluster patch NCVs
        patch_ncv = self._patch_properties_dict["patch_ncv"]

        assert np.array_equal(self._patch_properties_dict["patch_ncv_patch_xywh"], self._patch_properties_dict[feature_xywh]), \
            "NCV patch xywh does not match feature xywh"

        smart_umap = SmartUmap(n_neighbors=25, preprocess_strategy='z_score', n_components=2, min_dist=0.5, metric='cosine')
        embeddings_umap = smart_umap.fit_transform(patch_ncv)

        umap_graph = smart_umap.get_graph()
        smart_leiden = SmartLeiden(graph=umap_graph)

        leiden_clusters = smart_leiden.cluster(resolution=res, partition_type='RBC')

        return leiden_clusters
        
    
    def spot_train_test_split(self, patch_size: int):
        """
            Divide the puck into 4 train/test splits (forming quadrants) such that the no spot within train split is within 'patch_size' of corresponding test split
            The train/test ids (0/1) are stored in the spot dictionary under :attr:`train_test_fold_{num}`. Spots within the boundary are labelled -1. 
            Args:
                patch_size: determines the spatial boundary between train and test split
        """
        
        # get x and y keys
        spots_x_key = self._spot_properties_dict["x_key"]
        spots_y_key = self._spot_properties_dict["y_key"]
        
        # get median x and y locs
        x_median = np.median(spots_x_key)
        y_median = np.median(spots_y_key)
              

        ## patch width is width used to compute spot features
        ## patch height is height used to compute spot features
        x_boundary = self._pixel_size * patch_size
        y_boundary = self._pixel_size * patch_size
        
        
        ## compute train/test ids for split 1
        test_fold_1_inds = np.where((spots_x_key > x_median) & (spots_y_key > y_median))[0]

        train_fold_1_inds_1 = np.where((spots_x_key < (x_median - x_boundary)))[0]
        train_fold_1_inds_2 = np.where(spots_y_key < (y_median - y_boundary))[0]

        train_fold_1_inds = np.concatenate((train_fold_1_inds_1, train_fold_1_inds_2))

        train_test_fold_1 = -1*np.ones(spots_x_key.shape)

        train_test_fold_1[train_fold_1_inds] = 0
        train_test_fold_1[test_fold_1_inds] = 1

        self.write_to_spot_dictionary(key = 'train_test_fold_1', values=train_test_fold_1, overwrite=True)

        ## compute train/test ids for split 2
        test_fold_2_inds = np.where(np.logical_and(spots_x_key > x_median,spots_y_key < y_median))[0]

        train_fold_2_inds_1 = np.where((spots_x_key < (x_median - x_boundary)))[0]
        train_fold_2_inds_2 = np.where((spots_y_key > (y_median + y_boundary)))[0]
        train_fold_2_inds = np.concatenate((train_fold_2_inds_1, train_fold_2_inds_2))

        train_test_fold_2 = -1*np.ones(spots_x_key.shape)

        train_test_fold_2[train_fold_2_inds] = 0
        train_test_fold_2[test_fold_2_inds] = 1

        self.write_to_spot_dictionary(key = 'train_test_fold_2', values=train_test_fold_2, overwrite=True)


        ## compute train/test ids for split 3
        test_fold_3_inds = np.where(np.logical_and(spots_x_key < x_median,spots_y_key > y_median))[0]

        train_fold_3_inds_1 = np.where((spots_x_key > (x_median + x_boundary)))[0]
        train_fold_3_inds_2 = np.where((spots_y_key < (y_median - y_boundary)))[0]
        train_fold_3_inds = np.concatenate((train_fold_3_inds_1, train_fold_3_inds_2))

        train_test_fold_3 = -1*np.ones(spots_x_key.shape)

        train_test_fold_3[train_fold_3_inds] = 0
        train_test_fold_3[test_fold_3_inds] = 1

        self.write_to_spot_dictionary(key = 'train_test_fold_3', values=train_test_fold_3, overwrite=True)


        ## compute train/test ids for split 4
        test_fold_4_inds = np.where(np.logical_and(spots_x_key < x_median,spots_y_key < y_median))[0]

        train_fold_4_inds_1 = np.where((spots_x_key > (x_median + x_boundary)))[0]
        train_fold_4_inds_2 = np.where((spots_y_key > (y_median + y_boundary)))[0]
        train_fold_4_inds = np.concatenate((train_fold_4_inds_1, train_fold_4_inds_2))

        train_test_fold_4 = -1*np.ones(spots_x_key.shape)

        train_test_fold_4[train_fold_4_inds] = 0
        train_test_fold_4[test_fold_4_inds] = 1

        self.write_to_spot_dictionary(key = 'train_test_fold_4', values=train_test_fold_4, overwrite=True)

    def patch_train_test_split(self, feature_xywh: str=None,
                                res: int=0.4, 
                                stratify: bool=True,
                                write_to_spot_dictionary: bool=True, 
                                return_patches: bool=True,
                                train_size: float=0.8,
                                test_size: float=0.2,
                                random_state: int = 0) -> (dict, dict):
        
        """
        Split patch locations under feature into train/test split. Can stratify by patch cell composition (run compute_patch_ncv first). Useful for splitting data  
        without spatial overlap (if patches are computed with no overlap) for downstream regression tasks.

        Args:
            feature_xywh: Patch locations that would like to be split
            res: Leiden cluster resolution for determining patch NCV clusters
        """
        
        if train_size <= 0:
            raise ValueError("Train_size must be > 0")
        if test_size <= 0:
            raise ValueError("Test_size must be > 0")
        
        ## Cluster Patch NCVs
        if stratify:
            leiden_clusters = self.compute_patch_ncv_clusters(feature_xywh, res)

        ## Split patches into train/test, stratifying by NCV clusters
        
        if stratify:
            try:
                train_patch_xywh, test_patch_xywh, train_idx, test_idx = train_test_split(self._patch_properties_dict[feature_xywh], np.arange(self._patch_properties_dict[feature_xywh].shape[0]), train_size=train_size, test_size=test_size, stratify=leiden_clusters,
                                                                     random_state=random_state)
            except ValueError: 
                raise Exception("Not enough samples in each cluster to split the data. Try a smaller res")
        else:
            train_patch_xywh, test_patch_xywh, train_idx, test_idx = train_test_split(self._patch_properties_dict[feature_xywh], np.arange(self._patch_properties_dict[feature_xywh].shape[0]), train_size=train_size, test_size=test_size,
                                                                random_state=random_state)
        
        ## get idx of train/test/val split
        train_test_id = np.concatenate((np.zeros(train_patch_xywh.shape[0]), np.zeros(test_patch_xywh.shape[0])))
        train_test_id[test_idx] = 1
    
        sample_patch_xywh = np.concatenate((train_patch_xywh, test_patch_xywh))
        
        ## Write to patch dictionary
        self.write_to_patch_dictionary(key='train_test_split_id', values=train_test_id,
                patches_xywh=sample_patch_xywh, overwrite=True)
        
        ## Write to spot dictionary
        if write_to_spot_dictionary:
             self.transfer_patch_to_spot(
                keys_to_transfer='train_test_split_id',
                overwrite=True)
                
        if return_patches:
            return self._patch_properties_dict['train_test_split_id'], self._patch_properties_dict['train_test_split_id_patch_xywh']
        
    def patch_train_test_val_split(self, feature_xywh: str=None,
                                res: int=0.4, 
                                stratify: bool=True,
                                write_to_spot_dictionary: bool=True, 
                                return_patches: bool=True,
                                train_size: float = 0.8,
                                test_size: float = 0.15,
                                val_size: float = 0.05,
                                random_state: int = 0) -> (dict, dict):
        
        """
        Split patch locations under feature into train/test/val split. Can stratify by patch cell composition (run 
        compute_patch_ncv first). Useful for splitting data  without spatial overlap (if patches are computed with no 
        overlap) for downstream regression tasks.

        Args:
            feature_xywh: Patch locations that would like to be split
            res: Leiden cluster resolution for determining patch NCV clusters
            
        Returns:
            
        """
        
        ## Cluster Patch NCVs
        if stratify:
            leiden_clusters = self.compute_patch_ncv_clusters(feature_xywh, res)

        ## Split patches into train/test, stratifying by NCV clusters
        ## write custom train_test_split function in utils
        

        # Normalize the train/test/val sizes
        norm0 = train_size + test_size + val_size
        train_size_norm0 = train_size / norm0
        test_and_val_size_norm0 = (test_size + val_size) / norm0

        norm1 = test_size + val_size
        test_size_norm1 = test_size / norm1
        val_size_norm1 = val_size / norm1
        
        if stratify:
            try:
                train_patch_xywh, test_and_val_patch_xywh, train_leiden_clusters, test_and_val_leiden_clusters, train_idx, test_and_val_idx = train_test_split(self._patch_properties_dict[feature_xywh],
                                                                                       leiden_clusters, np.arange(self._patch_properties_dict[feature_xywh].shape[0]), train_size = train_size, test_size = test_and_val_size_norm0, 
                                                                                       stratify=leiden_clusters)
                val_patch_xywh, test_patch_xywh, val_idx, test_idx = train_test_split(test_and_val_patch_xywh, test_and_val_idx, train_size=val_size_norm1, test_size=test_size_norm1,  
                                                                   stratify=test_and_val_leiden_clusters)
            except ValueError: 
                raise Exception("Not enough samples in each cluster to split the data. Try a smaller res or adjusting train/test/val size")
        else:
            train_patch_xywh, test_and_val_patch_xywh, train_idx, test_and_val_idx = train_test_split(self._patch_properties_dict[feature_xywh], np.arange(self._patch_properties_dict[feature_xywh].shape[0]), train_size = train_size, test_size = test_and_val_size_norm0)
            val_patch_xywh, test_patch_xywh, val_idx, test_idx = train_test_split(test_and_val_patch_xywh, test_and_val_idx, train_size=val_size_norm1, test_size=test_size_norm1)
        
        ## get idx of train/test/val split
        train_test_val_id = np.concatenate((np.zeros(train_patch_xywh.shape[0]), np.zeros(test_patch_xywh.shape[0]), np.zeros(val_patch_xywh.shape[0])))
        train_test_val_id[test_idx] = 1
        train_test_val_id[val_idx] = 2
        sample_patch_xywh = np.concatenate((train_patch_xywh, test_patch_xywh, val_patch_xywh))
        
        ## Write to patch dictionary
        self.write_to_patch_dictionary(key='train_test_val_split_id', values=train_test_val_id,
                patches_xywh = sample_patch_xywh, overwrite=True)
        
        ## Write to spot dictionary
        if write_to_spot_dictionary:
             self.transfer_patch_to_spot(
                keys_to_transfer='train_test_val_split_id',
                overwrite=True)
                
        if return_patches:
            return self._patch_properties_dict['train_test_val_split_id'], self._patch_properties_dict['train_test_val_split_id_patch_xywh']
        
    ## TODO: way to ensure that each spot appears in a test fold exactly once
    ## TODO: e.g. run downstream task (such as CME) at patch level, then transfer to spot
    def kfold_patch_train_test_split(self, feature_xywh: str=None,
                                res: int=0.4, 
                                stratify: bool=True,
                                n_splits: int=5,
                                write_to_spot_dictionary: bool = True,
                                random_state: int = 0):
        
        """
        Split patch locations under feature into kfold train/test splits. Can stratify by patch cell composition (run compute_patch_ncv first). Useful for splitting data without spatial overlap (if patches are computed with no overlap) for downstream regression tasks.
        Writes patch train/test split ids to patch dictionary (and optionally to spot dictionary)

        Args:
            feature_xywh: Patch locations that would like to be split
            res: Leiden cluster resolution for determining patch NCV clusters
            stratify: whether to stratify by patch ncv
            n_splits: number of kfold splits to run
            write_to_spot_dictionary: whether to transfer patch train/test ids to spot and write to spot dictionary
            random_state: affects the ordering of the indices during shuffling; pass an int for reproducibility 
        """
        ## Cluster Patch NCVs
        if stratify:
            leiden_clusters = self.compute_patch_ncv_clusters(feature_xywh, res)
        
        ## run StratifiedKFold 
        if stratify:
            skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=random_state)
            n = skf.get_n_splits(self._patch_properties_dict[feature_xywh],leiden_clusters)
            for i, (train_index, test_index) in enumerate(skf.split(self._patch_properties_dict[feature_xywh],leiden_clusters)):
                train_test_id = -1 * np.ones(self._patch_properties_dict[feature_xywh].shape[0])
                train_test_id[train_index] = 0
                train_test_id[test_index] = 1
                self.write_to_patch_dictionary(key=f"train_test_split_fold_{i}_id", values=train_test_id,
                    patches_xywh = self._patch_properties_dict[feature_xywh], overwrite=True)
                
                ## Write to spot dictionary
                if write_to_spot_dictionary:
                     self.transfer_patch_to_spot(
                        keys_to_transfer=f"train_test_split_fold_{i}_id",
                        overwrite=True)
        else:
            kf = KFold(n_splits = n_splits, shuffle=True, random_state=random_state)
            n = kf.get_n_splits(self._patch_properties_dict[feature_xywh])
            # print(kf.split(self._patch_properties_dict[feature_xywh]))
            # train_index, test_index = kf.split(self._patch_properties_dict[feature_xywh])
            for i, (train_index, test_index) in enumerate(kf.split(self._patch_properties_dict[feature_xywh])):
                train_test_id = -1 * np.ones(self._patch_properties_dict[feature_xywh].shape[0])
                train_test_id[train_index] = 0
                train_test_id[test_index] = 1
                self.write_to_patch_dictionary(key=f"train_test_split_fold_{i}_id", values=train_test_id,
                    patches_xywh = self._patch_properties_dict[feature_xywh], overwrite=True)
                
                ## Write to spot dictionary
                if write_to_spot_dictionary:
                    self.transfer_patch_to_spot(
                        keys_to_transfer=f"train_test_split_fold_{i}_id",
                        strategy_patch_to_image="closest",
                        strategy_image_to_spot="closest",
                        overwrite=True)
            
        
    def get_spot_dictionary_subset_patch(self,
                                         patch_key: str,
                                         spot_keys: List[str]) -> dict:
        """
        Utility function that returns elements from spot properties dictionary that are spatially within patch coordinates of :attr:`patch_key`
        
        Args:
            patch_key: key in patch dictionary to obtain patch coordinates from
            spot_keys: keys within spot dictionary to return subsetted spot dictionary elements for
            
        Returns:
            Dictionary containg spot dictionary elements that are spatial subset of :attr:`patch_key` in patch dictionary
        """
        
        assert patch_key in self._patch_properties_dict.keys(), \
            f"Patch key {patch_key} is not in patch properties dict."
        
        ## assert spot keys is list or iterable; try converting to list first before passing error
        
        ## assert spot keys are in spot dictionary
        for spot_key in spot_keys:
            assert spot_key in self._spot_properties_dict.keys(), \
                f"Spot key {spot_key} is not in spot properties dict."
        
        # initialize empty dict
        dict_of_spot_subset_patch_dicts = {}
        
        # initialize to populate dictionary with patch key and patch xywh
        dict_of_spot_subset_patch_dicts[patch_key] = []
        dict_of_spot_subset_patch_dicts[patch_key+'_patch_xywh'] = []
        
        # initialize to populate spot subset patch dicts
        dict_of_spot_subset_patch_dicts['spot_subset_patch_dict'] = []
        
        # iterate over patches
        for patch_ind in range(len(self._patch_properties_dict[patch_key])):
            
            # add patch feature and xywh to ouptut dictionary
            dict_of_spot_subset_patch_dicts[patch_key].append(self._patch_properties_dict[patch_key][patch_ind])
            dict_of_spot_subset_patch_dicts[patch_key+'_patch_xywh'].append(self._patch_properties_dict[patch_key+'_patch_xywh'][patch_ind])
            
            x,y,w,h = self._patch_properties_dict[patch_key+'_patch_xywh'][patch_ind]
            
            # initialize spot subset patch dict
            spot_subset_patch_dict = {}
            spot_subset_patch_dict["x_key"] = []
            spot_subset_patch_dict["y_key"] = []
            for key in spot_keys:
                spot_subset_patch_dict[key] = []
            
            # convert image to spot coordinates
            patch_x_coord_lower, patch_y_coord_lower = self.pixel_to_raw(x, y)
            patch_x_coord_upper, patch_y_coord_upper = self.pixel_to_raw(x+w, y+h)
            
            for spot_ind in range(len(self._spot_properties_dict[spot_keys[0]])):

                spot_x_coord = self._spot_properties_dict["x_key"][spot_ind]
                spot_y_coord = self._spot_properties_dict["y_key"][spot_ind]
                
                # if spot coordinate is within a patch, add the corresponding spot dictionary elements to spot_subset_patch_dict
                if patch_x_coord_lower <= spot_x_coord <= patch_x_coord_upper:
                    if patch_y_coord_lower <= spot_y_coord <= patch_y_coord_upper:
                        spot_subset_patch_dict["x_key"].append(spot_x_coord)
                        spot_subset_patch_dict["y_key"].append(spot_y_coord)
                        for key in spot_keys:
                            if isinstance(self._spot_properties_dict[key], pandas.DataFrame) or isinstance(self._spot_properties_dict[key], pandas.Series):
                                spot_subset_patch_dict[key].append(self._spot_properties_dict[key].iloc[spot_ind])
                            else:
                                spot_subset_patch_dict[key].append(self._spot_properties_dict[key][spot_ind])
                            
            # add subset dict from this spot to list of dicts
            dict_of_spot_subset_patch_dicts['spot_subset_patch_dict'].append(spot_subset_patch_dict)
            
        return dict_of_spot_subset_patch_dicts               
                
        
    def transfer_patch_to_spot(
            self,
            keys_to_transfer: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy_patch_to_image: str = "average",
            strategy_image_to_spot: str = "bilinear"):
        """
        Utility function which sequentially transfer annotations from patch -> image -> spot
        """
        if verbose:
            print("transferring annotations from patch to image first")

        self.transfer_patch_to_image(
            keys_to_transfer=keys_to_transfer,
            overwrite=overwrite,
            verbose=verbose,
            strategy=strategy_patch_to_image,
        )

        if verbose:
            print("transferring annotations from image to spot last")

        self.transfer_image_to_spot(
            keys_to_transfer=keys_to_transfer,
            overwrite=overwrite,
            verbose=verbose,
            strategy=strategy_image_to_spot,
        )

    def transfer_patch_to_image(
            self,
            keys_to_transfer: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy: str = "average"):
        """
        Collect the properties computed separately for each patch and stored in patch_properties_dict to create
        an image properties which will be stored in image_properties_dict under the same name.

        Args:
            keys_to_transfer: keys of the quantity to transfer from patch_properties_dict to image_properties_dict.
                The patch_quantity can be: a scalar, a vector, a scalar field or a vector field.
                This corresponds to patch_quantity having shapes:
                (N_patches), (N_patches, ch), (N_patches, w, h) or (N_patches, ch, w, h) respectively.
            overwrite: bool, in case of collision between keys this variable controls when to overwrite the values in
                the image_properties_dict.
            strategy: str, either 'average' (default) or 'closest'. If 'average' the value of each pixel in the image
                is obtained by averaging the contribution of all patches which contain that pixel. If 'nearest' each
                pixel takes the value from the patch whose center is closets to the pixel.
            verbose: bool, if true print intermediate messages
        """
        # make sure keys_to_transfer is provided as a list
        if isinstance(keys_to_transfer, str):
            keys_to_transfer = [keys_to_transfer]
        assert isinstance(keys_to_transfer, list), \
            "Error, keys_to_transfer should be a list. Received {0}".format(type(keys_to_transfer))

        # check keys_to_transfer are present in the source
        assert set(keys_to_transfer).issubset(self._patch_properties_dict.keys()), \
            "Some keys are not present in patch_properties_dict."

        # Here is where the actual calculation starts
        for key in keys_to_transfer:
            if verbose:
                print("working on ->", key)

            patch_quantity, patch_xywh = self.read_from_patch_dictionary(key=key)

            len_shape = len(patch_quantity.shape)
            if len_shape == 1:
                # scalar property -> (N)
                ch = 1
                patch_quantity = patch_quantity[:, None, None, None]
            elif len_shape == 2:
                # vector property -> (N, ch)
                ch = patch_quantity.shape[-1]
                patch_quantity = patch_quantity[..., None, None]
            elif len_shape == 3:
                # scalar field -> (N,w,h)
                ch = 1
                patch_quantity = patch_quantity.unsqueeze(1)
            elif len_shape == 4:
                # vector field -> (N, ch, w, h)
                ch = patch_quantity.shape[-3]
            else:
                raise Exception("Can not interpret the dimension of the path_property {0} of \
                shape {1}".format(key, patch_quantity.shape))

            w_all, h_all = self.shape[-2:]
            tmp_result = torch.zeros((ch, w_all, h_all), device=patch_quantity.device, dtype=patch_quantity.dtype)
            tmp_counter = torch.zeros((w_all, h_all), device=patch_quantity.device, dtype=torch.int)
            tmp_distance: torch.Tensor = torch.ones((w_all, h_all),
                                                    device=patch_quantity.device,
                                                    dtype=torch.float) * numpy.inf
            for n, xywh in enumerate(patch_xywh):
                x, y, w, h = xywh.unbind(dim=0)
                if strategy == 'average':
                    tmp_counter[x:x+w, y:y+h] += 1
                    tmp_result[:, x:x+w, y:y+h] += patch_quantity[n]
                elif strategy == 'closest':
                    dw_from_center = torch.linspace(start=-0.5 * (w - 1), end=0.5 * (w - 1), steps=w)
                    dh_from_center = torch.linspace(start=-0.5 * (h - 1), end=0.5 * (h - 1), steps=h)
                    d2_from_center: torch.Tensor = dw_from_center[:, None].pow(2) + dh_from_center[None, :].pow(2)
                    
                    d2_from_center = d2_from_center.to(patch_quantity.device)
                    
                    mask = (d2_from_center < tmp_distance[x:x + w, y:y + h])  # shape (w, h)

                    # If the current patch has a smaller distance. Overwrite patch_quantity and tmp_distance
                    tmp_result[:, x:x+w, y:y+h] = \
                        torch.where(mask[None], patch_quantity[n], tmp_result[:, x:x+w, y:y+h])
                    tmp_distance[x:x+w, y:y+h] = torch.min(d2_from_center, tmp_distance[x:x+w, y:y+h])
                else:
                    raise ValueError("strategy can only be 'average' or 'closest'. Received {0}".format(strategy))

            if strategy == 'average':
                result = tmp_result / tmp_counter.clamp(min=1.0)
            elif strategy == 'closest':
                result = tmp_result
            else:
                raise ValueError("strategy can only be 'average' or 'closest'. Received {0}".format(strategy))

            # check that the transfer operation has generated finite values
            if not torch.all(torch.isfinite(result)):
                raise ValueError("Error. The image_property {} is not finite".format(key))

            self.write_to_image_dictionary(key=key, values=result, overwrite=overwrite)

    def transfer_image_to_spot(
            self,
            keys_to_transfer: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy: str = "bilinear"):
        """
        Evaluate the image_properties_dict at the spots location.
        Store the results in the spot_properties_dict under the same name.

        Args:
            keys_to_transfer: the keys of the quantity to transfer from image_properties_dict to spot_properties_dict.
            overwrite: bool, in case of collision between the keys this variable controls
                when the value will be overwritten.
            verbose: bool, if true intermediate messages are displayed.
            strategy: str, either 'closest' or 'bilinear' (default). This described the interpolation method.
        """
        def _interpolation(data_to_interpolate, x_float, y_float, _strategy):

            if _strategy == 'closest':
                ix_long, iy_long = torch.round(x_float).long(), torch.round(y_float).long()
                return data_to_interpolate[..., ix_long, iy_long]

            elif _strategy == 'bilinear':

                # compute the coordinates of the four corners
                x1 = torch.floor(x_float).long()
                y1 = torch.floor(y_float).long()
                x2 = x1 + 1
                y2 = y1 + 1

                # evaluate the function to interpolate at the four corner
                f11 = data_to_interpolate[..., x1, y1]
                f12 = data_to_interpolate[..., x1, y2]
                f21 = data_to_interpolate[..., x2, y1]
                f22 = data_to_interpolate[..., x2, y2]

                # compute four non-negative weights
                w11 = (x2 - x_float) * (y2 - y_float)
                w12 = (x2 - x_float) * (y_float - y1)
                w21 = (x_float - x1) * (y2 - y_float)
                w22 = (x_float - x1) * (y_float - y1)

                # actual interpolation formula.
                # In our case den is identically 1.0 therefore I can save some calculation
                # num = (w11 * f11 + w12 * f12 + w21 * f21 + w22 * f22)
                # den = ((x2 - x1) * (y2 - y1)).float()
                # result = num / den
                result = (w11 * f11 + w12 * f12 + w21 * f21 + w22 * f22)
                return result

        assert strategy == 'bilinear' or strategy == 'closest', "Invalid interpolation_method \
        Expected 'bilinear' or 'closest'. Received {0}. ".format(strategy)

        # make sure keys_to_transfer is provided as a list
        if isinstance(keys_to_transfer, str):
            keys_to_transfer = [keys_to_transfer]
        assert isinstance(keys_to_transfer, list), \
            "Error. keys_to_transfer must be a list. Received {0}".format(type(keys_to_transfer))

        assert set(keys_to_transfer).issubset(set(self._image_properties_dict.keys())), \
            "Some keys are not present in self.image_properties_dict"

        # actual calculation
        for key in keys_to_transfer:
            if verbose:
                print("working on ->", key)
            image_quantity = self.read_from_image_dictionary(key=key)

            assert isinstance(image_quantity, torch.Tensor)
            assert len(image_quantity.shape) == 3 and image_quantity.shape[-2:] == self.shape[-2:]

            x_raw = torch.from_numpy(self.x_raw).float()
            y_raw = torch.from_numpy(self.y_raw).float()
            x_pixel, y_pixel = self.raw_to_pixel(x_raw=x_raw, y_raw=y_raw)
            interpolated_values = _interpolation(
                image_quantity, x_pixel, y_pixel, strategy).cpu()
            assert len(interpolated_values.shape) == 2

            # check that the transfer operation has generated finite values
            if not torch.all(torch.isfinite(interpolated_values)):
                raise ValueError("Error. The spot_property {} is not finite".format(key))

            self.write_to_spot_dictionary(key=key, values=interpolated_values.permute(dims=(1, 0)), overwrite=overwrite)

    def transfer_spot_to_image(
            self,
            keys_to_transfer: List[str],
            overwrite: bool = False,
            verbose: bool = False,
            strategy: str = "bilinear"):
        """
        Evaluate the image_properties_dict at the spots location.
        Store the results in the image_properties_dict under the same name.

        Args:
            keys_to_transfer: the keys of the quantity to transfer from spot_properties_dict to image_properties_dict.
            overwrite: bool, in case of collision between the keys this variable controls
                when the value will be overwritten.
            verbose: bool, if true intermediate messages are displayed.
            strategy: str, either 'closest' or 'bilinear' (default). This described the interpolation method.
        """
        # make sure keys_to_transfer is provided as a list
        if isinstance(keys_to_transfer, str):
            keys_to_transfer = [keys_to_transfer]
        assert isinstance(keys_to_transfer, list), \
            "Error. keys_to_transfer must be a list. Received {0}".format(type(keys_to_transfer))

        assert set(keys_to_transfer).issubset(set(self._spot_properties_dict.keys())), \
            "Some keys are not present in self.spot_properties_dict"

        # actual calculation
        
        ## get width and height
        for key in keys_to_transfer:
            if verbose:
                print("working on ->", key)
            spot_quantity = self.read_from_spot_dictionary(key=key)

            
            assert isinstance(spot_quantity, torch.Tensor)
            #assert len(spot_quantity.shape) == 3 and spot_quantity.shape[-2:] == self.shape[-2:]

            x_raw = torch.from_numpy(self.x_raw).float()
            y_raw = torch.from_numpy(self.y_raw).float()
            x_pixel, y_pixel = self.raw_to_pixel(x_raw=x_raw, y_raw=y_raw)
            
            # Convert the coordinates and round to the closest integer

            ix = torch.round(x_pixel).long()
            iy = torch.round(y_pixel).long()

            # Create a sparse array with 1 in the correct channel and x,y location.
            # The coalesce make sure that if in case of a collision the values are summed.
            padding = self._padding
            dense_shape = (
                torch.max(ix).item() + 1 + 2 * padding,
                torch.max(iy).item() + 1 + 2 * padding
            )
            
            assert len(spot_quantity.shape) == 1, \
                "Key to transfer must be 1-D"
            
            result = torch.sparse_coo_tensor(
                indices=torch.stack((x_pixel, y_pixel)),
                values=spot_quantity,
                size=dense_shape,
                #device=,
                requires_grad=False,
                ).coalesce().to_dense()

            self.write_to_image_dictionary(key=key, values=result, overwrite=overwrite)
    
    def get_state_dict(self, include_anndata: bool = True) -> dict:
        """
        Get a dictionary with the state of the system

        Args:
            include_anndata: If True (default) the anndata is included

        Returns:
            state_dict: A dictionary with the state
        """
        state_dict = {
            'padding': self._padding,
            'pixel_size': self._pixel_size,
            'x_key': self._x_key,
            'y_key': self._y_key,
            'category_key': self._cat_key,
            'categories_to_codes': self._categories_to_codes,
            'spot_properties_dict': self._spot_properties_dict,
            'patch_properties_dict': self._patch_properties_dict,
            'image_properties_dict': self._image_properties_dict,
            'anndata': self._anndata if include_anndata else None,
        }
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> SparseImage:
        """
        Create a sparse image from the state_dictionary which was obtained by the :meth:`get_state_dict`

        Returns:
            sp_img: A sparse_image instance

        Example:
            >>> state_dict_v1 = sparse_image_old.get_state_dict(include_anndata=True)
            >>> torch.save(state_dict_v1, "ckpt.pt")
            >>> state_dict_v2 = torch.load("ckpt.pt")
            >>> sparse_image_new = SparseImage.from_state_dict(state_dict_v2)
        """
        sparse_img_obj = cls(**state_dict)
        return sparse_img_obj

    @classmethod
    def from_anndata(
            cls,
            anndata: AnnData,
            x_key: str,
            y_key: str,
            category_key: str,
            pixel_size: float = None,
            categories_to_channels: dict = None,
            padding: int = 10,
            status_key: str = "status"
    ):
        """
        Create a SparseImage object from an AnnData object.

        Note:
            The minimal adata object must have a categorical variable (such as cell_type or gene_identities)
            and the spatial coordinates. Additional fields can be present.

        Args:
            anndata: the AnnData object with the spatial data
            x_key: str, tha key associated with the x_coordinate in the AnnData object
            y_key: str, tha key associated with the y_coordinate in the AnnData object
            category_key: str, the key associated with the probability values (cell_types or gene_identities)
            pixel_size: float, pixel_size used to convert from raw coordinates to pixel coordinates.
                If it is not specified it will be chosen to be 1/3 of the median of the Nearest Neighbour distances
                between spots. Explicitely setting this attribute ensures that the pixel_size will be consistent
                across multiple images
            categories_to_channels: dictionary with the mapping from the names (of cell_types or genes) to channels.
                Explicitely setting this attribute ensures that the encoding between category
                and channels codes will be consistent across multiple images.
                If not given, the mapping will be inferred from the anndata object.
            padding: int, padding of the image so that the image has a bit of black around it

        Returns:
            sp_img: A sparse image instance

        Examples:
            >>> # create an AnnData object and a sparse image from it
            >>> anndata = AnnData(obs={"cell_type": cell_type}, obsm={"spatial": spot_xy_coordinates})
            >>> cell_types = ("ES", "Endothelial", "Leydig", "Macrophage", "Myoid", "RS", "SPC", "SPG", "Sertoli")
            >>> categories_to_codes = dict(zip(cell_types, range(len(cell_types))))
            >>> sparse_image = SparseImage.from_anndata(
            >>>     anndata=anndata,
            >>>     x_key="spatial",
            >>>     y_key="spatial",
            >>>     category_key="cell_type",
            >>>     categories_to_channels=categories_to_channels)

         Examples:
            >>> # create an AnnData object and a sparse image from it
            >>> anndata = AnnDatae(obs={"gene": gene, "x": gene_location_x, "y": gene_location_y})
            >>> sparse_image = SparseImage.from_anndata(
            >>>     anndata=anndata,
            >>>     x_key="gene_location_x",
            >>>     y_key="gene_location_y",
            >>>     category_key="gene",
            >>>     pixel_size=6.5,
            >>>     padding=8)
        """

        # extract the xy coordinates an category and build a minimal spot_dictionary
        if x_key == y_key:
            coordinates = numpy.asarray(anndata.obsm[x_key])
            assert len(coordinates.shape) == 2 and coordinates.shape[-1] == 2
            x_raw = coordinates[:, 0]
            y_raw = coordinates[:, 1]
        else:
            try:
                x_raw = numpy.asarray(anndata.obs[x_key])
            except Exception as e:
                print(e)
                x_raw = numpy.asarray(anndata.obsm[x_key])

            try:
                y_raw = numpy.asarray(anndata.obs[y_key])
            except Exception as e:
                print(e)
                y_raw = numpy.asarray(anndata.obsm[y_key])
        
        # try:
        #     cat_raw = anndata.obs[category_key]
        # except Exception as e:
        #     print(e)
        #     cat_raw = anndata.obsm[category_key]
            
        # category key must either be probabilistic assignment or one-hot encoded
        cat_raw = anndata.obsm[category_key]
        
        assert isinstance(x_raw, numpy.ndarray)
        assert isinstance(y_raw, numpy.ndarray)
        assert isinstance(cat_raw, pandas.DataFrame)
        assert x_raw.shape[0] == y_raw.shape[0] == cat_raw.shape[0] and len(x_raw.shape) == 1
        
        ## convert to float
        x_raw = x_raw.astype(float)
        y_raw = y_raw.astype(float)

        spot_dictionary = {
            "x_key": x_raw,
            "y_key": y_raw}
        
        for key in cat_raw:
            spot_dictionary[key] = cat_raw[key]
        
        for k in anndata.obs.keys():
            if k not in {x_key, y_key} and k not in cat_raw and k != category_key:
                spot_dictionary[k] = anndata.obs[k]
        for k in anndata.obsm.keys():
            if k not in {x_key, y_key} and k not in cat_raw and k != category_key:
                spot_dictionary[k] = anndata.obsm[k]
        # I have transferred all the observation I could on the spot_dict
        
        # check if anndata.uns["sparse_image_state_dict"] is present
        try:
            state_dict = anndata.uns.pop("sparse_image_state_dict")

            if status_key in anndata.uns.keys():
                sparse_img_object = cls(
                    spot_properties_dict=spot_dictionary,
                    x_key="x_key",
                    y_key="y_key",
                    category_key=category_key,
                    categories_to_codes=state_dict["categories_to_codes"],
                    pixel_size=state_dict["pixel_size"],
                    padding=state_dict["padding"],
                    patch_properties_dict=state_dict["patch_properties_dict"],
                    image_properties_dict=state_dict["image_properties_dict"],
                    anndata=anndata,
                    sample_status = anndata.uns[status_key]) 
            else:
                sparse_img_object = cls(
                    spot_properties_dict=spot_dictionary,
                    x_key="x_key",
                    y_key="y_key",
                    category_key=category_key,
                    categories_to_codes=state_dict["categories_to_codes"],
                    pixel_size=state_dict["pixel_size"],
                    padding=state_dict["padding"],
                    patch_properties_dict=state_dict["patch_properties_dict"],
                    image_properties_dict=state_dict["image_properties_dict"],
                    anndata=anndata) 
                
            return sparse_img_object

        except KeyError:

            if categories_to_channels is None:
                category_values = list(numpy.unique(cat_raw.columns))
                categories_to_channels = dict(zip(category_values, range(len(category_values))))

            assert set(cat_raw.columns).issubset(set(categories_to_channels.keys())), \
                " Error. The adata object contains values which are not present in category_values."

            # Get the pixel_size
            if pixel_size is None:
                mean_dnn, median_dnn = cls._check_mean_median_spacing(torch.tensor(x_raw), torch.tensor(y_raw))
                pixel_size = 0.25 * median_dnn
            
            if status_key in anndata.uns.keys():
                try: 
                    sparse_img_obj = cls(
                        spot_properties_dict=spot_dictionary,
                        x_key="x_key",
                        y_key="y_key",
                        category_key=category_key,
                        categories_to_codes=categories_to_channels,
                        pixel_size=pixel_size,
                        padding=padding,
                        patch_properties_dict=None,
                        image_properties_dict=None,
                        anndata=anndata,
                        sample_status = anndata.uns[status_key] 
                    )

                except KeyError:
                    sparse_img_obj = cls(
                        spot_properties_dict=spot_dictionary,
                        x_key="x_key",
                        y_key="y_key",
                        category_key=category_key,
                        categories_to_codes=categories_to_channels,
                        pixel_size=pixel_size,
                        padding=padding,
                        patch_properties_dict=None,
                        image_properties_dict=None,
                        anndata=anndata,
                        sample_status = anndata.obs[status_key][0] ## assumes anndata.obs[status_key] is all the same
                    )
            else:
                sparse_img_obj = cls(
                        spot_properties_dict=spot_dictionary,
                        x_key="x_key",
                        y_key="y_key",
                        category_key=category_key,
                        categories_to_codes=categories_to_channels,
                        pixel_size=pixel_size,
                        padding=padding,
                        patch_properties_dict=None,
                        image_properties_dict=None,
                        anndata=anndata
                    )
                

        return sparse_img_obj

    def to_anndata(self, export_full_state: bool = False, verbose: bool = False):
        """
        Export the spot_properties (and optionally the entire state dict) to the anndata object.

        Args:
            export_full_state: if True (default is False) the entire state_dict is exported into the anndata.uns
            verbose: if True (default is False) prints some intermediate statements

        Returns:
             AnnData: object containing the spot_properties_dict (and optionally the full state).

        Note:
            This will make a copy of the anndata object that was used to create the sparse image (if any)

        Examples:
            >>> adata = sparse_image.to_anndata()
            >>> sparse_image_new = SparseImage.from_anndata(adata, x_key="x", y_key="y", category_key="cell_type")
        """
        if self.anndata is None:
            minimal_spot_dict = {
                self._x_key: self.x_raw,
                self._y_key: self.y_raw,
                self._cat_key: self.cat_raw,
            }
            adata = AnnData(obs=minimal_spot_dict)
        else:
            adata = copy.deepcopy(self.anndata)

        # add the OTHER (missing) spot properties to either adata.obs or adata.obsm
        set_obs_keys = set(adata.obs_keys())
        set_obsm_keys = set(adata.obsm_keys())

        for k, v in self._spot_properties_dict.items():
            if k not in {self._x_key, self._y_key} and k not in self.anndata.obsm[self._cat_key].columns and\
                    k not in set_obs_keys and \
                    k not in set_obsm_keys:

                v_np = self._to_numpy(v)

                # squeeze if possible
                if v_np.shape[-1] == 1:
                    v_np = v_np[:, 0]

                if verbose:
                    print("working on {0} of shape {1}".format(k, v.shape))

                if len(v_np.shape) == 1:
                    adata.obs[k] = v_np
                else:
                    adata.obsm[k] = v_np

        # Add everything else to adata.uns
        if export_full_state:
            full_state_dict = self.get_state_dict(include_anndata=False)
            full_state_dict.pop("anndata")
            full_state_dict.pop("spot_properties_dict")
            adata.uns["sparse_image_state_dict"] = full_state_dict

        return adata

    @staticmethod
    def _check_mean_median_spacing(x_raw: torch.Tensor, y_raw: torch.Tensor) -> (float, float):
        """ The the mean and median distance between nearest neighbours """
        from sklearn.neighbors import KDTree
        locations = torch.stack((x_raw, y_raw), dim=-1).cpu().numpy()
        kdt = KDTree(locations, leaf_size=30, metric='euclidean')
        dist, index = kdt.query(locations, k=2, return_distance=True)
        dist_nn = dist[:, 1]
        return numpy.mean(dist_nn), numpy.median(dist_nn)

    @staticmethod
    def _to_numpy(_x):
        """ Convert to numpy """
        if isinstance(_x, numpy.ndarray):
            return _x
        elif isinstance(_x, torch.Tensor):
            return _x.cpu().numpy()
        elif isinstance(_x, list):
            return numpy.array(_x)
        else:
            raise Exception("Error. Expected torch.Tensor, numpy.array or list. Recieved {0}".format(type(_x)))

    @staticmethod
    def _to_torch(_x):
        """ Convert to torch """
        if isinstance(_x, torch.Tensor):
            return _x
        elif isinstance(_x, numpy.ndarray):
            return torch.from_numpy(_x).float().cpu()
        else:
            raise ValueError("Expect torch.Tensor or numpy.ndarray. Received {0}.".format(type(_x)))


    ##TODO: VALIDATE THESE METHODS

    def transfer_image_to_patch(
        self,
        keys_to_transfer: List[str],
        overwrite: bool = False,
        verbose: bool = False,
        strategy: str = "average",
        patch_key: str = None, ## patch key to serve as anchor for transfer
    ):
        """
        Transfer annotations from image to patch.

        Args:
            keys_to_transfer: List of keys to transfer from image_properties_dict to patch_properties_dict.
            overwrite: bool, if true, allows overwriting existing patch properties.
            verbose: bool, if true, prints intermediate messages.
            strategy: str, either 'average' (default) or 'sum'. Determines how pixel values are aggregated within patches.
        """
        # Ensure keys_to_transfer is a list
        if isinstance(keys_to_transfer, str):
            keys_to_transfer = [keys_to_transfer]
        assert isinstance(keys_to_transfer, list), \
            "Error. keys_to_transfer must be a list. Received {0}".format(type(keys_to_transfer))

        # Verify the keys exist in the image dictionary
        assert set(keys_to_transfer).issubset(self._image_properties_dict.keys()), \
            "Some keys are not present in image_properties_dict."

        for key in keys_to_transfer:
            if verbose:
                print(f"Working on -> {key}")

            # Retrieve the image-level data
            image_quantity = self.read_from_image_dictionary(key=key)

            # Ensure the data is a 3D tensor: (channels, width, height)
            # assert len(image_quantity.shape) == 3, \
            #     f"Expected 3D tensor for image properties. Got shape {image_quantity.shape} for key {key}."

            img_w, img_h = image_quantity.shape

            # Initialize patch-level storage
            patch_results = []

            assert patch_key in self._patch_properties_dict.keys(), \
                f"Error: The patch key '{patch_key}' is not present in the patch properties dictionary."
            
            assert patch_key + "_patch_xywh" in self._patch_properties_dict.keys(), \
                f"Error: The patch key '{patch_key}_patch_xywh' is not present in the patch properties dictionary."
            
            _patch_xywh = self._patch_properties_dict[patch_key + "_patch_xywh"]



            for n, xywh in enumerate(_patch_xywh):  # Iterate over patches

                x = xywh[0]
                y = xywh[1]
                w = xywh[2]
                h = xywh[3]

                # Extract the region of interest from the image
                patch_region = image_quantity[x:x+w, y:y+h]

                if strategy == "average":
                    patch_value = patch_region.mean(dim=(0, 1))  # Average over width and height
                elif strategy == "sum":
                    patch_value = patch_region.sum(dim=(0, 1))  # Sum over width and height
                elif strategy == "median":
                    patch_value = torch.median(patch_region) ## median over width and height
                else:
                    raise ValueError(f"Unsupported strategy '{strategy}'. Use 'average' or 'sum'.")

                # Append the result for the current patch
                patch_results.append(patch_value)

            # Stack patch-level results into a single tensor
            patch_results = torch.stack(patch_results, dim=0)  # Shape: (num_patches, channels)

            # Check for finite values in the results
            if not torch.all(torch.isfinite(patch_results)):
                raise ValueError(f"Error: The patch_property {key} contains non-finite values.")

            # Store the result in the patch dictionary
            self.write_to_patch_dictionary(key=key, values=patch_results, patches_xywh = _patch_xywh, overwrite=overwrite)

            if verbose:
                print(f"Successfully transferred {key} to patches.")



    def transfer_spot_to_patch(
        self,
        keys_to_transfer: List[str],
        overwrite: bool = False,
        verbose: bool = False,
        strategy_spot_to_image: str = "bilinear",
        strategy_image_to_patch: str = "average",
        patch_key: str = None, ## patch key to serve as anchor for transfer
    ):
        """
        Utility function which sequentially transfers annotations from spot -> image -> patch.

        Args:
            keys_to_transfer: List of keys to transfer from spot_properties_dict to patch_properties_dict.
            overwrite: bool, if true allows overwriting existing patch properties.
            verbose: bool, if true, prints intermediate messages.
            strategy_spot_to_image: str, interpolation strategy for spot to image transfer.
            strategy_image_to_patch: str, aggregation strategy for image to patch transfer.
        """
        if verbose:
            print("Transferring annotations from spot to image first")

        # Step 1: Transfer spot to image
        self.transfer_spot_to_image(
            keys_to_transfer=keys_to_transfer,
            overwrite=overwrite,
            verbose=verbose,
            strategy=strategy_spot_to_image,
        )

        if verbose:
            print("Transferring annotations from image to patch last")

        # Step 2: Transfer image to patch
        self.transfer_image_to_patch(
            keys_to_transfer=keys_to_transfer,
            overwrite=overwrite,
            verbose=verbose,
            strategy=strategy_image_to_patch,
            patch_key=patch_key
        )


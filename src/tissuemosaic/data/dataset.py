from typing import List, Optional, Tuple, Union, NamedTuple, Callable, Any
import torch
import collections.abc
from torch.utils.data import Dataset, DataLoader


class MetadataCropperDataset(NamedTuple):
    f_name: Union[str, int]
    loc_x: Union[int, float]
    loc_y: Union[int, float]
    moran: Union[float, list, torch.Tensor, None]
    sample_status: Union[int, None]
    composition: Union[list, None]


class CropperTensor(torch.nn.Module):
    """
    Base class for cropping a tensor and returning the crops and its coordinates, i.e. (crops, x_loc, y_loc)
    This does NOT fit into a standard transform since it returns a tuple and not just an augmented tensor.
    """

    def __init__(
        self,
        crop_size: int = 224,
        strategy: str = 'random',
        stride: int = 200,
        fraction_patch_overlap: float=0.5,
        n_crops: int = 10,
        random_order: bool = True,
        criterium_fn: Callable = None,
        **kargs,
    ):
        """
        Args:
            crop_size: int, the size in pixel of the sliding tiling windows
            strategy: str, can be either 'random' or 'tiling' or 'identity' or 'spot_level'
            stride: Used only when :attr:'strategy' is 'tiling'.
                Displacement among consecutive sliding window. This allow to control the overlap between crops.
            fraction_patch_overlap: Used only when :attr:'strategy' is 'tiling'.
                Used to compute stride. Stride parameter supersedes fraction_patch_overlap parameter if both are passed.
            n_crops: int, the size of crops to generate from a single image.
            random_order: Used only when :attr:'strategy' is 'tiling'.
                If true the crops are shuffled before being returned.
            criterium_fn: Callable which returns true if it is a valid crop, return False otherwise
        """
        
            
        super().__init__()
        self.crop_size_ = crop_size
        self.strategy_ = strategy
        self.fraction_patch_overlap_ = fraction_patch_overlap
        ## if neither fraction_patch_overlap or stride are passed, defaults to fraction_patch_overlap = 0.5. 
        ## If both stride and fraction_patch_overlap are passed, defaults to stride
        self.stride_ = stride 
        self.n_crops_ = n_crops
        self.random_order_ = random_order
        self.criterium_fn_ = criterium_fn
        self._assert_params(crop_size, stride, n_crops, random_order, strategy, criterium_fn)

    @staticmethod
    def _assert_params(crop_size, stride, n_crops, random_order, strategy, criterium_fn):
        assert isinstance(crop_size, int)
        assert isinstance(stride, int)
        assert isinstance(n_crops, int)
        assert isinstance(random_order, bool)
        assert isinstance(criterium_fn, collections.abc.Callable)
        assert strategy == 'random' or strategy == 'tiling' or strategy == 'identity'

    def forward(
            self,
            tensor: torch.Tensor,
            crop_size: int = None,
            strategy: str = None,
            stride: int = None,
            fraction_patch_overlap: float = None,
            n_crops: int = None,
            random_order: bool = None,
            criterium_fn: Callable = None) -> (List[torch.Tensor], List[int], List[int]):
        
        # All parameters default to the one used during initialization if they are not specified except for stride which is computed
        crop_size = self.crop_size_ if crop_size is None else crop_size
        strategy = self.strategy_ if strategy is None else strategy
        fraction_patch_overlap = self.fraction_patch_overlap_ if fraction_patch_overlap is None else fraction_patch_overlap
        stride = crop_size - int(crop_size * fraction_patch_overlap) if stride is None else stride
        n_crops = self.n_crops_ if n_crops is None else n_crops
        random_order = self.random_order_ if random_order is None else random_order
        criterium_fn = self.criterium_fn_ if criterium_fn is None else criterium_fn

        crops, x_locs, y_locs = self._crop(tensor, crop_size, strategy, stride, n_crops, random_order, criterium_fn)
        return crops, x_locs, y_locs

    @staticmethod
    def reapply_crops(tensor, patches_xywh) -> (List[torch.Tensor], List[int], List[int]):
        raise NotImplementedError

    def _crop(self,
              tensor,
              crop_size: int,
              strategy: str,
              stride: int,
              n_crops: int,
              random_order: bool,
              criterium_fn: Callable) -> (List[torch.Tensor], List[int], List[int]):
        """ This must be overwritten in derived class """
        raise NotImplementedError

    def __repr__(self) -> str:
        """ This must be overwritten in derived class """
        raise NotImplementedError


class CropperDenseTensor(CropperTensor):
    SAFETY_FACTOR = 3

    def __init__(self, min_threshold_value: float,  min_threshold_fraction: float, **kargs):
        """
        Args:
            min_threshold_value: binarize a crop according to
                :math:'tensor.sum(dim=-3, keepdim=True) > min_threshold_value'
            min_threshold_fraction: A crop with a fraction of True entry below this value is considered
                empty and disregarded.
        """
        assert isinstance(min_threshold_value, float)
        self.min_threshold_value = min_threshold_value 
        assert isinstance(min_threshold_fraction, float)
        self.min_threshold_fraction = min_threshold_fraction 

        def criterium_fn(potential_crops):
            masks = potential_crops.sum(dim=-3, keepdim=False) > min_threshold_value
            number_of_true = masks.flatten(start_dim=-2).sum(dim=-1)
            area_of_crops = masks.shape[-1] * masks.shape[-2]
            return number_of_true.float() > area_of_crops * min_threshold_fraction

        super().__init__(criterium_fn=criterium_fn,
                         **kargs)

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={0}, strategy={1}, stride={2}, random_order={3}, \
        min_threshold_value={4}, min_threshold_fraction={5})'.format(self.crop_size_,
                                                                     self.strategy_,
                                                                     self.stride_,
                                                                     self.random_order_,
                                                                     self.min_threshold_value,
                                                                     self.min_threshold_fraction)

    @staticmethod
    def reapply_crops(tensor, patches_xywh) -> (List[torch.Tensor], List[int], List[int]):
        assert isinstance(patches_xywh, torch.LongTensor)
        assert len(patches_xywh.shape) == 2 and patches_xywh.shape[-1] == 4
        x_patch, y_patch, w_patch, h_patch = patches_xywh.chunk(chunks=4, dim=-1)  # each one has shape (batch, 1)

        crops = []
        for ix, iy, iw, ih, in zip(x_patch, y_patch, w_patch, h_patch):
            tensor_tmp = tensor.narrow(dim=-2, start=ix.item(), length=iw.item())
            crop = tensor_tmp.narrow(dim=-1, start=iy.item(), length=ih.item())
            crops.append(crop.clone())
        return crops, x_patch.squeeze(-1).tolist(), y_patch.squeeze(-1).tolist()

    def _crop(self,
              tensor: torch.Tensor,
              crop_size: int,
              strategy: str,
              stride: int,
              n_crops: int,
              random_order: bool,
              criterium_fn: Callable) -> (List[torch.Tensor], List[int], List[int]):

        assert isinstance(tensor, torch.Tensor)
        self._assert_params(crop_size, stride, n_crops, random_order, strategy, criterium_fn)

        if strategy == 'identity':
            return [tensor]*n_crops, [0]*n_crops, [0]*n_crops

        elif strategy == 'tiling' or strategy == 'random':

            w_img, h_img = tensor.shape[-2:]
            if strategy == 'tiling':
                # generate a random starting point
                x_corner_list, y_corner_list = [], []
                i0 = torch.randint(low=0, high=stride, size=[1]).item()
                j0 = torch.randint(low=0, high=stride, size=[1]).item()
                for i in range(i0, w_img-crop_size, stride):
                    for j in range(j0, h_img-crop_size, stride):
                        x_corner_list.append(i)
                        y_corner_list.append(j)

                x_corner = torch.tensor(x_corner_list, device=tensor.device, dtype=torch.long)
                y_corner = torch.tensor(y_corner_list, device=tensor.device, dtype=torch.long)

                if random_order:
                    index_shuffle = torch.randperm(n=x_corner.shape[0], dtype=torch.long, device=x_corner.device)
                    x_corner = x_corner[index_shuffle]
                    y_corner = y_corner[index_shuffle]
            elif strategy == 'random':
                # create two tensors (x_corner, y_corner) with the location of the bottom left corner of the crop
                x_corner = torch.randint(
                    low=0,
                    high=max(1, w_img - crop_size),
                    size=[n_crops * self.SAFETY_FACTOR],
                    device=tensor.device,
                    dtype=torch.long,
                )  # low is included, high is excluded

                y_corner = torch.randint(
                    low=0,
                    high=max(1, h_img - crop_size),
                    size=[n_crops * self.SAFETY_FACTOR],
                    device=tensor.device,
                    dtype=torch.long,
                )  # low is included, high is excluded
            else:
                raise Exception("strategy is not recognized", strategy)

            # compute the crops
            crops, x_locs, y_locs = [], [], []
            for ix, iy in zip(x_corner, y_corner):
                tensor_tmp = torch.narrow(tensor, dim=-2, start=ix, length=crop_size)
                crop = torch.narrow(tensor_tmp, dim=-1, start=iy, length=crop_size)
                if self.criterium_fn_(crop):
                    crops.append(crop.clone())
                    x_locs.append(ix.item())
                    y_locs.append(iy.item())

            # return at most n_crops items
            return crops[:n_crops], x_locs[:n_crops], y_locs[:n_crops]


class CropperSparseTensor(CropperTensor):
    SAFETY_FACTOR = 5

    def __init__(self,
                 n_element_min: int = 100,
                 **kargs,
                 ):
        """
        Args:
            n_element_min: create crops with (at least) this number of elements (i.e. cells or genes)
        """
        assert isinstance(n_element_min, int)
        self.n_element_min = n_element_min

        def criterium_fn(n_elements):
            return n_elements >= n_element_min
            
        super().__init__(criterium_fn=criterium_fn,
                         **kargs)

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={0}, strategy={1}, stride={2}, random_order={3}, \
        n_element_min={4})'.format(self.crop_size_,
                                   self.strategy_,
                                   self.stride_,
                                   self.random_order_,
                                   self.n_element_min)

    @staticmethod
    def reapply_crops(sparse_tensor, patches_xywh) -> (List[torch.sparse.Tensor], List[int], List[int]):
        assert isinstance(patches_xywh, torch.Tensor)
        assert len(patches_xywh.shape) == 2 and patches_xywh.shape[-1] == 4
        assert isinstance(sparse_tensor, torch.sparse.Tensor)
        codes: torch.Tensor
        x_pixel: torch.Tensor
        y_pixel: torch.Tensor
        codes, x_pixel, y_pixel = sparse_tensor.indices()  # each has shape (n_element)
        
        values = sparse_tensor.values()
        ch, w_img, h_img = sparse_tensor.size()

        x_patch, y_patch, w_patch, h_patch = patches_xywh.chunk(chunks=4, dim=-1)  # each one has shape (batch, 1)
        
        mask = (x_pixel >= x_patch) * \
               (x_pixel < x_patch + w_patch) * \
               (y_pixel >= y_patch) * \
               (y_pixel < y_patch + h_patch)  # shape (batch, n_element)

        assert mask.shape[0] == x_patch.shape[0] == y_patch.shape[0] == w_patch.shape[0] == h_patch.shape[0]

        crops = []
        for n in range(mask.shape[0]):
            mask_n = mask[n]  # shape (n_element)
            codes_n = codes[mask_n]
            x_pixel_n = x_pixel[mask_n] - x_patch[n, 0]
            y_pixel_n = y_pixel[mask_n] - y_patch[n, 0]
            values_n = values[mask_n]

            ## assert these are ints and if not cast to ints

            crops.append(
                torch.sparse_coo_tensor(
                    indices=torch.stack((codes_n, x_pixel_n, y_pixel_n), dim=0),
                    values=values_n,
                    size=(ch, int(w_patch[n, 0]), int(h_patch[n, 0])),
                    device=x_pixel.device,
                    requires_grad=False,
                ).coalesce()
            )
        return crops, x_patch.squeeze(-1).tolist(), y_patch.squeeze(-1).tolist()

    def _crop(self,
              sparse_tensor,
              crop_size: int,
              strategy: str,
              stride: int,
              n_crops: int,
              random_order: bool,
              criterium_fn: Callable) -> Tuple[list, list, list]:

        if strategy == 'identity':
            return [sparse_tensor]*n_crops, [0]*n_crops, [0]*n_crops

        self._assert_params(crop_size, stride, n_crops, random_order, strategy, criterium_fn)
        assert sparse_tensor.is_sparse

        #print(sparse_tensor.device)
        # this might break the code if num_worked>0 in dataloader
        # if torch.cuda.is_available():
        #    sparse_tensor = sparse_tensor.cuda()

        codes, x_pixel, y_pixel = sparse_tensor.indices()  # each has shape (n_elements)
        values = sparse_tensor.values()
        
        ch, w_img, h_img = sparse_tensor.size()

        if strategy == 'tiling':
            x_corner_list, y_corner_list = [], []
            i0 = torch.randint(low=-crop_size, high=0, size=[1]).item()
            j0 = torch.randint(low=-crop_size, high=0, size=[1]).item()
            for i in range(i0, w_img, stride):
                for j in range(j0, h_img, stride):
                    x_corner_list.append(i)
                    y_corner_list.append(j)

            x_corner = torch.tensor(x_corner_list, device=x_pixel.device, dtype=x_pixel.dtype).view(-1, 1)
            y_corner = torch.tensor(y_corner_list, device=x_pixel.device, dtype=x_pixel.dtype).view(-1, 1)

            if random_order:
                index_shuffle = torch.randperm(n=x_corner.shape[0], dtype=torch.long, device=x_corner.device)
                x_corner = x_corner[index_shuffle]
                y_corner = y_corner[index_shuffle]

        elif strategy == 'random':
            x_corner = torch.randint(
                low=0,
                high=max(1, sparse_tensor.shape[-2] - crop_size),
                size=[n_crops * self.SAFETY_FACTOR],
                device=x_pixel.device,
                dtype=x_pixel.dtype,
            ).view(-1, 1)  # low is included, high is excluded

            y_corner = torch.randint(
                low=0,
                high=max(1, sparse_tensor.shape[-1] - crop_size),
                size=[n_crops * self.SAFETY_FACTOR],
                device=y_pixel.device,
                dtype=y_pixel.dtype,
            ).view(-1, 1)  # low is included, high is excluded

        else:
            raise Exception("strategy is not recognized", strategy)

        element_mask = (x_pixel >= x_corner) * \
                       (x_pixel < x_corner + crop_size) * \
                       (y_pixel >= y_corner) * \
                       (y_pixel < y_corner + crop_size)  # shape: (n_crops * SAFETY_FACTOR, n_elements)

        if torch.any(values < 1.0): ## if values are probabilistic, change to one hot for purpose of determining valid crop 
            values_oh = values.clone()
            values_oh[values_oh != 0] = 1
            
            n_elements = (values_oh * element_mask).sum(dim=-1)  # shape (n_crops * SAFETY_FACTOR)   
        else:
            n_elements = (values * element_mask).sum(dim=-1)
        
        valid_patch = criterium_fn(n_elements)
        n_valid_patches = valid_patch.sum().item()
        
        
        if n_valid_patches < n_crops and strategy == 'random':
            # import warnings
            # warnings.warn("Warning. Not enough valid crops found. Change the parameters. ")
            print("Warning. Only {0} valid crops found when requested {1}. \
            Change the parameters.".format(n_valid_patches, n_crops))
        
        dense_crop_shape = (ch, crop_size, crop_size)

        ## TODO: allow n_crops with tiling -> calculate necessary fraction patch overlap
        if strategy == 'tiling':
            crops = []
            ix = x_corner[valid_patch, 0]  # shape: n_valid_patches
            iy = y_corner[valid_patch, 0]  # shape: n_valid_patches
            mask = element_mask[valid_patch]  # shape: n_valid_patches, element_in_sparse_array
            for n in range(n_valid_patches):
                mask_n = mask[n]
                codes_n = codes[mask_n]
                x_pixel_n = x_pixel[mask_n] - ix[n]
                y_pixel_n = y_pixel[mask_n] - iy[n]
                values_n = values[mask_n]

                crops.append(
                    torch.sparse_coo_tensor(
                        indices=torch.stack((codes_n, x_pixel_n, y_pixel_n), dim=0),
                        values=values_n,
                        size=dense_crop_shape,
                        device=x_pixel.device,
                        requires_grad=False,
                    ).coalesce()
                )

            x_locs = [ix[n].item() for n in range(n_valid_patches)]
            y_locs = [iy[n].item() for n in range(n_valid_patches)]
        else:
            n_max = min(n_crops, n_valid_patches)

            ix = x_corner[valid_patch, 0][: n_max]  # shape: n_max
            iy = y_corner[valid_patch, 0][: n_max]  # shape: n_max
            mask = element_mask[valid_patch][: n_max]  # shape: n_max, element_in_sparse_array
            crops = []
            for n in range(n_max):
                mask_n = mask[n]
                codes_n = codes[mask_n]
                x_pixel_n = x_pixel[mask_n] - ix[n]
                y_pixel_n = y_pixel[mask_n] - iy[n]
                values_n = values[mask_n]

                crops.append(
                    torch.sparse_coo_tensor(
                        indices=torch.stack((codes_n, x_pixel_n, y_pixel_n), dim=0),
                        values=values_n,
                        size=dense_crop_shape,
                        device=x_pixel.device,
                        requires_grad=False,
                    ).coalesce()
                )

            x_locs = [ix[n].item() for n in range(n_max)]
            y_locs = [iy[n].item() for n in range(n_max)]
        return crops, x_locs, y_locs


class CropperDataset(Dataset):
    """
    Dataset with imgs, labels, metadata and possibly a cropper for cropping img on the fly
    """

    def __init__(
            self,
            imgs: Union[
                List[torch.Tensor],
                List[torch.sparse.Tensor],
                List["SparseImage"],
            ],
            labels: List[Any],
            metadatas: List[MetadataCropperDataset],
            cropper: Optional[CropperTensor] = None,
    ):
        """
        Args:
            imgs: (list of) images representing spatial data.
            labels: (list of) labels.
            metadatas: (list of) metadata.
            cropper: Callable which crops the image on the fly
        """
        assert isinstance(imgs, list)
        assert isinstance(labels, list)
        assert isinstance(metadatas, list)
        assert len(imgs) == len(labels) == len(metadatas), (
            "These number should be the same {0} {1} {2}".format(len(imgs),
                                                                 len(labels),
                                                                 len(metadatas))
        )
        assert len(imgs) >= 1, "I can not create a dataset with less than 1 image."

        # check that all sparse_images have a _categories_to_code before putting them together into a dataset.
        if hasattr(imgs[0], '_categories_to_codes'):
            list_of_cat_to_code_dict = [img._categories_to_codes for img in imgs]
            for i in range(len(list_of_cat_to_code_dict)-1):
                assert list_of_cat_to_code_dict[i] == list_of_cat_to_code_dict[i+1], \
                    "The sparse images have different cat_to_code dictionaries {0} and {1}. \
                    These images can not be combined into a dataset. \
                    You can re-create the sparse images and specify the cat_to_code dictionary \
                    to be used.".format(list_of_cat_to_code_dict[i], list_of_cat_to_code_dict[i+1])
            print("All cat_to_codes dictionaries are identical {0}".format(list_of_cat_to_code_dict[-1]))

        unique_y_labels = list(sorted(set(labels)))
        unique_y_codes = [i for i in range(len(unique_y_labels))]
        self._labels_to_codes = dict(zip(unique_y_labels, unique_y_codes))
        self.codes = [self._labels_to_codes[label] for label in labels]  # list of integers
        self.metadatas = metadatas
        self.imgs = imgs
        self.cropper = cropper
        if self.cropper is None:
            self.duplicating_factor = 1
            self.n_crops_per_tissue = None
        elif self.cropper.strategy_ == 'random':
            # If n_crops >= batch_size then a single tissue generates all crops for the mini_batch.
            # This results in a very imbalanced mini_batch.
            # Here, we implement a trick for generating more balanced mini_batches.
            # We pretend to have more tissues and generate fewer crops from each tissue resulting in the same overall
            # number of crops but a more diverse mini_batch.
            # See __len__ and __getitem__ for how this trick is implemented.
            tmp_n_crops = self.cropper.n_crops_
            while tmp_n_crops > 10 and tmp_n_crops % 2 == 0:
                tmp_n_crops /= 2
            self.n_crops_per_tissue = int(tmp_n_crops)
            self.duplicating_factor = int(self.cropper.n_crops_ // self.n_crops_per_tissue)

    def to(self, device: torch.device) -> "CropperDataset":
        """ Move the images to a particular device """
        self.imgs = [img.to(device) for img in self.imgs]
        return self

    def __len__(self):
        # We pretend that the dataset contains extra samples.
        # Note that the data_loader will generate RANDOM indices between 0 and this (inflated) length
        return len(self.imgs) * self.duplicating_factor

    def __getitem__(self, index: int) -> Union[
                                         Tuple[torch.Tensor, int, MetadataCropperDataset],
                                         List[Tuple[torch.Tensor, int, MetadataCropperDataset]]]:

        # Remap the index from the inflated interval to the original interval.
        new_index = index % len(self.imgs)  # this is strictly in [0, len(self.imgs))

        if self.cropper is None:
            img = self.imgs[new_index]
            code = self.codes[new_index]
            metadata = self.metadatas[new_index]
            return img, code, metadata

        else:
            code_base = self.codes[new_index]
            crop_list, loc_x_list, loc_y_list = self.cropper(self.imgs[new_index], n_crops=self.n_crops_per_tissue)

            metadata_base: MetadataCropperDataset = self.metadatas[new_index]

            return [(crop, code_base, MetadataCropperDataset(f_name=metadata_base.f_name,
                                                             loc_x=metadata_base.loc_x + x_loc,
                                                             loc_y=metadata_base.loc_y + y_loc,
                                                             moran=None,
                                                             sample_status=None,
                                                             composition=None
                                                            )) for
                    crop, x_loc, y_loc in zip(crop_list, loc_x_list, loc_y_list)]


class CollateFnListTuple:
    @staticmethod
    @torch.no_grad()
    def __call__(data):
        """
        Args:
            data: Output of the batchloader calling the __getitem__ method i.e.:
                Either: List[Tuple]
                Or: List[List[Tuple]

        Returns:
            List[imgs], List[labels], List[Metadata]
        """
        if isinstance(data, list) and isinstance(data[0], list):
            # I have to flatten a list of list
            data = [val for sublist in data for val in sublist]

        tuple_imgs, tuple_labels, tuple_metadata = zip(*data)
        return list(tuple_imgs), list(tuple_labels), list(tuple_metadata)


class DataLoaderWithLoad(DataLoader):
    def load(self, index: Union[List[int], torch.Tensor]):
        tmp = []
        for idx in index:
            tmp.append(self.dataset.__getitem__(idx))
        return self.collate_fn(tmp)

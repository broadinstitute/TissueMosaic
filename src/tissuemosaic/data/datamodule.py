from __future__ import annotations
from argparse import ArgumentParser
from argparse import Action as ArgparseAction
import numpy
import os.path
from pytorch_lightning import LightningDataModule
from anndata import read_h5ad
from typing import Dict, Callable, Optional, Tuple, List, Iterable, Any
import torch
import torchvision
from os import cpu_count
from scanpy import AnnData

from tissuemosaic.models.patch_analyzer import SpatialAutocorrelation, Composition

from .sparse_image import SparseImage
from .transforms import (
    DropoutSparseTensor,
    SparseToDense,
    TransformForList,
    Rasterize,
    RandomHFlip,
    RandomVFlip,
    RandomStraightCut,
    RandomGlobalIntensity,
    DropChannel,
    # LargestSquareCrop,
    # ToRgb,
)
from .dataset import (
    CropperDataset,
    DataLoaderWithLoad,
    CollateFnListTuple,
    MetadataCropperDataset,
    # CropperDenseTensor,
    CropperSparseTensor,
    CropperTensor,
)


# SparseTensor can not be used in dataloader using num_workers > 0.
# See https://github.com/pytorch/pytorch/issues/20248
# Therefore I put the dataset in GPU and use num_workers = 0.


class ParseDict(ArgparseAction):
    """ Make argparse able to parse a dictionary from command line """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


class SslDM(LightningDataModule):
    """
    Base class to inherit from to make a DataModule which can be used with any
    Self Supervised Learning framework
    """

    @classmethod
    def get_default_params(cls) -> dict:
        # Get the default parameters to instantiate an object
        parser = ArgumentParser()
        parser = cls.add_specific_args(parser)
        args = parser.parse_args(args=[])
        return args.__dict__

    @classmethod
    def add_specific_args(cls, parent_parser):
        # Utility functions which add parameters to argparse to simplify setting up a CLI
        raise NotImplementedError

    def get_metadata_to_regress(self, metadata) -> Dict[str, float]:
        # Extract one or more quantities to regress from the metadata """
        raise NotImplementedError

    def get_metadata_to_classify(self, metadata) -> Dict[str, int]:
        # Extract one or more quantities to classify from the metadata """
        raise NotImplementedError

    @property
    def ch_in(self) -> int:
        # How many channels will be present in the images returned by the train/test/val dataloaders?
        raise NotImplementedError

    @property
    def local_size(self) -> int:
        # Size in pixel of the local crops (used only for Dino)
        raise NotImplementedError

    @property
    def global_size(self) -> int:
        # Size in pixel of the global crops
        raise NotImplementedError

    @property
    def n_local_crops(self) -> int:
        # Number of local crops for each image to use for training (used only for Dino)
        raise NotImplementedError

    @property
    def n_global_crops(self) -> int:
        # Number of global crops for each image to use for training (used only for Dino)
        raise NotImplementedError

    @property
    def cropper_test(self) -> CropperTensor:
        # Cropper to be used at test time. This specify the cropping strategy to use at test time.
        raise NotImplementedError

    @property
    def trsfm_test(self) -> Callable:
        # Transformation to be applied at test time. This specify the data-augmentation at test time.
        raise NotImplementedError

    @property
    def cropper_train(self) -> CropperTensor:
        # Cropper to be used at train time. This specify the cropping strategy to use at train time.
        raise NotImplementedError

    @property
    def trsfm_train_local(self) -> Callable:
        # Local Transformation to be applied at train time. This specify the data augmentation for the local crops.
        # Used by Dino only.
        raise NotImplementedError

    @property
    def trsfm_train_global(self) -> Callable:
        # Global Transformation to be applied at train time. This specify the data augmentation for the global crops.
        raise NotImplementedError

    def prepare_data(self):
        # Use this to download and prepare the data.
        # These operations will be done only once in distributed settings.
        # For example, one GPU might be used to prepare data and write the results to disk so that the other
        # GPUs can read the pre-process data.
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        # Called on every GPU at the beginning of fit (train + validate), validate, test, and predict.
        # This is a good place to set the internal state, i.e. self.something = something_else
        raise NotImplementedError

    def train_dataloader(self) -> DataLoaderWithLoad:
        # Returns the train dataloader.
        raise NotImplementedError

    def val_dataloader(self) -> List[DataLoaderWithLoad]:
        # Returns the validation dataloader.
        raise NotImplementedError

    def test_dataloader(self) -> List[DataLoaderWithLoad]:
        # Returns the test dataloader.
        raise NotImplementedError

    def predict_dataloader(self) -> List[DataLoaderWithLoad]:
        # Returns the predict dataloader.
        raise NotImplementedError


class SparseSslDM(SslDM):
    """
    Datamodule for sparse Images with the parameter for the transform (i.e. data augmentation) specified.
    If you are inheriting from this class then you only have to overwrite:
    'prepara_data', 'setup', 'get_metadata_to_classify' and 'get_metadata_to_regress'.
    """
    def __init__(self,
                 global_size: int = 96,
                 local_size: int = 64,
                 n_local_crops: int = 2,
                 n_global_crops: int = 2,
                 global_scale: Tuple[float] = (0.8, 1.0),
                 local_scale: Tuple[float] = (0.5, 0.8),
                 global_intensity: Tuple[float, float] = (0.8, 1.2),
                 n_element_min_for_crop: int = 200,
                 drop_spot_probs: Tuple[float] = (0.1, 0.2, 0.3),
                 rasterize_sigmas: Tuple[float] = (1.0, 1.5),
                 occlusion_fraction: Tuple[float, float] = (0.1, 0.3),
                 drop_channel_prob: float = 0.0,
                 drop_channel_relative_freq: Iterable[float] = None,
                 n_crops_for_tissue_train: int = 50,
                 n_crops_for_tissue_test: int = 50,
                 n_cuts_for_tissue_train: int = 1,
                 fraction_patch_overlap_for_tissue_test: float = 0.0,
                 batch_size_per_gpu: int = 64,
                 **kargs):
        """
        Args:
            global_size: size in pixel of the global crops
            local_size: size in pixel of the local crops
            n_local_crops: number of global crops
            n_global_crops: number of local crops
            global_scale: in RandomResizedCrop the scale of global crops will be drawn uniformly between these values
            local_scale: in RandomResizedCrop the scale of global crops will be drawn uniformly between these values
            global_intensity: all channels will be multiplied by a number in this range
            n_element_min_for_crop: minimum number of beads/cell in a crop
            drop_spot_probs: Probability of dropping out spots (in sparse image). Should be > 0.0
            rasterize_sigmas: Possible values of the sigma of the gaussian kernel used for rasterization.
            occlusion_fraction: Fraction of the sample which is occluded is drawn uniformly between these values
            drop_channel_prob: Probability that a channel will be set to zero,
            drop_channel_relative_freq: Relative probability of each channel to be set to zero. If None (default) all
                channels are equally likely to be set to zero.
            n_crops_for_tissue_test: The number of crops in each validation epoch will be
                :math:`n_{tissue} \\times \\text{n_crops_for_tissue_test}`
            n_crops_for_tissue_train: The number of crops in each training epoch will be
                :math:`n_{tissue} \\times \\text{n_crops_for_tissue_train}`
            batch_size_per_gpu: batch size for EACH GPUs.
        """
        super(SparseSslDM, self).__init__()
        
        # params for overwriting the abstract property
        self._global_size = global_size
        self._local_size = local_size
        self._n_global_crops = n_global_crops
        self._n_local_crops = n_local_crops

        # specify the transform
        self._global_scale = global_scale
        self._local_scale = local_scale
        self._global_intensity = global_intensity
        self._drop_spot_probs = drop_spot_probs
        self._rasterize_sigmas = rasterize_sigmas
        self._occlusion_fraction = occlusion_fraction
        self._drop_channel_prob = drop_channel_prob
        self._drop_channel_relative_freq = drop_channel_relative_freq
        self._n_element_min_for_crop = n_element_min_for_crop
        self._n_crops_for_tissue_train = n_crops_for_tissue_train
        self._n_crops_for_tissue_test = n_crops_for_tissue_test
        self._n_cuts_for_tissue_train = n_cuts_for_tissue_train
        self._fraction_patch_overlap_for_tissue_test = fraction_patch_overlap_for_tissue_test

        # batch_size
        self._batch_size_per_gpu = batch_size_per_gpu
        self._dataset_train: CropperDataset = None
        self._dataset_test: CropperDataset = None

    @classmethod
    def add_specific_args(cls, parent_parser) -> ArgumentParser:
        """
        Utility functions which add parameters to argparse to simplify setting up a CLI

        Example:
            >>> import sys
            >>> import argparse
            >>> parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')
            >>> parser = SslDM.add_specific_args(parser)
            >>> args = parser.parse_args(sys.argv[1:])
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler='resolve')

        parser.add_argument("--global_size", type=int, default=96, help="size in pixel of the global crops")
        parser.add_argument("--local_size", type=int, default=64, help="size in pixel of the local crops")
        parser.add_argument("--n_global_crops", type=int, default=2, help="number of global crops")
        parser.add_argument("--n_local_crops", type=int, default=2, help="number of local crops")
        parser.add_argument("--global_scale", type=float, nargs=2, default=[0.8, 1.0],
                            help="in RandomResizedCrop the scale of global crops will be drawn uniformly \
                            between these values")
        parser.add_argument("--local_scale", type=float, nargs=2, default=[0.5, 0.8],
                            help="in RandomResizedCrop the scale of local crops will be drawn uniformly \
                            between these values")
        parser.add_argument("--global_intensity", type=float, nargs=2, default=[0.8, 1.2],
                            help="All channels will be multiplied by a value within this range")
        parser.add_argument("--n_element_min_for_crop", type=int, default=200,
                            help="minimum number of beads/cell in a crop")
        parser.add_argument("--drop_spot_probs", type=float, nargs='*', default=[0.1, 0.2, 0.3],
                            help="Probability of dropping out spots in the sparse image. Should be in (0.0, 1.0). \
                                  If a tuple is given. A random value for the tuple is chosen.")
        parser.add_argument("--rasterize_sigmas", type=float, nargs='*', default=[1.0, 1.5],
                            help="Possible values of the sigma of the gaussian kernel used for rasterization")
        parser.add_argument("--occlusion_fraction", type=float, nargs=2, default=[0.1, 0.3],
                            help="Fraction of the sample which is occluded is drawn uniformly between these values.")
        parser.add_argument("--drop_channel_prob", type=float, default=0.2,
                            help="Probability that a channel in the image will be set to zero.")
        parser.add_argument("--drop_channel_relative_freq", type=float, nargs='*', default=None,
                            help="Relative probability of each channel to be set to zero. \
                            If None, all channels have the same probability of being zero")
        parser.add_argument("--n_crops_for_tissue_train", type=int, default=50,
                            help="The number of crops in each training epoch will be: n_tissue * n_crops. \
                               Set small for rapid prototyping")
        parser.add_argument("--n_cuts_for_tissue_train", type=int, default=1,
                            help="The number of Random Straight Cuts to apply during training transform. \
                               Set small for rapid prototyping")
        parser.add_argument("--fraction_patch_overlap_for_tissue_test", type=float, default=0.0,
                            help="The stride size for generating tissue crops will be: global_size - global_size * fraction_patch_overlap_for_tissue_test. \
                               Set large for more accurate patch features. Set small for more disjoint patches.")
        parser.add_argument("--batch_size_per_gpu", type=int, default=64,
                            help="Batch size for EACH GPUs. Set small for rapid prototyping. \
                            The total batch_size will increase linearly with the number of GPUs.")
        return parser

    @property
    def global_size(self) -> int:
        """
        Size in pixel of the global crops.
        This specify the size of the patch processed by the ssl model.
        """
        return self._global_size

    @property
    def local_size(self) -> int:
        """
        Size in pixel of the local crops (used only for Dino).
        This specify the size of the patch processed by the ssl model.
        """
        return self._local_size

    @property
    def n_global_crops(self) -> int:
        """ Number of global crops for each image to use for training (used only for Dino). """
        return self._n_global_crops

    @property
    def n_local_crops(self) -> int:
        """ Number of local crops for each image to use for training (used only for Dino). """
        return self._n_local_crops
    
    @property
    def n_element_min_for_crop(self) -> int:
        """
        Minimum number of beads/cells in a crop
        """
        return self._n_element_min_for_crop
    
    @property
    def cropper_test(self) -> CropperSparseTensor:
        """ Cropper to be used at test time. This specify the cropping strategy to use at test time. """
        return CropperSparseTensor(
            strategy='random',
            crop_size=self._global_size,
            n_element_min=self._n_element_min_for_crop,
            frac_overlap = self._fraction_patch_overlap_for_tissue_test,
            n_crops=self._n_crops_for_tissue_test,
            random_order=True,
        )

    @property
    def cropper_train(self) -> CropperSparseTensor:
        """ Cropper to be used at train time. This specify the cropping strategy to use at train time. """
        return CropperSparseTensor(
            strategy='random',
            crop_size=int(self._global_size * 1.5),
            n_element_min=int(self._n_element_min_for_crop * 1.5 * 1.5),
            n_crops=self._n_crops_for_tissue_train,
            random_order=True,
        )

    @property
    def trsfm_test(self) -> TransformForList:
        """ Transformation to be applied at test time. This specify the data-augmentation at test time. """
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                SparseToDense(),
                Rasterize(sigmas=self._rasterize_sigmas, normalize=False)
            ]),
            transform_after_stack=torchvision.transforms.CenterCrop(size=self.global_size),
        )
    
    @property
    def trsfm_train_global(self) -> TransformForList:
        """
        Global Transformation to be applied at train time.
        This specify the data augmentation for the global crops.
        """
        
        ## assumes same occlusion fraction for each straight cut
        if self._n_cuts_for_tissue_train == 1:
            n_RandomStraightCuts = RandomStraightCut(p=0.5, occlusion_fraction=self._occlusion_fraction)
        elif self._n_cuts_for_tissue_train == 2:
            n_RandomStraightCuts = torchvision.transforms.Compose([RandomStraightCut(p=0.5, occlusion_fraction=self._occlusion_fraction), RandomStraightCut(p=0.25, 
                                                                    occlusion_fraction=self._occlusion_fraction)])
        else:
            raise Exception("Only 1 or 2 cuts for tissue train are allowed.")
            
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=1.0, dropout_rate=self._drop_spot_probs),
                SparseToDense(),
                RandomGlobalIntensity(f_min=self._global_intensity[0], f_max=self._global_intensity[1])
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self._global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self._global_size, self._global_size),
                    scale=self._global_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                n_RandomStraightCuts,
                DropChannel(p=self._drop_channel_prob, relative_frequency=self._drop_channel_relative_freq),
                Rasterize(sigmas=self._rasterize_sigmas, normalize=False),
            ])
        )

    @property
    def trsfm_train_local(self) -> TransformForList:
        """
        Local Transformation to be applied at train time. This specify the data augmentation for the local crops.
        Used by Dino only.
        """
        return TransformForList(
            transform_before_stack=torchvision.transforms.Compose([
                DropoutSparseTensor(p=1.0, dropout_rate=self._drop_spot_probs),
                SparseToDense(),
                RandomGlobalIntensity(f_min=self._global_intensity[0], f_max=self._global_intensity[1])
            ]),
            transform_after_stack=torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(
                    degrees=(-180.0, 180.0),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0),
                torchvision.transforms.CenterCrop(size=self.global_size),
                RandomVFlip(p=0.5),
                RandomHFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(
                    size=(self._local_size, self._local_size),
                    scale=self._local_scale,
                    ratio=(0.95, 1.05),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                RandomStraightCut(p=0.5, occlusion_fraction=self._occlusion_fraction),
                DropChannel(p=self._drop_channel_prob, relative_frequency=self._drop_channel_relative_freq),
                Rasterize(sigmas=self._rasterize_sigmas, normalize=False),
            ])
        )
    

    def train_dataloader(self) -> DataLoaderWithLoad:
        try:
            device = self.trainer._model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # print("Inside train_dataloader", device)
        
        assert isinstance(self._dataset_train, CropperDataset)
        if self._dataset_train.n_crops_per_tissue is None:
            batch_size_dataloader = self._batch_size_per_gpu
        else:
            batch_size_dataloader = max(1, int(self._batch_size_per_gpu // self._dataset_train.n_crops_per_tissue))
        
        dataloader_train = DataLoaderWithLoad(
            # move the dataset to GPU so that the cropping happens there
            dataset=self._dataset_train.to(device),
            # each sample generate n_crops therefore reduce batch_size
            batch_size=batch_size_dataloader,
            collate_fn=CollateFnListTuple(),
            # problem if this is larger than 0, see https://github.com/pytorch/pytorch/issues/20248
            num_workers=0,
            # in the train dataloader, I DO shuffle and drop the last partial_batch
            shuffle=True,
            drop_last=True,
        )
        return dataloader_train

    def val_dataloader(self) -> List[DataLoaderWithLoad]:  # the same as test
        try:
            device = self.trainer._model.device
        except AttributeError:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        assert isinstance(self._dataset_test, CropperDataset)
        if self._dataset_test.n_crops_per_tissue is None:
            batch_size_dataloader = self._batch_size_per_gpu
        else:
            batch_size_dataloader = max(1, int(self._batch_size_per_gpu // self._dataset_train.n_crops_per_tissue))

        assert isinstance(self._dataset_test, CropperDataset)
        test_dataloader = DataLoaderWithLoad(
            # move the dataset to GPU so that the cropping happens there
            dataset=self._dataset_test.to(device),
            # each sample generate n_crops therefore reduce batch_size
            batch_size=batch_size_dataloader,
            collate_fn=CollateFnListTuple(),
            # problem if num_workers > 0, see https://github.com/pytorch/pytorch/issues/20248
            num_workers=0,
            # in the test dataloader, I do NOT shuffle and do not drop the last partial_batch
            shuffle=False,
            drop_last=False,
        )
        return [test_dataloader]

    def test_dataloader(self) -> List[DataLoaderWithLoad]:
        return self.val_dataloader()

    def predict_dataloader(self) -> List[DataLoaderWithLoad]:
        return self.val_dataloader()

    def prepare_data(self):
        raise NotImplementedError

    def get_metadata_to_classify(self, metadata) -> Dict[str, int]:
        raise NotImplementedError

    def get_metadata_to_regress(self, metadata) -> Dict[str, float]:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        self._dataset_train = None
        self._dataset_test = None
        raise NotImplementedError

class AnndataFolderDM(SparseSslDM):
    """
    Create a Datamodule ready for Self-supervised learning starting
    from a folder full of anndata files in .h5ad format.
    """
    def __init__(self,
                 data_folder: str,
                 pixel_size: float,
                 x_key: str,
                 y_key: str,
                 category_key: str,
                 categories_to_channels: Dict[Any, int],
                 status_key: str,
                 metadata_to_classify: Callable,
                 metadata_to_regress: Callable,
                 num_workers: int,
                 gpus: int,
                 n_neighbours_moran: int,
                 **kargs):
        """
        Args:
            data_folder: path to folder with the anndata in h5ad format
            pixel_size: size of the pixel (used to convert raw_coordinates to pixel_coordinates)
            x_key: key associated with the x_coordinate in the AnnData object
            y_key: key associated with the y_coordinate in the AnnData object
            category_key: key associated with the assignment probabilities (cell_types or gene_identities; can be one-hot encoded for categorical assignments)
                in the AnnData object
            categories_to_channels: dictionary with the mapping from categorical values to channels in the image.
                The values must be non-negative integers
            metadata_to_classify: callable which defines the values to classify during training
            metadata_to_regress: callable which defines the values to regress during training
            num_workers: number of worker to load data. Meaningful only if dataset is on disk.
                Set to zero if data in memory
            gpus: number of gpus to use for training.
            n_neighbours_moran: number of neighbours used to compute Moran's I score
            kargs: all these parameters will be passed to :class:`SparseSslDM`
        """ 
            
        assert isinstance(categories_to_channels, dict) and len(categories_to_channels.keys()) >= 1, \
            "Error. Specify a valid categories_to_channels mapping. Received {}".format(categories_to_channels)

        set_chs = set(categories_to_channels.values())
        set_chs_should_be = set([i for i in range(max(set_chs)+1)])
        assert set_chs == set_chs_should_be, \
            "The values of the categories_to_channels must be integers starting at zero. Received {}".format(set_chs)

        self._data_folder = data_folder
        self._pixel_size = pixel_size
        self._x_key = x_key
        self._y_key = y_key
        self._category_key = category_key
        self._categories_to_channels = categories_to_channels
        self._metadata_to_regress = metadata_to_regress
        self._metadata_to_classify = metadata_to_classify
        self._status_key = status_key

        self._num_workers = cpu_count() if num_workers is None else num_workers
        self._gpus = torch.cuda.device_count() if gpus is None else gpus
        self._n_neighbours_moran = n_neighbours_moran

        # Callable on dataset
        self.compute_moran = SpatialAutocorrelation(
            modality='moran',
            n_neighbours=self._n_neighbours_moran,
            neigh_correct=False)

        # list of all the files used to create the dataset
        self._all_filenames = None

        super(AnndataFolderDM, self).__init__(**kargs)

    @classmethod
    def add_specific_args(cls, parent_parser) -> ArgumentParser:
        """
        Utility functions which add parameters to argparse to simplify setting up a CLI

        Example:
            >>> import sys
            >>> import argparse
            >>> parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')
            >>> parser = AnndataFolderDM.add_specific_args(parser)
            >>> args = parser.parse_args(sys.argv[1:])
        """
        parser_from_super = super().add_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser_from_super], add_help=False, conflict_handler='resolve')

        parser.add_argument("--data_folder", type=str, default="./",
                            help="directory where to find the anndata in h5ad format")
        parser.add_argument("--pixel_size", type=float, default=4.0,
                            help="size of the pixel (used to convert raw_coordinates to pixel_coordinates)")
        parser.add_argument("--x_key", type=str, default="x",
                            help="key associated with the x_coordinate in the AnnData object")
        parser.add_argument("--y_key", type=str, default="y",
                            help="key associated with the y_coordinate in the AnnData object")
        parser.add_argument("--category_key", type=str, default="cell_type",
                            help="key associated with the the probability values (cell_types or gene_identities; can be one-hot encoded) \
                            in the AnnData object")
        parser.add_argument("--status_key", type=str, default="status",
                            help="keys associated with sample status, located in anndata.uns")
        parser.add_argument("--weights_key", type=str, default="cell_type_proportions",
                    help="obsm key for weights in each channel")
        parser.add_argument("--categories_to_channels", nargs='*', action=ParseDict,
                            help="dictionary in the form 'foo'=1 'bar'=2 to define \
                            how the categorical values are mapped to the different channels in the image")
        parser.add_argument("--metadata_to_classify", default=None,
                            help="callable which defines the values to classify during training")
        parser.add_argument("--metadata_to_regress", default=None,
                            help="callable which defines the values to regress during training")
        parser.add_argument("--num_workers", default=cpu_count(), type=int,
                            help="number of worker to load data. Meaningful only if dataset is on disk. \
                            Set to zero if data in memory")
        parser.add_argument("--gpus", default=torch.cuda.device_count(), type=int,
                            help="number of gpus to use for training.")
        parser.add_argument("--n_neighbours_moran", type=int, default=6,
                            help="number of neighbours used to compute Moran's I score")
        return parser

    @property
    def ch_in(self) -> int:
        """ How many channels will be present in the images returned by the train/test/val dataloaders? """
        return numpy.max(list(self._categories_to_channels.values())) + 1

    def anndata_to_sparseimage(self, anndata: AnnData):
        """ Converts a anndata object to :class:`SparseImage`. """
        return SparseImage.from_anndata(
            anndata=anndata,
            x_key=self._x_key,
            y_key=self._y_key,
            category_key=self._category_key,
            pixel_size=self._pixel_size,
            categories_to_channels=self._categories_to_channels,
            status_key = self._status_key,
            padding=10)

    def prepare_data(self):
        # create train_dataset_random and write to file
        all_metadatas = []
        all_sparse_images = []
        all_labels = []

        for filename in os.listdir(self._data_folder):
            f = os.path.join(self._data_folder, filename)
            # checking if it is a file
            if os.path.isfile(f) and filename.endswith('h5ad'):
                print("reading file {}".format(f))
                
                # import psutil
                # print(psutil.virtual_memory())
                
                anndata = read_h5ad(filename=f)
                
                anndata.X = None  # set the count matrix to None
                sp_img = self.anndata_to_sparseimage(anndata=anndata).cpu()
                all_sparse_images.append(sp_img)
                metadata = MetadataCropperDataset(f_name=filename, loc_x=0.0, loc_y=0.0, moran=-99, sample_status = 0, composition=None) ## dummy metadata
                all_metadatas.append(metadata)

                all_labels.append(filename)
                
                del anndata ## delete anndata after converting to sparse image

        self._all_filenames: list = all_labels

        torch.save((all_sparse_images, all_labels, all_metadatas),
                   os.path.join(self._data_folder, "train_dataset.pt"))
        print("saved the file", os.path.join(self._data_folder, "train_dataset.pt"))

        # create test_dataset_random and write to file
        all_names = [metadata.f_name for metadata in all_metadatas]

        if torch.cuda.is_available():
            all_sparse_images = [sp_img.cuda() for sp_img in all_sparse_images]

        test_imgs, test_labels, test_metadatas = [], [], []
        for sp_img, label, fname in zip(all_sparse_images, all_labels, all_names):
            
            sps_tmp, loc_x_tmp, loc_y_tmp = self.cropper_test(sp_img, fraction_patch_overlap = self._fraction_patch_overlap_for_tissue_test) #n_crops=self._n_crops_for_tissue_test)
            labels = [label] * len(sps_tmp)

            ### add majority cell type label 
            morans = [self.compute_moran(sparse_tensor).max().item() for sparse_tensor in sps_tmp]
            statuses = [sp_img._sample_status for sparse_tensor in sps_tmp] ## same status for all patches in this sp img
            list_composition = Composition(return_fraction=True)(sps_tmp)
            metadatas = [MetadataCropperDataset(f_name=fname, loc_x=loc_x, loc_y=loc_y, moran=moran, sample_status=status, composition=composition) for
                     loc_x, loc_y, moran, status, composition in zip(loc_x_tmp, loc_y_tmp, morans, statuses,list_composition)] 
            
            # metadatas = MetadataCropperDataset(f_name=filename, loc_x=0.0, loc_y=0.0, moran=-99, sample_status = 0, composition=None)
            
            test_imgs += [sp_img.cpu() for sp_img in sps_tmp]
            test_labels += labels
            test_metadatas += metadatas

        torch.save((test_imgs, test_labels, test_metadatas), os.path.join(self._data_folder, "test_dataset.pt"))
        print("saved the file", os.path.join(self._data_folder, "test_dataset.pt"))

    def get_metadata_to_classify(self, metadata) -> Dict[str, int]:
        """ Extract one or more quantities to classify from the metadata """
        if self._metadata_to_classify is None:
            return {"tissue_label": self._all_filenames.index(metadata.f_name), "sample_status": int(metadata.sample_status)}
        else:
            return self._metadata_to_classify(metadata)

    def get_metadata_to_regress(self, metadata) -> Dict[str, float]:
        """ Extract one or more quantities to regress from the metadata """
        if self._metadata_to_regress is None:
            
            regress_dict = {
                    "loc_x": float(metadata.loc_x),
                    "loc_y": float(metadata.loc_y)
                }
            
            if isinstance(metadata.moran,list) or isinstance(metadata.moran,torch.Tensor):
                for ch in range(len(metadata.moran)):
                    regress_dict["moran_ch_" + str(ch)] = metadata.moran[ch].item()

                regress_dict["moran"] = float(metadata.moran.max().item())
                
            else:
                regress_dict["moran"] = float(metadata.moran)

            for ch in range(len(metadata.composition)):
                regress_dict["ch_" + str(ch)] = metadata.composition[ch].item()
                
            return regress_dict
        else:
            return self._metadata_to_regress(metadata)

    def setup(self, stage: Optional[str] = None) -> None:
        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self._data_folder, "train_dataset.pt"))
        print("read the file {}".format(os.path.join(self._data_folder, "train_dataset.pt")))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self._dataset_train = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=self.cropper_train,
        )
        print("created train_dataset device = {0}, length = {1}".format(self._dataset_train.imgs[0].device,
                                                                        self._dataset_train.__len__()))

        list_imgs, list_labels, list_metadata = torch.load(os.path.join(self._data_folder, "test_dataset.pt"))
        print("read the file {}".format(os.path.join(self._data_folder, "test_dataset.pt")))
        list_imgs = [img.coalesce().cpu() for img in list_imgs]
        self._dataset_test = CropperDataset(
            imgs=list_imgs,
            labels=list_labels,
            metadatas=list_metadata,
            cropper=None,
        )
        print("created test_dataset device = {0}, length = {1}".format(self._dataset_test.imgs[0].device,
                                                                       self._dataset_test.__len__()))

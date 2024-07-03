import pytest
import torch
from torch.utils.data import DataLoader
from tissuemosaic.data_utils.sparse_image import SparseImage
from tissuemosaic.data_utils.dataset import CropperDataset
from tissuemosaic.data_utils.datamodule import DummyDM


@pytest.mark.parametrize("config_type", ["default", "manual"])
def test_dummy_dm(config_type, capsys):
    if config_type == 'default':
        dm = DummyDM()
    elif config_type == 'manual':
        config = {
            'batch_size_per_gpu': 10,
            'n_crops_for_tissue_test': 10,
            'n_crops_for_tissue_train': 10,
            'n_element_min_for_crop': 1,
            'global_size': 64,
            'global_scale': (0.75, 1.0),
            'local_size': 32,
            'local_scale': (0.5, 0.75),
        }
        dm = DummyDM(**config)
        assert dm._batch_size_per_gpu == 10

    # with capsys.disabled():
    #     print(vars(dm))

    dm.prepare_data()
    dm.setup(stage=None)

    # test the train_loader
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    batch = next(iter(train_loader))
    assert isinstance(batch, tuple)
    list_of_tensors = batch[0]
    assert isinstance(list_of_tensors, list) and 1 <= len(list_of_tensors) <= dm._batch_size_per_gpu
    assert isinstance(list_of_tensors[0], torch.sparse.Tensor)

    # check the transforms
    train_imgs_global = dm.trsfm_train_global(list_of_tensors)
    train_imgs_local = dm.trsfm_train_local(list_of_tensors)
    assert train_imgs_global.shape[-3:] == torch.Size([dm.ch_in, dm.global_size, dm.global_size])
    assert train_imgs_local.shape[-3:] == torch.Size([dm.ch_in, dm.local_size, dm.local_size])
    train_dataset = train_loader.dataset
    assert isinstance(train_dataset, CropperDataset)

    sp_images = train_dataset.imgs
    assert isinstance(sp_images[0], SparseImage) or \
           isinstance(sp_images[0], torch.sparse.Tensor) or \
           isinstance(sp_images[0], torch.Tensor)
    assert sp_images[0].shape[-3] == dm.ch_in

    # test the val_loader
    test_loaders = dm.val_dataloader()
    assert isinstance(test_loaders, list)
    assert isinstance(test_loaders[0], DataLoader)

    batch = next(iter(test_loaders[0]))
    assert isinstance(batch, tuple)
    list_of_tensors = batch[0]
    assert isinstance(list_of_tensors, list)
    assert isinstance(list_of_tensors[0], torch.sparse.Tensor)

    test_dataset = test_loaders[0].dataset
    assert isinstance(test_dataset, CropperDataset)

    sp_images = test_dataset.imgs
    assert isinstance(sp_images[0], SparseImage) or \
           isinstance(sp_images[0], torch.sparse.Tensor) or \
           isinstance(sp_images[0], torch.Tensor)
    assert sp_images[0].shape[-3] == dm.ch_in
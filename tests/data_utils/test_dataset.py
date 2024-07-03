import pytest
import torch
from torch.utils.data import TensorDataset
from tissuemosaic.data_utils.dataset import (
    AddFakeMetadata,
    MetadataCropperDataset,
    CropperDenseTensor,
    CropperSparseTensor)


def test_metadata_cropper_dataset_construction():
    """ Construct MetadataCropperDataset """
    assert MetadataCropperDataset(f_name="a", loc_x=0, loc_y=0, moran=-99)


def test_add_fake_metadata():
    """
    Starting from a standard dataset with 2 outputs (imgs, labels)
    I can create a dataset with 3 outputs (imgs, labels, metadata)
    """
    b, n = 10, 2
    x = torch.zeros((b, n))
    y = torch.arange(b)
    dataset = TensorDataset(x, y)
    new_dataset = AddFakeMetadata(dataset)

    batch = new_dataset.__getitem__(index=0)
    assert len(batch) == 3
    assert isinstance(batch[2], MetadataCropperDataset)


@pytest.mark.parametrize("strategy", ["identity", "tiling", "random"])
def test_cropper_dense_tensor(strategy):
    """ Test cropping on dense tensors """
    crop_size = 20
    stride = 10
    n_crops = 10
    c, w, h = 9, 300, 300
    tensor = torch.randn((c, w, h))
    cropper = CropperDenseTensor(strategy=strategy,
                                 min_threshold_value=0.0,
                                 min_threshold_fraction=0.5,
                                 crop_size=crop_size)

    # test the cropping
    crops, x_locs, y_locs = cropper.forward(tensor, strategy=strategy, stride=stride, n_crops=n_crops)
    crops_batched = torch.stack(crops, dim=-4)
    assert isinstance(crops, list) and isinstance(crops[0], torch.Tensor)
    assert isinstance(x_locs, list) and isinstance(x_locs[0], int)
    assert isinstance(y_locs, list) and isinstance(y_locs[0], int)
    assert len(crops) == len(x_locs) == len(y_locs) == n_crops

    if strategy == "identity":
        assert crops_batched.shape == torch.Size([n_crops, c, w, h])
    else:
        assert crops_batched.shape == torch.Size([n_crops, c, crop_size, crop_size])

    # test reapply crops
    patches_xywh = torch.zeros((len(crops), 4), dtype=torch.long)
    patches_xywh[:, 0] = torch.tensor(x_locs)
    patches_xywh[:, 1] = torch.tensor(y_locs)
    patches_xywh[:, 2] = crops_batched.shape[-2]
    patches_xywh[:, 3] = crops_batched.shape[-1]
    new_crops, new_x_locs, new_y_locs = cropper.reapply_crops(tensor, patches_xywh)
    new_crops_batched = torch.stack(new_crops, dim=0)
    assert isinstance(new_crops, list) and isinstance(new_crops[0], torch.Tensor)
    assert isinstance(new_x_locs, list) and isinstance(new_x_locs[0], int)
    assert isinstance(new_crops, list) and isinstance(new_y_locs[0], int)
    assert torch.all(new_crops_batched == crops_batched)


@pytest.mark.parametrize("strategy", ["identity", "tiling", "random"])
def test_cropper_sparse_tensor(strategy):
    """ test cropping of sparse tensors """
    crop_size = 20
    stride = 10
    n_crops = 10
    n_elements = 2000
    c, w, h = 9, 300, 300
    values = torch.randint(low=1, high=3, size=[n_elements])
    indices_ch = torch.randint(low=0, high=c, size=[n_elements])
    indices_x = torch.randint(low=0, high=w, size=[n_elements])
    indices_y = torch.randint(low=0, high=h, size=[n_elements])
    indices = torch.stack((indices_ch, indices_x, indices_y), dim=0)
    sparse_tensor = torch.sparse_coo_tensor(indices=indices,
                                            values=values,
                                            size=(c, w, h)).coalesce()
    cropper = CropperSparseTensor(strategy=strategy,
                                  n_element_min=10,
                                  crop_size=crop_size)

    # test the cropping
    sparse_crops, x_locs, y_locs = cropper.forward(sparse_tensor, strategy=strategy, stride=stride, n_crops=n_crops)
    crops_batched = torch.stack([tmp.to_dense() for tmp in sparse_crops], dim=-4)
    assert isinstance(sparse_crops, list) and isinstance(sparse_crops[0], torch.sparse.Tensor)
    assert isinstance(x_locs, list) and isinstance(x_locs[0], int)
    assert isinstance(y_locs, list) and isinstance(y_locs[0], int)
    assert len(sparse_crops) == len(x_locs) == len(y_locs) == n_crops

    if strategy == "identity":
        assert crops_batched.shape == torch.Size([n_crops, c, w, h])
    else:
        assert crops_batched.shape == torch.Size([n_crops, c, crop_size, crop_size])

    # test reapply crops
    patches_xywh = torch.zeros((len(sparse_crops), 4), dtype=torch.long)
    patches_xywh[:, 0] = torch.tensor(x_locs)
    patches_xywh[:, 1] = torch.tensor(y_locs)
    patches_xywh[:, 2] = crops_batched.shape[-2]
    patches_xywh[:, 3] = crops_batched.shape[-1]
    new_sparse_crops, new_x_locs, new_y_locs = cropper.reapply_crops(sparse_tensor, patches_xywh)
    new_crops_batched = torch.stack([tmp.to_dense() for tmp in new_sparse_crops], dim=0)
    assert isinstance(new_sparse_crops, list) and isinstance(new_sparse_crops[0], torch.sparse.Tensor)
    assert isinstance(new_x_locs, list) and isinstance(new_x_locs[0], int)
    assert isinstance(new_sparse_crops, list) and isinstance(new_y_locs[0], int)
    assert torch.all(new_crops_batched == crops_batched)


@pytest.mark.skip("not implemented yet")
@pytest.mark.parametrize("input_type", ["tensor", "sparse_tensor", "sparse_image"])
def test_cropper_dataset(input_type):
    assert True


@pytest.mark.skip("not implemented yet")
def test_collate_function():
    """ Test collate function on List[Tuple] and List[List[Tuple]] """
    assert True

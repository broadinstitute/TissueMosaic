import torch
from tissuemosaic.data_utils.sparse_image import (
    SparseImage
)
from tissuemosaic.misc_utils.dict_util import are_dicts_equal
from scanpy import AnnData


def _are_state_dict_identical(state_dict1, state_dict2):
    """ Helper function to compare state_dict which are nested dict """
    assert set(state_dict1.keys()) == set(state_dict2.keys())

    for k in set(state_dict1.keys()):
        v1 = state_dict1[k]
        v2 = state_dict2[k]
        if isinstance(v1, AnnData):
            pass
        elif isinstance(v1, dict):
            if not are_dicts_equal(v1, v2):
                return False
        else:
            if not (v1 == v2):
                return False
    return True


def test_construction_from_adata(ann_data):
    """ Test constructing sparse_image from anndata saving and reloading """
    assert isinstance(ann_data, AnnData)
    sparse_image = SparseImage.from_anndata(ann_data,
                                            x_key='x_raw',
                                            y_key='y_raw',
                                            category_key='cell_type')
    assert isinstance(sparse_image, SparseImage)

    state_dict = sparse_image.get_state_dict(include_anndata=True)
    new_sparse_image = SparseImage.from_state_dict(state_dict)
    new_state_dict = new_sparse_image.get_state_dict(include_anndata=True)
    assert isinstance(new_sparse_image, SparseImage)
    assert _are_state_dict_identical(state_dict, new_state_dict)


def test_sparse_image_to_anndata(ann_data, capsys):
    """ Test that I can save a sparse image to ann_data object """
    sparse_image = SparseImage.from_anndata(ann_data,
                                            x_key='x_raw',
                                            y_key='y_raw',
                                            category_key='cell_type')
    new_ann_data = sparse_image.to_anndata(export_full_state=True)
    new_sparse_image = SparseImage.from_anndata(new_ann_data,
                                                x_key='x_raw',
                                                y_key='y_raw',
                                                category_key='cell_type')
    new_state_dict = new_sparse_image.get_state_dict(include_anndata=False)
    old_state_dict = sparse_image.get_state_dict(include_anndata=False)
    # compare a dict of dict
    assert _are_state_dict_identical(new_state_dict, old_state_dict)


def test_sparse_image_state_dict(sparse_image):
    """ Test I can save and recreate image based on state_dict """
    assert isinstance(sparse_image, SparseImage)
    state_dict = sparse_image.get_state_dict(include_anndata=False)
    new_sparse_image = SparseImage.from_state_dict(state_dict)
    new_state_dict = new_sparse_image.get_state_dict(include_anndata=False)
    assert isinstance(new_sparse_image, SparseImage)
    assert state_dict == new_state_dict





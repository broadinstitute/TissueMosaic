import pytest
import pandas
import random
import numpy
import torch
from scanpy import AnnData
from scipy import stats
from scipy.sparse import random as random_sparse
from tissuemosaic.data.sparse_image import SparseImage
#from tissuemosaic.data.datamodule import DummyDM
from tissuemosaic.models.logger import NeptuneLoggerCkpt
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from tissuemosaic.data.dataset import MetadataCropperDataset


def _random_string_generator(str_size, allowed_chars=None):
    if allowed_chars is None:
        allowed_chars = list('abcdefghijklmnopqrstuvxywz')
    return ''.join(random.choice(allowed_chars) for x in range(str_size))


# @pytest.fixture
# def dummy_dino_dm():
#     config = {
#         'batch_size_per_gpu': 10,
#         'n_crops_for_tissue_test': 10,
#         'n_crops_for_tissue_train': 10,
#         'n_element_min_for_crop': 1,
#         'global_size': 64,
#         'global_scale': (0.75, 1.0),
#         'local_size': 32,
#         'local_scale': (0.5, 0.75),
#     }
#     return DummyDM(**config)


@pytest.fixture
def neptune_logger_async():
    return NeptuneLoggerCkpt(
        project="cellarium/tissue-mosaic",
        run=None,
        log_model_checkpoints=True,
        mode="async",
        tags=["pytest"],
        fail_on_exception=True,
    )


@pytest.fixture
def neptune_logger_offline():
    return NeptuneLoggerCkpt(
        project="cellarium/tissue-mosaic",
        run=None,
        log_model_checkpoints=True,
        mode="offline",
        tags=["pytest"],
        fail_on_exception=True,
    )


@pytest.fixture
def trainer_with_logger(neptune_logger_async):
    return Trainer(
        weights_save_path="saved_ckpt",
        num_nodes=1,
        gpus=0,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        limit_predict_batches=2,
        logger=neptune_logger_async,
        check_val_every_n_epoch=1,
        accelerator='cpu',  # DDPPlugin(find_unused_parameters=True),
        num_sanity_val_steps=0,
        max_epochs=2,
        log_every_n_steps=1,
    )


@pytest.fixture
def trainer_no_logger(neptune_logger_offline):
    return Trainer(
        weights_save_path="saved_ckpt",
        num_nodes=1,
        gpus=0,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        limit_predict_batches=2,
        logger=neptune_logger_offline,
        check_val_every_n_epoch=1,
        accelerator='cpu',  # DDPPlugin(find_unused_parameters=True),
        num_sanity_val_steps=0,
        max_epochs=2,
        log_every_n_steps=1,
    )


@pytest.fixture
def spot_dict():
    n_beads = 5
    cell_list = ["ES", "Endothelial", "Leydig", "Macrophage", "Myoid", "RS", "SPC", "SPG", "Sertoli"]
    tmp_dict = {
        "x_raw": 200.0 + 100.0 * numpy.random.rand(n_beads),
        "y_raw": 200.0 + 100.0 * numpy.random.rand(n_beads),
        "cell_type": [cell_list[i] for i in numpy.random.randint(low=0, high=len(cell_list), size=n_beads)],
        "barcodes": [_random_string_generator(10, allowed_chars=list("ACTG")) for i in range(n_beads)]
    }
    return tmp_dict


@pytest.fixture
def ann_data(spot_dict):
    n_genes = 10
    metadata_df = pandas.DataFrame(data=spot_dict).set_index("barcodes")
    n_barcodes = metadata_df.shape[0]
    gene_names = [_random_string_generator(6) for i in range(n_genes)]
    gene_names_df = pandas.DataFrame(data={"genes": gene_names, "gene_ids": numpy.arange(n_genes)}).set_index("genes")
    rvs = stats.poisson(5, loc=3).rvs
    gene_counts = random_sparse(n_barcodes, n_genes, density=0.3, format='coo', dtype=numpy.int, data_rvs=rvs)
    adata = AnnData(
        gene_counts,
        obs=metadata_df,
        var=gene_names_df,
        dtype='float32'
    )
    return adata


@pytest.fixture
def sparse_image(spot_dict):
    unique_key = set(spot_dict["cell_type"])
    categories_to_codes = dict(zip(unique_key, range(len(unique_key))))
    sparse_image = SparseImage(spot_dict,
                               x_key='x_raw',
                               y_key='y_raw',
                               category_key='cell_type',
                               categories_to_codes=categories_to_codes,
                               pixel_size=1.0)
    return sparse_image

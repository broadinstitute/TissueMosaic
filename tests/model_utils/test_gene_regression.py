import pytest
import pyro.poutine as poutine
from tissuemosaic.model_utils.gene_regression import (
    model_poisson_log_normal,
    guide_poisson_log_normal,
    train_helper
)


def test_pyro_model(gene_regression_dataset):
    trace = poutine.trace(model_poisson_log_normal).get_trace(gene_regression_dataset)
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())


def test_pyro_guide(gene_regression_dataset):
    trace = poutine.trace(guide_poisson_log_normal).get_trace(gene_regression_dataset)
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())


@pytest.mark.parametrize("regularize_alpha", [True, False])
@pytest.mark.parametrize("subsample_size_cells", [None, 2])
@pytest.mark.parametrize("subsample_size_genes", [None, 3])
@pytest.mark.parametrize("use_covariates", [True, False])
def test_pyro_model_guide(
        gene_regression_dataset,
        regularize_alpha,
        subsample_size_genes,
        subsample_size_cells,
        use_covariates,
        capsys):

    model_kargs = {
        "dataset": gene_regression_dataset,
        "regularize_alpha": regularize_alpha,
        "subsample_size_cells": subsample_size_cells,
        "subsample_size_genes": subsample_size_genes,
        "use_covariates": use_covariates,
        "observed": True
    }

    # construction Vae with params compatible with dummy_dm
    train_helper(
        model_poisson_log_normal,
        guide_poisson_log_normal,
        model_kargs=model_kargs,
        n_steps=2,
        clear_param_store=True)

    # with capsys.disabled():
    #     print(vae)

import pytest

import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from tissuemosaic.genex.pyro_model import *
from tissuemosaic.genex.gene_utils import *

import anndata
from scipy import sparse

import numpy
import numpy as np
import torch
import seaborn
import tarfile
import os
import matplotlib
import matplotlib.pyplot as plt
from anndata import read_h5ad
import scanpy as sc
import pandas as pd



## Generate simulated counts from generative distribution of pyro model 

def generate_counts(k, n, g, l):
    ### draw beta coefficients for 10 cell types and 100 genes from normal(0,1) distribution
    ### draw epsilon coefficients for 100 genes from normal(0,0.1)
    ### draw ncv's for 1000 spots/cells from dirichlet distribution
    ### draw mixture coefficients for 1000 spots/cells from dirichlet distribution
    ### nUmis from log-normal centered around 300
    ### lambdas from laplace distribution (100 genes x 10 ncv)

    beta0_k1g = dist.Normal(-5, 1).sample((k,g))[:, None, :]
    eps_1g = torch.abs(dist.Normal(0,0.1).sample((1,g)))
    prob = np.array([0.8])
    ncv_n1l = dist.Dirichlet(torch.tensor(np.repeat(prob,[k]))).sample((n,1))
    ncv_nl = ncv_n1l.squeeze()

    #mixtures = dist.Dirichlet(torch.tensor(np.repeat(prob,[10]))).sample((1000,1))
    lambda_klg = dist.Normal(0, 2.0).sample((k,l,g))
    lambda_sparsity = 0.95
    lambda_klg.view(-1)[dist.Bernoulli(probs=lambda_sparsity).sample((k,l,g)).bool().view(-1)] = 0.

    total_umi_n1 = (300 * dist.LogNormal(0,0.1).sample((n,1))).int().float()

    ### get 1000 random cell types
    cell_ids_n = (torch.rand((n))*k).int().long() ##randint instead

    beta0_n1g = beta0_k1g[cell_ids_n]
    lambda_nlg = lambda_klg[cell_ids_n]
    total_umi_n11 = total_umi_n1[..., None]



    covariate_nl1 = ncv_nl.unsqueeze(dim=-1)

    log_mu_n1g = beta0_n1g + \
                     torch.sum(covariate_nl1 * lambda_nlg, dim=-2, keepdim=True)

    ### simulate gene counts for these 1000 spots and 100 genes

    eps_n1g = eps_1g.expand(n,1,g)
    sim_counts_n1g = LogNormalPoisson(n_trials=total_umi_n11,
                                 log_rate=log_mu_n1g,
                                 noise_scale=eps_n1g,
                                 num_quad_points=8).sample()

    sim_counts_baseline_n1g = LogNormalPoisson(n_trials=total_umi_n11,
                                 log_rate=beta0_n1g,
                                 noise_scale=eps_n1g,
                                 num_quad_points=8).sample()
    
    return sim_counts_n1g, sim_counts_baseline_n1g, eps_1g, beta0_k1g, lambda_klg, ncv_nl, cell_ids_n


                              
# pyro.sample("counts",
#             LogNormalPoisson(n_trials=total_umi_n11,
#                              log_rate=log_mu_n1g,
#                              noise_scale=eps_n1g,
#                              num_quad_points=8),
#             obs=counts_ng[ind_n.cpu(), None].index_select(dim=-1, index=ind_g.cpu()).to(device))


### run through tissuemosaic pipeline and see if the correct coefficients are learned back


def fit_pyro_model(sim_counts_n1g, ncv_nl, cell_ids_n):
    ## Create GeneDataset

    counts = np.float32(sim_counts_n1g.squeeze())
    sim_anndata = anndata.AnnData(X=sparse.csr_matrix(counts))
    sim_anndata.obsm['ncv'] = ncv_nl
    sim_anndata.obs['cell_type'] = cell_ids_n

    ## Run GLM

    gene_dataset = make_gene_dataset_from_anndata(
            anndata=sim_anndata,
            cell_type_key='cell_type',
            covariate_key='ncv',
            preprocess_strategy='raw',
            apply_pca=False)

    train_dataset, test_dataset, val_dataset = next(iter(train_test_val_split(gene_dataset, random_state=0, train_size=0.8,test_size=0.19,val_size=0.01)))

    torch.cuda.empty_cache()
    gr = GeneRegression()
    gr.configure_optimizer(optimizer_type='adam', lr=5e-4)
    gr.train(
        dataset=train_dataset,
        n_steps=10000,
        print_frequency=500,
        use_covariates=True,
        l1_regularization_strength=10.,
        l2_regularization_strength=None,
        eps_range=(1.0E-5, 1.0E-4),  # <-- note the upper bound was decreased
        subsample_size_cells=None,
        subsample_size_genes=None,
        initialization_type="scratch")

    # fig, axes = plt.subplots(ncols=2, figsize=(8,4))
    # gr.show_loss(ax=axes[0])
    # gr.show_loss(ax=axes[1], logy=True, logx=True)

    ## Get fit parameters
    pred_eps_k1g = pyro.get_param_store().get_param("eps").float().cpu()
    pred_beta0_k1g = pyro.get_param_store().get_param("beta0").float().cpu()
    pred_beta_klg = pyro.get_param_store().get_param("beta").float().cpu()
    
    return pred_eps_k1g, pred_beta0_k1g, pred_beta_klg


## Look at correlation of predicted parameteres to true parameters


@pytest.mark.parametrize("k", [10])
@pytest.mark.parametrize("n", [10000])
@pytest.mark.parametrize("g", [100])
@pytest.mark.parametrize("l", [10])
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_pyro_model(k, n, g, l):

    sim_counts_n1g, sim_counts_baseline_n1g, eps_1g, beta0_k1g, lambda_klg, ncv_nl, cell_ids_n = generate_counts(k, n, g, l)
    pred_eps_k1g, pred_beta0_k1g, pred_beta_klg = fit_pyro_model(sim_counts_n1g, ncv_nl, cell_ids_n)
    
    ## eps correlation
    print("eps correlation:")
    eps_k1g = eps_1g.expand(k,1,g)
    print(np.corrcoef(eps_k1g.detach().numpy().flatten(), pred_eps_k1g.detach().numpy().flatten())[0,1])

    ## beta 0 correlation
    print("beta0 correlation:")
    print(np.corrcoef(beta0_k1g.detach().numpy().flatten(), pred_beta0_k1g.detach().numpy().flatten())[0,1])

    ## beta correlation
    print("beta correlation:")
    print(np.corrcoef(lambda_klg.detach().numpy().flatten(), pred_beta_klg.detach().numpy().flatten())[0,1])
    
    ## perfect correlation is 1
    ## corr of 0.9 is good
    
    np.testing.assert_allclose(np.corrcoef(eps_k1g.detach().numpy().flatten(), pred_eps_k1g.detach().numpy().flatten())[0,1], 1.0, rtol=0.1)
    np.testing.assert_allclose(np.corrcoef(beta0_k1g.detach().numpy().flatten(), pred_beta0_k1g.detach().numpy().flatten())[0,1], 1.0, rtol=0.1)
    np.testing.assert_allclose(np.corrcoef(lambda_klg.detach().numpy().flatten(), pred_beta_klg.detach().numpy().flatten())[0,1], 1.0, rtol=0.1)
    
    
#!/usr/bin/env python

# This script performs gene expression regression with learned SSL features and compares to NCV 

import argparse
import torch
import sys
from typing import List
from anndata import read_h5ad
from tissuemosaic.data import AnndataFolderDM
from tissuemosaic.models.ssl_models import *
import anndata

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
# import gseapy as gp
import pickle

import pdb

import time

# tissuemosaic import
import tissuemosaic as tp

from tissuemosaic.genex.gene_utils import *
from tissuemosaic.utils.anndata_util import *
from tissuemosaic.genex.poisson_glm import *

from multiprocessing import Pool

## stratified by majority cell type label
def regress(train_dataset, val_dataset, test_dataset, config_dict_, ctype, fold_prefix):
    

    gr_baseline = GeneRegression(use_covariates=False,scale_covariates=config_dict_['scale_covariates'], umi_scaling=config_dict_['umi_scaling'], cell_type_prop_scaling=config_dict_['cell_type_prop_scaling'])
    

    ## do alpha regularization sweep on every gene or random subset of genes and then apply to all 
    ## to speed up training?

    ## alpha = 0 is unpenalized GLM
    ## In this case, the design matrix X must have full column rank (no collinearities).
    ## but our X (cell type proportions) has collinearity so we set alpha = 1.0 as default for baseline model

    ## TODO: set max_iter as user parameter
    ## TODO: provide metric to confirm convergence with default max_iter?
    print("Training baseline model")
    start_time = time.time()
    gr_baseline.train(
        train_dataset=train_dataset,
        use_covariates=False,
        regularization_sweep=False,
        alpha_regularization_strengths = np.array([1.0]))
    end_time = time.time()
    
    print(str(end_time - start_time) + " seconds to train baseline model")
    
    gr = GeneRegression(use_covariates=True, scale_covariates=config_dict_['scale_covariates'], umi_scaling=config_dict_['umi_scaling'], cell_type_prop_scaling=config_dict_['cell_type_prop_scaling'])
    

    ## TODO: allow multiple covariates in GeneDataset / GeneRegression
    ## TODO: regularization sweep as user passed parameters
    print("Training covariate model")
    start_time = time.time()
    if config_dict_['regularization_sweep']:
        print("Running regularization sweep")
        gr.train(
            train_dataset=train_dataset,
            regularization_sweep=True,
            val_dataset = val_dataset,
            alpha_regularization_strengths = np.array([0.001, 0.005, 0.01, 0.05, 0.1]))
    else:
        gr.train(
            train_dataset=train_dataset,
            regularization_sweep=False,
            val_dataset = val_dataset,
            alpha_regularization_strengths = np.array([config_dict_['alpha_regularization_strength']]))
    end_time = time.time()
    print(str(end_time - start_time) + " seconds to train covariate model")
    if config_dict_['save_alpha_dict']:
        
        alpha_dict_outfile_name = config_dict_["out_prefix"] + "_" + fold_prefix + "_alpha_dict.pickle"
        alpha_dict_outfile = os.path.join(config_dict_["out_dir"], alpha_dict_outfile_name) 
        with open(alpha_dict_outfile, 'wb') as file:
            pickle.dump(gr.get_alpha_dict(), file)
    
    ## stratify d_sq by cell type
    ## save d_sq_g and q_z_kg as dataframe with gene names/cell type names
    
    pred_counts_ng, counts_ng = gr.predict(test_dataset, return_true_counts=True)
    pred_counts_ng_baseline, counts_ng_baseline = gr_baseline.predict(test_dataset, return_true_counts=True)
    
    ## save counts to file
    pred_counts_ng_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_" + fold_prefix + "_pred_counts_ng.pickle"
    pred_counts_ng_outfile = os.path.join(config_dict_["out_dir"], pred_counts_ng_outfile_name)
    
    with open(pred_counts_ng_outfile, 'wb') as file:
        pickle.dump(pred_counts_ng, file)
        
    counts_ng_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_" + fold_prefix + "_counts_ng.pickle"
    counts_ng_outfile = os.path.join(config_dict_["out_dir"], counts_ng_outfile_name)
    
    with open(counts_ng_outfile, 'wb') as file:
        pickle.dump(counts_ng, file)
        
    pred_counts_ng_baseline_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_" + fold_prefix + "_pred_counts_ng_baseline.pickle"
    pred_counts_ng_baseline_outfile = os.path.join(config_dict_["out_dir"], pred_counts_ng_baseline_outfile_name)
    
    with open(pred_counts_ng_baseline_outfile, 'wb') as file:
        pickle.dump(pred_counts_ng_baseline, file)
    
     ## save gr object to file
    gr_covar_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_" + fold_prefix + "_gr_covar.pickle"
    gr_covar_outfile = os.path.join(config_dict_["out_dir"], gr_covar_outfile_name)
    
    with open(gr_covar_outfile, 'wb') as file:
        pickle.dump(gr, file)
        
    ## save cell type ids
    cell_type_ids_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_" + fold_prefix + "_cell_type_ids.pickle"
    cell_type_ids_outfile = os.path.join(config_dict_["out_dir"], cell_type_ids_outfile_name)
    
    with open(cell_type_ids_outfile, 'wb') as file:
        pickle.dump(test_dataset.cell_type_ids, file)
        
    ## save gene names
    gene_names_outfile_name = config_dict_["out_prefix"]  + "_" + ctype + "_" + fold_prefix + "_gene_names.pickle"
    gene_names_outfile = os.path.join(config_dict_["out_dir"], gene_names_outfile_name)
    
    with open(gene_names_outfile, 'wb') as file:
        pickle.dump(test_dataset.gene_names, file)
    
    return pred_counts_ng, pred_counts_ng_baseline, counts_ng

def run_regression(filtered_anndata, ctype, kfold):
    
    print(f"Running train/test fold {kfold}")
    
    train_anndata = filtered_anndata[filtered_anndata.obs[f'train_test_fold_{kfold}'] == 0]
    test_anndata = filtered_anndata[filtered_anndata.obs[f'train_test_fold_{kfold}'] == 1]


    train_gene_dataset = make_gene_dataset_from_anndata(
        anndata=train_anndata,
        cell_type_key=config_dict_["cell_type_key"],
        covariate_key=config_dict_["feature_key"],
        preprocess_strategy='raw',
        cell_type_prop_key=config_dict_["cell_type_proportions_key"],
        apply_pca=False)

    test_gene_dataset = make_gene_dataset_from_anndata(
        anndata=test_anndata,
        cell_type_key=config_dict_["cell_type_key"],
        covariate_key=config_dict_["feature_key"],
        preprocess_strategy='raw',
        cell_type_prop_key=config_dict_["cell_type_proportions_key"],
        apply_pca=False)

    test_fold_pred_counts_ng,test_fold_pred_counts_ng_baseline, test_fold_counts_ng = regress(train_gene_dataset, None, test_gene_dataset, config_dict_, ctype, str(kfold))
    
    return test_fold_pred_counts_ng, test_fold_pred_counts_ng_baseline, test_fold_counts_ng, test_gene_dataset.cell_type_ids


def parse_args(argv: List[str]) -> dict:
    """
    Read argv from command-line and produce a configuration dictionary.
    If the command-line arguments include

    If the command-line arguments include '--to_yaml my_yaml_file.yaml' the configuration dictionary is written to file.

    Args:
        argv: the parameter passed from the command line.

    Note:
        If argv includes '--config input.yaml' the parameters are read from file.
        The config.yaml parameters have priority over the CLI parameters.

    Note:
        If argv includes '--to_yaml output.yaml' the configuration dictionary is written to file.

    Note:
        Parameters which are missing from both argv and config.yaml will be set to their default values.

    Returns:
        config_dict: a dictionary with all the configuration parameters.
    """
    parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')

    parser.add_argument("--anndata_in", type=str, required=True,
                        help="path to the directory containing the annotated anndatas OR single input anndata.h5ad")
    
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory name to save images/plots to.")
    
    parser.add_argument("--out_prefix", type=str, required=True,
                        help="Output prefix to name output files.")
    
    parser.add_argument("--feature_key", type=str, required=True,
                        help="The computed features to regress on.")
    
    parser.add_argument("--regularization_sweep", type=bool, required=False,
                        help="Whether to run regularization sweep", default=False)
        
    parser.add_argument("--alpha_regularization_strength", type=float, required=False,
                        help="Regularization for covariate gene regression", default=0.0)
    
    parser.add_argument("--save_alpha_dict", type=bool, required=False,
                        help="Whether to save per-gene alpha regularization dictionary", default=False)
    
    parser.add_argument("--scale_covariates", type=bool, required=False,
                        help="Whether to standardize covariates before regression", default=False)
        
    parser.add_argument("--umi_scaling", type=int, required=False,
                        help="Scaling factor for log umi coefficient", default=10e3)
    
    parser.add_argument("--cell_type_prop_scaling", type=int, required=False,
                        help="Scaling factor for cell type proportions coefficients", default=10e1)
    
    parser.add_argument("--category_key", type=str, required=False,
                        help="Key in obsm containing categories", default="rctd_doublet_weights")
    
    parser.add_argument("--cell_type_key", type=str, required=False,
                        help="Key in obs containing majority cell types per spot", default="cell_type")
    
    parser.add_argument("--cell_type_proportions_key", type=str, required=False,
                        help="Key in obsm deconvolution of cell types per spot", default="cell_type_proportions")
    
    parser.add_argument("--fg_bc_high_var", type=int, required=False,
                        help="Filtering criteria", default=None)
    
    parser.add_argument("--fc_bc_min_umi", type=int, required=False,
                        help="Filtering criteria", default=500)
    
    parser.add_argument("--fg_bc_min_pct_cells_by_counts", type=int, required=False,
                        help="Filtering criteria", default=10)
    
    parser.add_argument("--cell_types", type=str, nargs='*', required=False,
                        help="Cell types to run regression on; defaults to all cell types")
    
    parser.add_argument("--filter_feature", type=float, required=False,
                        help="If provided, set outlier values in feature_key beyond filter threshold to 0")
    
    parser.add_argument("--OMP_NUM_THREADS", type=str, required=False, default="1",
                    help="Set number of OMP threads for Poisson regression")
    
    parser.add_argument("--MKL_NUM_THREADS", type=str, required=False, default="1",
                help="Set number of MKL threads for Poisson regression")
    
    ## TODO: add the rest of the filtering criteria as user parameters
    ## TODO: save arguments / config file in out directory 
    
    # Add help at the very end
    parser = argparse.ArgumentParser(parents=[parser], add_help=True)

    # Process everything and check
    args = parser.parse_args(argv)
    
    return vars(args)


if __name__ == '__main__':
    config_dict_ = parse_args(sys.argv[1:])

    if config_dict_["anndata_in"].endswith('.h5ad'):
        merged_anndata = read_h5ad(filename=config_dict_["anndata_in"])
    else:
        annotated_anndata_folder = config_dict_["anndata_in"]
        
        fname_list = []
        for f in os.listdir(annotated_anndata_folder):
            if f.endswith('.h5ad'):
                fname_list.append(f)
        
        ## set num threads ; need to set in environment before running script
        # os.environ["OMP_NUM_THREADS"] = config_dict_["OMP_NUM_THREADS"]
        # os.environ["MKL_NUM_THREADS"] = config_dict_["MKL_NUM_THREADS"]
        
        ## read in all anndatas and create one big anndata out of them
        adata_list = []

        for i, fname in enumerate(fname_list):
            adata = read_h5ad(filename=os.path.join(annotated_anndata_folder, fname))
            
            adata.obs['sample_id'] = i * np.ones(adata.X.shape[0])
            adata_list.append(adata)
            
        merged_anndata = merge_anndatas_inner_join(adata_list)
    
    ## add majority cell type labels if not already present
    if config_dict_["cell_type_key"] not in list(merged_anndata.obs.keys()):
        merged_anndata.obs[config_dict_["cell_type_key"]] = pd.DataFrame(merged_anndata.obsm[config_dict_["cell_type_proportions_key"]].idxmax(axis=1))
    
    ## loop regression over all cell types
    if config_dict_["cell_types"] is not None:
        print("cell types to regress:")
        cell_types = config_dict_["cell_types"]
        print((' ').join(cell_types))
    else:
        print("cell types to regress:")
        cell_types = np.unique(merged_anndata.obs[config_dict_["cell_type_key"]])
        print((' ').join(cell_types))
        
    for ctype in cell_types:
        
        print("Running regression on cell-type: " + ctype)
        
        ## assert that cell_types are in anndata.obs
        
        merged_anndata_ctype = merged_anndata[merged_anndata.obs[config_dict_["cell_type_key"]] == ctype]
        

        ## flag in cell type prop key
        filtered_anndata = filter_anndata(merged_anndata_ctype, cell_type_key = config_dict_["cell_type_key"], fg_bc_high_var=config_dict_["fg_bc_high_var"], fc_bc_min_umi=config_dict_["fc_bc_min_umi"], fg_bc_min_pct_cells_by_counts=config_dict_["fg_bc_min_pct_cells_by_counts"])

        # filter spatial covariates 
        if config_dict_["filter_feature"] is not None:
            threshold = config_dict_["filter_feature"]   
          
            filtered_anndata.obsm[config_dict_["feature_key"]][filtered_anndata.obsm[config_dict_["feature_key"]] > threshold] = 0                                                           


        ## Split data into train/test sets based on spatial split assigned in main_2_featurize.py
        ## If running regularization sweep, 'train_test_val_split_id' must be present in obs

        if config_dict_["regularization_sweep"]:
                                                             
            assert "train_test_val_split_id" in filtered_anndata.obs.keys(), "Train_test_val_split_id must be present in obs to run regularization sweep"
                                                               
            train_anndata = filtered_anndata[filtered_anndata.obs['train_test_val_split_id'] == 0]
            val_anndata = filtered_anndata[filtered_anndata.obs['train_test_val_split_id'] == 1]
            test_anndata = filtered_anndata[filtered_anndata.obs['train_test_val_split_id'] == 2]

            train_gene_dataset = make_gene_dataset_from_anndata(
                anndata=train_anndata,
                cell_type_key=config_dict_["cell_type_key"],
                covariate_key=config_dict_["feature_key"],
                preprocess_strategy='raw',
                cell_type_prop_key=config_dict_["cell_type_proportions_key"],
                apply_pca=False)

            test_gene_dataset = make_gene_dataset_from_anndata(
                anndata=test_anndata,
                cell_type_key=config_dict_["cell_type_key"],
                covariate_key=config_dict_["feature_key"],
                preprocess_strategy='raw',
                cell_type_prop_key=config_dict_["cell_type_proportions_key"],
                apply_pca=False)

            val_gene_dataset = make_gene_dataset_from_anndata(
                anndata=val_anndata,
                cell_type_key=config_dict_["cell_type_key"],
                covariate_key=config_dict_["feature_key"],
                preprocess_strategy='raw',
                cell_type_prop_key=config_dict_["cell_type_proportions_key"],
                apply_pca=False)

            pred_counts_ng, pred_counts_ng_baseline, counts_ng = regress(train_gene_dataset, val_gene_dataset, test_gene_dataset, config_dict_, ctype, "")

            cell_type_ids = test_gene_dataset.cell_type_ids
        ## if not regularization sweep do spatial train test kfolds
        else:
            list_of_folds_pred_counts_ng = []
            list_of_folds_pred_counts_ng_baseline = []
            list_of_folds_counts_ng = []
            list_of_folds_cell_type_ids = []

            ## TODO: make num kfold / range user parameter
            ## TODO: add in number of cores as a user parameter / don't parallelize by default

            ## parallelize over kfolds

            with Pool(6) as p:
                kfold_iterable = p.starmap(run_regression, [(filtered_anndata, ctype, 1), (filtered_anndata, ctype, 2), (filtered_anndata, ctype, 3), (filtered_anndata, ctype, 4)])

            for result in kfold_iterable:
                list_of_folds_pred_counts_ng.append(result[0])
                list_of_folds_pred_counts_ng_baseline.append(result[1])
                list_of_folds_counts_ng.append(result[2])
                list_of_folds_cell_type_ids.append(result[3])

            pred_counts_ng = np.concatenate(list_of_folds_pred_counts_ng, axis=0)
            pred_counts_ng_baseline = np.concatenate(list_of_folds_pred_counts_ng_baseline, axis=0)

            assert np.all(pred_counts_ng >= 0), "Some elements in pred_counts_ng are not greater than 0"
            assert np.all(pred_counts_ng_baseline >= 0), "Some elements in pred_counts_ng are not greater than 0"

            counts_ng = np.concatenate(list_of_folds_counts_ng, axis=0)
            cell_type_ids = torch.cat(list_of_folds_cell_type_ids, dim=0)
        
    
        ## compute evaluation metrics:
        df_d_sq_gk_ssl, df_rel_q_gk_ssl = GeneRegression.compute_eval_metrics(pred_counts_ng=pred_counts_ng, 
                                                        counts_ng=counts_ng,
                                                        cell_type_ids = cell_type_ids,
                                                        gene_names = np.array(filtered_anndata.var.index),
                                                        pred_counts_ng_baseline = pred_counts_ng_baseline)


        df_d_sq_gk_baseline, df_rel_q_gk_baseline = GeneRegression.compute_eval_metrics(pred_counts_ng=pred_counts_ng_baseline, 
                                                        counts_ng=counts_ng,
                                                        cell_type_ids = cell_type_ids,
                                                        gene_names = np.array(filtered_anndata.var.index),
                                                        pred_counts_ng_baseline = pred_counts_ng_baseline) 

        #### Write baseline metrics to out files ###

        ## write d_sq_gk to file
        baseline_d_sq_gk_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_df_d_sq_gk" + "_baseline.pickle"
        baseline_d_sq_gk_outfile = os.path.join(config_dict_["out_dir"], baseline_d_sq_gk_outfile_name)

        with open(baseline_d_sq_gk_outfile, 'wb') as file:
            pickle.dump(df_d_sq_gk_baseline, file)

        ## write rel_q_gk to file
        baseline_rel_q_gk_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_df_rel_q_gk" + "_baseline.pickle"
        baseline_rel_q_gk_outfile = os.path.join(config_dict_["out_dir"], baseline_rel_q_gk_outfile_name)

        with open(baseline_rel_q_gk_outfile, 'wb') as file:
            pickle.dump(df_rel_q_gk_baseline, file)

        #### Write spatial metrics to out files####

        ## write rel_q_gk to file
        d_sq_gk_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_df_d_sq_gk" + "_ssl.pickle"
        d_sq_gk_outfile = os.path.join(config_dict_["out_dir"], d_sq_gk_outfile_name)

        with open(d_sq_gk_outfile, 'wb') as file:
            pickle.dump(df_d_sq_gk_ssl, file)

        ## write rel_q_gk to file
        rel_q_gk_outfile_name = config_dict_["out_prefix"] + "_" + ctype + "_df_rel_q_gk" + "_ssl.pickle"
        rel_q_gk_outfile = os.path.join(config_dict_["out_dir"], rel_q_gk_outfile_name)

        with open(rel_q_gk_outfile, 'wb') as file:
            pickle.dump(df_rel_q_gk_ssl, file)

    
    
                        
                        
        
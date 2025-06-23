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

def save_to_outfile(out_dir: str=None, 
                    out_prefix: str=None, 
                    ctype: str=None, 
                    fold_prefix: str=None, 
                    suffix: str=None, 
                    data=None):
    """Saves data to a pickle file with a standardized naming convention.

    Creates a pickle file in the specified output directory with a name constructed from the
    provided prefix, cell type, fold, and suffix parameters. The data is serialized using pickle.

    Args:
        out_dir (str): Directory path where the output file will be saved
        out_prefix (str): Prefix for the output filename
        ctype (str): Cell type identifier to include in filename
        fold_prefix (str): Cross-validation fold identifier for filename
        suffix (str): Suffix to append to filename
        data (Any): Data object to serialize and save to file

    Returns:
        str: Full path to the created output file
    """

    if fold_prefix is not None:
        outfile_name = f"{out_prefix}_{ctype}_{fold_prefix}_{suffix}.pickle"
    else:
        outfile_name = f"{out_prefix}_{ctype}_{suffix}.pickle"
        
    outfile = os.path.join(out_dir, outfile_name)
    with open(outfile, 'wb') as file:
        pickle.dump(data, file)
    return outfile

## stratified by majority cell type label
def regress(train_dataset, val_dataset, test_dataset, config_dict_, ctype, fold_prefix):
    """Performs gene expression regression using baseline and covariate models.

    This function trains two gene regression models:
    1. A baseline model using only cell type proportions
    2. A covariate model using both cell type proportions and additional features

    Args:
        train_dataset: Dataset containing training data
        val_dataset: Dataset containing validation data, used for regularization sweep
        test_dataset: Dataset containing test data
        config_dict_: Dictionary containing configuration parameters including:
            - scale_covariates: Factor to scale covariates by
            - umi_scaling: Factor to scale UMI counts by  
            - cell_type_prop_scaling: Factor to scale cell type proportions by
            - regularization_sweep: Whether to perform regularization parameter sweep
            - alpha_regularization_strength: Regularization strength if not doing sweep
            - save_alpha_dict: Whether to save regularization parameters
            - out_dir: Directory to save output files
            - out_prefix: Prefix for output filenames
        ctype: Cell type being analyzed
        fold_prefix: String identifying the current cross-validation fold

    Returns:
        tuple: Contains:
            - pred_counts_ng: Predicted counts from covariate model
            - pred_counts_ng_baseline: Predicted counts from baseline model  
            - counts_ng: True counts
    """

    gr_baseline = GeneRegression(use_covariates=False,scale_covariates=config_dict_['scale_covariates'], umi_scaling=config_dict_['umi_scaling'], cell_type_prop_scaling=config_dict_['cell_type_prop_scaling'])

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
    ## TODO: allow user to modify regularization strengths
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
        alpha_dict_outfile = save_to_outfile(config_dict_["out_dir"], config_dict_["out_prefix"], ctype, fold_prefix, "alpha_dict", gr.get_alpha_dict())
    
    ## stratify d_sq by cell type
    ## save d_sq_g and q_z_kg as dataframe with gene names/cell type names
    pred_counts_ng, counts_ng = gr.predict(test_dataset, return_true_counts=True)
    pred_counts_ng_baseline, counts_ng_baseline = gr_baseline.predict(test_dataset, return_true_counts=True)
    
    ## Save results to outfiles
    pred_counts_ng_outfile = save_to_outfile(config_dict_["out_dir"], config_dict_["out_prefix"], ctype, fold_prefix, "pred_counts_ng", pred_counts_ng)
    counts_ng_outfile = save_to_outfile(config_dict_["out_dir"], config_dict_["out_prefix"], ctype, fold_prefix, "counts_ng", counts_ng)
    pred_counts_ng_baseline_outfile = save_to_outfile(config_dict_["out_dir"], config_dict_["out_prefix"], ctype, fold_prefix, "pred_counts_ng_baseline", pred_counts_ng_baseline)
    gr_covar_outfile = save_to_outfile(config_dict_["out_dir"], config_dict_["out_prefix"], ctype, fold_prefix, "gr_covar", gr)
    
    cell_type_ids_outfile = save_to_outfile(config_dict_["out_dir"], config_dict_["out_prefix"], ctype, fold_prefix, "cell_type_ids", test_dataset.cell_type_ids)
    gene_names_outfile = save_to_outfile(config_dict_["out_dir"], config_dict_["out_prefix"], ctype, fold_prefix, "gene_names", test_dataset.gene_names)
    
    
    return pred_counts_ng, pred_counts_ng_baseline, counts_ng

def run_regression(filtered_anndata, ctype, kfold):
    """Runs gene expression regression on a train/test fold of the data.

    This function splits the input anndata into train and test sets based on the fold index,
    creates gene datasets from the splits, and runs regression to predict gene expression.
    It performs both a baseline regression using only cell type proportions and a covariate 
    regression that includes additional features.

    Args:
        filtered_anndata (AnnData): Input anndata object containing gene expression, cell types,
            and features. Must have a column in .obs indicating train/test split for the given fold.
        ctype (str): Cell type to analyze
        kfold (int): Index of the cross-validation fold to use for train/test split

    Returns:
        tuple:
            - test_fold_pred_counts_ng (array): Predicted gene counts from covariate model on test set
            - test_fold_pred_counts_ng_baseline (array): Predicted gene counts from baseline model on test set  
            - test_fold_counts_ng (array): True gene counts for test set
            - cell_type_ids (array): Cell type labels for test set observations
    """
    
    print(f"Running train/test fold {kfold}")
    
    ## TODO: double check why train labelled as -1/ check main_2
    train_anndata = filtered_anndata[filtered_anndata.obs[f'train_test_fold_{kfold}'] == -1]
    # train_anndata = filtered_anndata[filtered_anndata.obs[f'train_test_fold_{kfold}'] == 0]
    test_anndata = filtered_anndata[filtered_anndata.obs[f'train_test_fold_{kfold}'] == 1]

    train_gene_dataset = make_gene_dataset_from_anndata(
        anndata=train_anndata,
        cell_type_key=config_dict_["cell_type_key"],
        covariate_key=config_dict_["feature_key"],
        preprocess_strategy='raw',
        cell_type_prop_key=config_dict_["cell_type_proportions_key"],
        apply_pca=config_dict_["apply_pca"],
        n_components=config_dict_["n_components"]) 

    test_gene_dataset = make_gene_dataset_from_anndata(
        anndata=test_anndata,
        cell_type_key=config_dict_["cell_type_key"],
        covariate_key=config_dict_["feature_key"],
        preprocess_strategy='raw',
        cell_type_prop_key=config_dict_["cell_type_proportions_key"],
        apply_pca=config_dict_["apply_pca"],
        n_components=config_dict_["n_components"])

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

    parser.add_argument("--donotfilter_anndata", type=bool, required=False,
                        help="If True, skip all filtering")
    
    parser.add_argument("--fg_bc_high_var", type=int, required=False,
                        help="Filtering criteria", default=None)
    
    parser.add_argument("--fc_bc_min_umi", type=int, required=False,
                        help="Filtering criteria", default=500)
    
    parser.add_argument("--fg_bc_min_pct_cells_by_counts", type=int, required=False,
                        help="Filtering criteria", default=10)

    parser.add_argument("--gene_list", type=str, required=False,
                    help="Path to file containing gene names (supersedes gene filtering criteria)")
    
    parser.add_argument("--cell_types", type=str, nargs='*', required=False,
                        help="Cell types to run regression on; defaults to all cell types")
    
    parser.add_argument("--filter_feature", type=float, required=False,
                        help="If provided, set outlier values in feature_key beyond filter threshold to 0")

    parser.add_argument("--apply_pca", type=bool, required=False,
                        help="If provided, apply PCA to feature_key before regression", default=False)

    parser.add_argument("--n_components", type=bool, required=False,
                        help="If provided, number of components to return from PCA. Integer specifies dimensionality after PCA reduction. \
                        Float results in dimensionality such that explained variance is at least that value", default=10)
    
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
            # print(fname)
            adata = read_h5ad(filename=os.path.join(annotated_anndata_folder, fname))
            
            adata.obs['sample_id'] = i * np.ones(adata.X.shape[0])

            # if config_dict_["feature_key"] in adata.obsm.keys():
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
    
        if config_dict_["donotfilter_anndata"]:
            filtered_anndata = merged_anndata_ctype
        else:
            filtered_anndata = filter_anndata(merged_anndata_ctype, cell_type_key = config_dict_["cell_type_key"], fg_bc_high_var=config_dict_["fg_bc_high_var"], fc_bc_min_umi=config_dict_["fc_bc_min_umi"], fg_bc_min_pct_cells_by_counts=config_dict_["fg_bc_min_pct_cells_by_counts"])

        if config_dict_["gene_list"] is not None:
            with open(config_dict_["gene_list"], 'r') as file:
                gene_list = file.read().splitlines()

            try:
                filtered_anndata = filtered_anndata[:, gene_list]
            except KeyError:
                print(f"Gene list contains genes not present in anndata after filtering, please try a different filtering criteria")
                continue

        # filter spatial covariates 
        if config_dict_["filter_feature"] is not None:
            threshold = config_dict_["filter_feature"]   
          
            filtered_anndata.obsm[config_dict_["feature_key"]][filtered_anndata.obsm[config_dict_["feature_key"]] > threshold] = 0
        
        ## Split data into train/test sets based on spatial split assigned in main_2_featurize.py
        ## If running regularization sweep, 'train_test_val_split_id' must be present in obs

        print("filtered anndata:")
        print(filtered_anndata)

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
                apply_pca=config_dict_["apply_pca"],
                n_components=config_dict_["n_components"])

            test_gene_dataset = make_gene_dataset_from_anndata(
                anndata=test_anndata,
                cell_type_key=config_dict_["cell_type_key"],
                covariate_key=config_dict_["feature_key"],
                preprocess_strategy='raw',
                cell_type_prop_key=config_dict_["cell_type_proportions_key"],
                apply_pca=config_dict_["apply_pca"],
                n_components=config_dict_["n_components"])

            val_gene_dataset = make_gene_dataset_from_anndata(
                anndata=val_anndata,
                cell_type_key=config_dict_["cell_type_key"],
                covariate_key=config_dict_["feature_key"],
                preprocess_strategy='raw',
                cell_type_prop_key=config_dict_["cell_type_proportions_key"],
                apply_pca=config_dict_["apply_pca"],
                n_components=config_dict_["n_components"])

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

            print(filtered_anndata)
            
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

        ## write baseline_d_sq_gk to file
        baseline_d_sq_gk_outfile = save_to_outfile(out_dir=config_dict_["out_dir"], out_prefix=config_dict_["out_prefix"], ctype=ctype, suffix="df_d_sq_gk_baseline", data=df_d_sq_gk_baseline)

        ## write baseline_rel_q_gk to file
        baseline_rel_q_gk_outfile = save_to_outfile(out_dir=config_dict_["out_dir"], out_prefix=config_dict_["out_prefix"], ctype=ctype, suffix="df_rel_q_gk_baseline", data=df_rel_q_gk_baseline)

        #### Write spatial metrics to out files####

        ## write d_sq_gk to file
        d_sq_gk_outfile = save_to_outfile(out_dir=config_dict_["out_dir"], out_prefix=config_dict_["out_prefix"], ctype=ctype, suffix="df_d_sq_gk_ssl", data=df_d_sq_gk_ssl)

        ## write rel_q_gk to file
        rel_q_gk_outfile = save_to_outfile(out_dir=config_dict_["out_dir"], out_prefix=config_dict_["out_prefix"], ctype=ctype, suffix="df_rel_q_gk_ssl", data=df_rel_q_gk_ssl)
    
                        
                        
        
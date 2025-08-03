

from typing import Union, Tuple, List
import anndata
# import anndata as ad
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV, RidgeCV

import scanpy as sc
import os
import sys
import time
from anndata import read_h5ad
import pickle

import anndata as ad
import subprocess
import pickle

from tissuemosaic.utils import *
from tissuemosaic.plots import *

from tissuemosaic.data import SparseImage
from tissuemosaic.utils.anndata_util import *

import time

def delta_tmi_permutation_test(anndata1, 
                               anndata2, 
                               out_dir, 
                               out_file, 
                               ctype, 
                               ctype_key, 
                               ctype_proportions_key, 
                               feature_key, 
                               alpha_regularization, 
                               n_perms=100, 
                               subsample=False, 
                               subsample_proportion=1.0, 
                               gene_list_file=None, 
                               suppress_output = True):
    """Performs permutation testing to compare differential tissue motif information between two conditions.

    This function runs a permutation test to test for differential tissue motif information between two conditions,
    represented by two AnnData objects. For each permutation, it shuffles the data between conditions and 
    calculates a differential tissue motif information score to construct the null distribution.

    Args:
        anndata1 (AnnData): First condition AnnData object
        anndata2 (AnnData): Second condition AnnData object  
        out_dir (str): Directory path to save output files
        out_file (str): Name of output file to save permutation results
        ctype (str): Cell type to analyze
        ctype_key (str): Key in AnnData obs containing cell type labels
        ctype_proportions_key (str): Key in AnnData obs containing cell type proportions
        feature_key (str): Key in AnnData obs containing feature data
        alpha_regularization (float): Regularization strength parameter
        n_perms (int, optional): Number of permutations to run. Defaults to 100.
        subsample (bool, optional): Whether to subsample the data. Defaults to False.
        subsample_proportion (float, optional): Proportion to subsample if subsample=True. Defaults to 1.0.
        gene_list_file (str, optional): Path to file containing genes to analyze. Defaults to None.
        suppress_output (bool, optional): Whether to suppress print statements. Defaults to True.

    Returns:
        list: List containing results from each permutation indicating whether condition 2 had higher
             gene expression statistics compared to condition 1.
    """

    higher_si_in_cond_2_across_perms = []
    ## get start time
    start_time = time.time()
    for i in range(n_perms):

        higher_si_in_cond_2 = run_permutation(anndata1, anndata2, os.path.join(out_dir, './temp1.h5ad'), os.path.join(out_dir, './temp2.h5ad'), out_dir, ctype, ctype_proportions_key, ctype_key, feature_key, alpha_regularization, subsample=subsample, subsample_proportion=subsample_proportion, gene_list_file=gene_list_file, suppress_output = suppress_output)
        higher_si_in_cond_2_across_perms.append(higher_si_in_cond_2)

        if i % 10 == 0:
            print(f"Finished permutation {i + 1}")
            curr_time = time.time()
            print(f"Time elapsed: {curr_time - start_time}")
            pickle.dump(higher_si_in_cond_2_across_perms, open(os.path.join(out_dir, out_file), 'wb'))

    return higher_si_in_cond_2_across_perms

## shuffle rows between anndata 1 and anndata 2
def shuffle_anndata(anndata1, anndata2):
    
    # Step 1: Concatenate along the rows
    combined_adata = ad.concat([anndata1, anndata2], axis=0, join='inner')

    # Step 2: Randomly shuffle the rows (observations)
    n_total_cells = combined_adata.n_obs
    permuted_indices = np.random.permutation(n_total_cells)

    # Apply the permutation to the `AnnData` object
    shuffled_adata = combined_adata[permuted_indices].copy()

    # Step 3: Split the shuffled `AnnData` back into two
    n_cells_adata1 = anndata1.n_obs
    shuffled_adata1 = shuffled_adata[:n_cells_adata1]
    shuffled_adata2 = shuffled_adata[n_cells_adata1:]

    return shuffled_adata1, shuffled_adata2

# return pickled file
def read_pickled_file(out_dir, filename):
    with open(os.path.join(out_dir, filename), 'rb') as f:
        data = pickle.load(f)
    return data


def run_true_stat(anndata_1, anndata_2, out_dir, ctype, ctype_proportions_key, ctype_key, feature_key, alpha_regularization, out_prefix, gene_list_file=None):
    """Runs gene regression on two AnnData objects and compares true delta TMI statistic

    This function runs gene regression separately on two AnnData objects using the same parameters,
    then compares the independent TMI statistics to compute delta TMI scores for each gene.

    Args:
        anndata_1 (str): Path to first AnnData h5ad file (condition 1)
        anndata_2 (str): Path to second AnnData h5ad file (condition 2) 
        out_dir (str): Directory to save output files
        ctype (str): Cell type to analyze
        ctype_proportions_key (str): Key in AnnData.obsm containing cell type proportions
        ctype_key (str): Key in AnnData.obs containing cell type labels
        feature_key (str): Key for features to use in regression
        alpha_regularization (float): Regularization strength for regression
        out_prefix (str): Prefix for output filenames
        gene_list_file (str, optional): Path to file containing gene names to analyze. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the true delta TMI scores for each gene.
    """

    # Build the command as a list
    cond_1_command = f"python /home/skambha6/chenlab/tissue_purifier/TissueMosaic/run/main_3_gene_regression.py --anndata_in {anndata_1} --out_dir {out_dir} --out_prefix {out_prefix}_1 --feature_key {feature_key} --alpha_regularization_strength {alpha_regularization} --donotfilter_anndata True --cell_types {ctype} --cell_type_key {ctype_key} --cell_type_proportions_key {ctype_proportions_key}"
    
    if gene_list_file is not None:
        cond_1_command += f" --gene_list {gene_list_file}"
    
    cond_1_command = cond_1_command.split(" ")

    # Run the command
    cond_1_result = subprocess.run(cond_1_command, capture_output=True, text=True)


    # Build the command as a list
    cond_2_command = f"python /home/skambha6/chenlab/tissue_purifier/TissueMosaic/run/main_3_gene_regression.py --anndata_in {anndata_2} --out_dir {out_dir} --out_prefix {out_prefix}_2 --feature_key {feature_key} --alpha_regularization_strength {alpha_regularization} --donotfilter_anndata True --cell_types {ctype} --cell_type_key {ctype_key} --cell_type_proportions_key {ctype_proportions_key}"
    cond_2_command = cond_2_command.split(" ")

    # Run the command
    cond_2_result = subprocess.run(cond_2_command, capture_output=True, text=True)

    ## look at delta TMI genes b/w enriched motifs
    cond_1_rel_q_gk_outfile_name = f"{out_prefix}_1" + '_' + ctype + f"_df_rel_q_gk_ssl.pickle"
    cond_1_rel_q_gk = read_pickled_file(out_dir, cond_1_rel_q_gk_outfile_name)
    cond_1_rel_q_gk = -1 * cond_1_rel_q_gk

    cond_2_rel_q_gk_outfile_name = f"{out_prefix}_2" + '_' + ctype + f"_df_rel_q_gk_ssl.pickle"
    cond_2_rel_q_gk = read_pickled_file(out_dir, cond_2_rel_q_gk_outfile_name)
    cond_2_rel_q_gk = -1 * cond_2_rel_q_gk

    higher_si_in_cond_2 = cond_2_rel_q_gk.sub(cond_1_rel_q_gk, fill_value=0).dropna()

    return higher_si_in_cond_2 


def run_permutation(anndata1, anndata2, temp_file_1, temp_file_2, out_dir, ctype, ctype_proportions_key, ctype_key, feature_key, alpha_regularization, subsample=False, subsample_proportion=1.0, gene_list_file=None, suppress_output = True):
    """Runs a single permutation to compute null delta TMI distribution

    This function performs a permutation test by:
    1. Shuffling cells between two AnnData objects
    2. Optionally subsampling the shuffled data
    3. Running gene regression on both shuffled datasets
    4. Computing and returning the null delta TMI scores

    Args:
        anndata1 (AnnData): First AnnData object to shuffle and analyze
        anndata2 (AnnData): Second AnnData object to shuffle and analyze
        temp_file_1 (str): Path to save temporary shuffled version of anndata1
        temp_file_2 (str): Path to save temporary shuffled version of anndata2
        out_dir (str): Directory to save output files
        ctype (str): Cell type to analyze
        ctype_proportions_key (str): Key in AnnData.obsm containing cell type proportions
        ctype_key (str): Key in AnnData.obs containing cell type labels
        feature_key (str): Key for features to use in regression
        alpha_regularization (float): Regularization strength for regression
        subsample (bool, optional): Whether to subsample the shuffled data. Defaults to False.
        subsample_proportion (float, optional): Proportion of data to sample if subsampling. Defaults to 1.0.
        gene_list_file (str, optional): Path to file containing gene names to analyze. Defaults to None.
        suppress_output (bool, optional): Whether to suppress stdout/stderr from regression. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame containing the delta TMI scores for each gene in this permutation.
    """

    shuffled_adata1, shuffled_adata2 = shuffle_anndata(anndata1, anndata2)

    if subsample:

        # Generate random indices
        random_indices1 = np.random.choice(shuffled_adata1.n_obs, size=int(shuffled_adata1.n_obs * subsample_proportion), replace=False)
        shuffled_adata1 = shuffled_adata1[random_indices1, :].copy()

        random_indices2 = np.random.choice(shuffled_adata2.n_obs, size=int(shuffled_adata2.n_obs * subsample_proportion), replace=False)
        shuffled_adata2 = shuffled_adata2[random_indices2, :].copy()


    shuffled_adata1.write_h5ad(temp_file_1)
    shuffled_adata2.write_h5ad(temp_file_2)

    
    # Build the command as a list
    cond_1_command = f"python ../../../TissueMosaic/run/main_3_gene_regression.py --anndata_in {temp_file_1} --out_dir {out_dir} --out_prefix dino_cond_1_null --feature_key {feature_key} --alpha_regularization_strength {alpha_regularization} --donotfilter_anndata True --cell_types {ctype} --cell_type_key {ctype_key} --cell_type_proportions_key {ctype_proportions_key}"
    
    if gene_list_file is not None:
        cond_1_command += f" --gene_list {gene_list_file}"

    cond_1_command = cond_1_command.split(" ")

    # Run the command
    if suppress_output:
        cond_1_result = subprocess.run(cond_1_command, capture_output=True, text=True)
    else:
        cond_1_result = subprocess.run(cond_1_command, capture_output=False, text=True)


    # Build the command as a list
    cond_2_command = f"python ../../../TissueMosaic/run/main_3_gene_regression.py --anndata_in {temp_file_2} --out_dir {out_dir} --out_prefix dino_cond_2_null --feature_key {feature_key} --alpha_regularization_strength {alpha_regularization} --donotfilter_anndata True --cell_types {ctype} --cell_type_key {ctype_key} --cell_type_proportions_key {ctype_proportions_key}"
    
    if gene_list_file is not None:
        cond_2_command += f" --gene_list {gene_list_file}"

    cond_2_command = cond_2_command.split(" ")

    # Run the command
    if suppress_output:
        cond_2_result = subprocess.run(cond_2_command, capture_output=True, text=True)
    else:
        cond_2_result = subprocess.run(cond_2_command, capture_output=False, text=True)

    ## calculated permuted delta TMI value
    cond_1_rel_q_gk_outfile_name = 'dino_cond_1_null' + '_' + ctype + f"_df_rel_q_gk_ssl.pickle"
    cond_1_rel_q_gk = read_pickled_file(out_dir, cond_1_rel_q_gk_outfile_name)
    cond_1_rel_q_gk = -1 * cond_1_rel_q_gk

    cond_2_rel_q_gk_outfile_name = 'dino_cond_2_null' + '_' + ctype + f"_df_rel_q_gk_ssl.pickle"
    cond_2_rel_q_gk = read_pickled_file(out_dir, cond_2_rel_q_gk_outfile_name)
    cond_2_rel_q_gk = -1 * cond_2_rel_q_gk

    higher_si_in_cond_2 = cond_2_rel_q_gk.sub(cond_1_rel_q_gk, fill_value=0).dropna()

    return higher_si_in_cond_2 


def compute_p_values(higher_si_in_cond_2_true, higher_si_in_cond_2_across_perms):
    """Computes empirical p-values by comparing true statistics against permuted null distributions.

    For each gene, computes two-sided empirical p-values by comparing the true statistic against
    a null distribution generated from permutations. Returns both left-tailed and right-tailed p-values.

    Args:
        higher_si_in_cond_2_true (pd.DataFrame): DataFrame containing true statistics for each gene.
            Should have genes as index and a single column of statistics.
        higher_si_in_cond_2_across_perms (list): List of DataFrames containing permuted statistics.
            Each DataFrame should have same structure as higher_si_in_cond_2_true.

    Returns:
        tuple: Contains:
            - p_vals_left (list): Left-tailed p-values for each gene, computed as fraction of
                permuted statistics that are less than true statistic
            - p_vals_right (list): Right-tailed p-values for each gene, computed as fraction of
                permuted statistics that are greater than true statistic

    Raises:
        AssertionError: If indices of true and permuted DataFrames don't match
    """

    assert higher_si_in_cond_2_true.index.equals(higher_si_in_cond_2_across_perms[0].index)  
        # " The indices of the true and permuted dataframes are not the same"

    genes = higher_si_in_cond_2_true.index

    p_vals_left = []
    p_vals_right = []

    for gene in genes:
        real_stat = -1*higher_si_in_cond_2_true.loc[gene][0]

        null_stats = []
        for perm in higher_si_in_cond_2_across_perms:
            null_stats.append(perm.loc[gene][0])

        ## count how many times the real stat is greater than the null stats
        p_val_left = (real_stat < null_stats).sum()/len(null_stats)
        p_val_right = (real_stat > null_stats).sum()/len(null_stats)

        p_vals_left.append(p_val_left)
        p_vals_right.append(p_val_right)

    return p_vals_left, p_vals_right
        

def compute_true_tmi(cell_type_log_gene_rates_ng, spatial_log_gene_rates_ng, umi_n):
    """Computes the true Tissue Motif information score (TMI) for each gene given log gene rates and nUMI.

    Args:
        cell_type_log_gene_rates_ng (array-like): Log gene expression rates predicted from cell type
            composition alone. Shape (n_spots, n_genes).
        spatial_log_gene_rates_ng (array-like): Log gene expression rates predicted using spatial
            features. Shape (n_spots, n_genes).
        umi_n (array-like): UMI counts per spot. Shape (n_spots,).

    Returns:
        array-like: True TMI values for each gene, averaged across spots. Shape (n_genes,).
    """

    cell_type_log_gene_rates_norm_ng = torch.tensor(cell_type_log_gene_rates_ng) - torch.logsumexp(torch.tensor(cell_type_log_gene_rates_ng), dim=1, keepdim=True) 
    spatial_log_gene_rates_norm_ng = torch.tensor(spatial_log_gene_rates_ng) - torch.logsumexp(torch.tensor(spatial_log_gene_rates_ng), dim=1, keepdim=True) 

    cell_type_gene_rates_norm_ng = np.exp(cell_type_log_gene_rates_norm_ng)
    spatial_gene_rates_norm_ng = np.exp(spatial_log_gene_rates_norm_ng)

    cell_type_gene_counts_norm_ng = cell_type_gene_rates_norm_ng * umi_n[:,None]
    spatial_gene_counts_norm_ng = spatial_gene_rates_norm_ng * umi_n[:,None]

    true_tmi_ng = np.abs(spatial_gene_counts_norm_ng - cell_type_gene_counts_norm_ng) / cell_type_gene_counts_norm_ng 

    # true_tmi_ng = (np.abs(spatial_gene_counts_norm_ng - true_counts_ng) - np.abs(cell_type_gene_counts_norm_ng - true_counts_ng)) / np.abs(cell_type_gene_counts_norm_ng - true_counts_ng)

    true_tmi_g = true_tmi_ng.mean(axis=0)

    return true_tmi_g


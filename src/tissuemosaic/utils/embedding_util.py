"""
Set of functions to perform common desired tasks with embeddings, 
i.e. motif query, clustering, and conditional motif enrichment
"""

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


def motif_query(adata_ref: anndata.AnnData,
                adata_query: anndata.AnnData,
                query_point: Tuple[int, int],
                rep_key: str,
                dist_type: str = 'cosine',
                exponentiate: bool=True, ## used for cosine distance
                exponentiate_factor: int=5, ## used for cosine distance,
                sigmoid_transform: bool=False,
                sigmoid_factor: int=2
                ):
    """
    Query motif similarity between a query AnnData object at a specific point and a reference AnnData.

    Parameters:
    adata_ref (anndata.AnnData): Reference AnnData object.
    adata_query (anndata.AnnData): Query AnnData object.
    query_point (tuple): Coordinates (x, y) of the query point.
    rep_key (str): Key in `.obsm` for the representation to use.
    dist_type (str): Type of distance metric to use ('cosine' or 'euclidean'). Default is 'cosine'.
    exponentiate (bool): Whether to exponentiate the cosine similarity. Default is True.

    Returns:
    anndata.AnnData: Updated reference AnnData object with similarity scores in `.obs['sim']`.
    """

    x_query, y_query = query_point
    query_point_idx = np.argmin((adata_ref.obs['y'].values - x_query) ** 2 + (adata_ref.obs['x'].values - y_query) **2)

    ## Compute similarity of query patch to all patches in reference sample

    query_z = adata_query.obsm[rep_key][query_point_idx]

    if dist_type == 'euclidean':
        dist_n = np.linalg.norm(adata_ref.obsm[rep_key] - query_z[None, :], axis=-1)
        sim_n = np.exp(-dist_n / (2.0 * np.std(dist_n)))

    elif dist_type == 'cosine':
        sim_n = np.sum(adata_ref.obsm[rep_key] * query_z[None, :], -1) / (np.linalg.norm(adata_ref.obsm[rep_key], axis=-1) * np.linalg.norm(query_z))
        sim_n = np.clip(sim_n, a_min=0., a_max=1.) ** exponentiate_factor
        # if exponentiate:
        #     sim_n = np.clip(sim_n, a_min=0., a_max=1.) ** exponentiate_factor
        # else:
        #     sim_n = np.clip(sim_n, a_min=0., a_max=1.)

        if sigmoid_transform:
            sim_n = ((1 / (1 + np.exp(-sigmoid_factor * (sim_n))))-0.5) * 2

        if exponentiate:
            sim_n = np.clip(sim_n, a_min=0., a_max=1.) ** exponentiate_factor

    adata_ref.obs['sim'] = sim_n

    return adata_ref


def cluster(adata: anndata.AnnData,
               key: str,
               leiden_res: Union[float, list],
               n_neighbors: int = 500,
               umap_preprocess_strategy: str = 'raw',
               cluster_patches: bool = True,
               x_key: str = 'x',
               y_key: str = 'y',
               category_key: str = 'cell_type_proportions',
               low_memory: bool = False):
    """
    Cluster embeddings in an AnnData object by computing UMAP and running leiden clustering on the UMAP graph

    Parameters:
    adata (anndata.AnnData): Input AnnData object.
    key (str): Key in `.obsm` for the representation to use.
    leiden_res (Union[float, List[float]]): Resolution(s) for Leiden clustering.
    n_neighbors (int): Number of neighbors for UMAP. Default is 500.
    umap_preprocess_strategy (str): Preprocessing strategy for UMAP. Default is 'raw'.
    cluster_patches (bool): If true, use coarser patch representations for clustering. If false, use spot representations "{key}_spot_features"
    x_key (str): Key for x-coordinates in `.obs`. Default is 'x'.
    y_key (str): Key for y-coordinates in `.obs`. Default is 'y'.
    category_key (str): Key for category information in `.obs`. Default is 'cell_type_proportions'.
    low_memory (bool): Whether to use low memory mode for UMAP. Default is False.

    Returns:
    anndata.AnnData: Updated AnnData object with UMAP embeddings under "umap_{key}" and clustering results under "leiden_feature_{key}_res_{res}_one_hot" in .obsm.
    """
    
    if cluster_patches:
        patch_properties_dict = adata.uns['sparse_image_state_dict']['patch_properties_dict']
        print("Running UMAP")
        smart_pca = SmartPca(preprocess_strategy='z_score')
        smart_umap = SmartUmap(n_neighbors=n_neighbors, preprocess_strategy=umap_preprocess_strategy, n_components=2, min_dist=0.5, metric='euclidean', low_memory=low_memory)

        input_features = patch_properties_dict[key]
        embeddings_pca = smart_pca.fit_transform(input_features, n_components=0.9)
        embeddings_umap = smart_umap.fit_transform(embeddings_pca)

        # adata.obsm["pca_"+key] = np.array(embeddings_pca)
        patch_properties_dict["umap_"+key] = np.array(embeddings_umap)
        patch_properties_dict["umap_graph_"+key] = smart_umap.get_graph()

        ## Compute clustering
        print("Computing clusters")
        umap_graph = smart_umap.get_graph()
        smart_leiden = SmartLeiden(graph=umap_graph)

        if not isinstance(leiden_res, list):
            leiden_res = [leiden_res]

        leiden_keys_one_hot = []
        leiden_keys = []
        for res in leiden_res:
            leiden_clusters = smart_leiden.cluster(resolution=res, partition_type='RBC')
            patch_properties_dict["leiden_feature_" + key + "_" +str(res)] = leiden_clusters
            patch_properties_dict["leiden_feature_" + key + "_res_"+str(res)+"_one_hot"] = torch.nn.functional.one_hot(torch.from_numpy(leiden_clusters).long())  
            leiden_keys_one_hot.append("leiden_feature_" + key + "_res_"+str(res)+"_one_hot")
            leiden_keys.append("leiden_feature_" + key + "_" +str(res))

        sp_img = SparseImage.from_anndata(adata, x_key=x_key, y_key=y_key, category_key=category_key)

        for leiden_key_one_hot in leiden_keys_one_hot:
                sp_img.write_to_patch_dictionary(leiden_key_one_hot, torch.Tensor(patch_properties_dict[leiden_key_one_hot]).float(), patches_xywh = patch_properties_dict[key + '_patch_xywh'], overwrite = True)

        for leiden_key in leiden_keys:
                sp_img.write_to_patch_dictionary(leiden_key, torch.Tensor(patch_properties_dict[leiden_key]).float(), patches_xywh = patch_properties_dict[key + '_patch_xywh'], overwrite = True)


        # Note that I use the one_hot version of the Leiden clusters to transfer the annotations
        sp_img.transfer_patch_to_spot(
            keys_to_transfer=leiden_keys_one_hot,
            overwrite=True)

        sp_img.transfer_patch_to_spot(
            keys_to_transfer=leiden_keys,
            overwrite=True)

        # print(sp_img.spot_properties_dict['leiden_feature_dino_res_0.05_one_hot'].shape)
        out_adata = sp_img.to_anndata()

        return out_adata
    else:
        assert isinstance(adata, anndata.AnnData), 'Need to pass an anndata object'

        ## Compute UMAP 
            
        print("Running UMAP")
        smart_pca = SmartPca(preprocess_strategy='z_score')
        smart_umap = SmartUmap(n_neighbors=n_neighbors, preprocess_strategy=umap_preprocess_strategy, n_components=2, min_dist=0.5, metric='euclidean', low_memory=low_memory)
                
        input_features = adata.obsm[key]
        embeddings_pca = smart_pca.fit_transform(input_features, n_components=0.9)
        embeddings_umap = smart_umap.fit_transform(embeddings_pca)
            
        # adata.obsm["pca_"+key] = np.array(embeddings_pca)
        adata.obsm["umap_"+key] = np.array(embeddings_umap)
        adata.obsm["umap_graph_"+key] = smart_umap.get_graph()
        
        ## Compute clustering
        print("Computing clusters")
        umap_graph = smart_umap.get_graph()
        smart_leiden = SmartLeiden(graph=umap_graph)

        if not isinstance(leiden_res, list):
            leiden_res = [leiden_res]

        for res in leiden_res:
            leiden_clusters = smart_leiden.cluster(resolution=res, partition_type='RBC')
            adata.obs["leiden_feature_" + key + "_" +str(res)] = leiden_clusters
            adata.obsm["leiden_feature_" + key + "_res_"+str(res)+"_one_hot"] = np.array(torch.nn.functional.one_hot(torch.from_numpy(leiden_clusters).long()))  
        
        return adata

def conditional_motif_enrichment(adata: Anndata,
                                 feature_key: str,
                                 classify_or_regress: str,
                                 alpha_regularization: Union[float, list],
                                 train_test_split_strategy: str="spot",
                                 verbose: bool = False):
    """
    Perform conditional motif enrichment by training classifiers on spatially partitioned train/test folds.

    Parameters:
    adata (anndata.AnnData): Input AnnData object.
    feature_key (str): Key in `.obsm` for the representation to use.
    classify_or_regress (str): Enrichment, either 'classify' or 'regress' depending on whether external condition is categorical or continuous variable.
    alpha_regularization (Union[float, List[float]]): Regularization parameter(s) for Ridge classifier.
    verbose (bool): Whether to print detailed output. Default is False.

    Returns:
    anndata.AnnData: Merged AnnData object containing enrichment scores (classification or regression output) for spatially partitioned test folds
    """

    all_folds_test_anndatas = []

    if not isinstance(alpha_regularization, list):
        alpha_regularization = [alpha_regularization]
    
    if classify_or_regress == 'classify':
    
        ## Train classifier on spatially partitioned train/test folds
        for kfold in range(1,5):

            print(f"Running kfold {kfold}")
            
            ## spatial split
            if train_test_split_strategy == "spot":
                train_anndata = adata[adata.obs[f'train_test_fold_{kfold}'] == 0]
                test_anndata = adata[adata.obs[f'train_test_fold_{kfold}'] == 1]
            elif train_test_split_strategy == "patch":
                train_anndata = adata[adata.obs[f'train_test_split_fold_{kfold}_id'] == 0]
                test_anndata = adata[adata.obs[f'train_test_split_fold_{kfold}_id'] == 1]

            ## balance after train/test split b/c of spatial partitioning to downsample majority class
            train_anndata = balance_anndata(train_anndata, 'classify_condition')
            
            ## Train classifier
            classifier = RidgeClassifierCV(alphas=alpha_regularization)
            classifier.fit(train_anndata.obsm[feature_key], train_anndata.obs['classify_condition'])

            if verbose:
                print("Fit alpha:")
                print(classifier.alpha_)

                ## Confirm train/test scores are similar to ensure regularization worked as expected
                print("Train score:")
                print(classifier.score(train_anndata.obsm[feature_key], train_anndata.obs['classify_condition']))

                print("Test score:")
                print(classifier.score(test_anndata.obsm[feature_key], test_anndata.obs['classify_condition']))
                
            ## Get conditional motif enrichment from trained classifier and write to anndata
            predicted_condition = classifier.decision_function(test_anndata.obsm[feature_key])
            test_anndata.obs['predicted_condition'] = predicted_condition
            all_folds_test_anndatas.append(test_anndata)
    elif classify_or_regress == "regress":
            
        ## Train regressor on spatially partitioned train/test folds
        for kfold in range(1,5):

            print(f"Running kfold {kfold}")
            
            ## spatial split
            train_anndata = adata[adata.obs[f'train_test_fold_{kfold}'] == 0]
            test_anndata = adata[adata.obs[f'train_test_fold_{kfold}'] == 1]
            
            ## Train regressor
            regressor = RidgeCV(alphas=alpha_regularization)
            regressor.fit(train_anndata.obsm[feature_key], train_anndata.obs['regress_condition'])

            if verbose:
                print("Fit alpha:")
                print(regressor.alpha_)

                ## Confirm train/test scores are similar to ensure regularization worked as expected
                print("Train score:")
                print(regressor.score(train_anndata.obsm[feature_key], train_anndata.obs['regress_condition']))

                print("Test score:")
                print(regressor.score(test_anndata.obsm[feature_key], test_anndata.obs['regress_condition']))
                
            ## Get conditional motif enrichment from trained classifier and write to anndata
            predicted_condition = regressor.predict(test_anndata.obsm[feature_key])
            test_anndata.obs['predicted_condition'] = predicted_condition
            all_folds_test_anndatas.append(test_anndata)


    ## Merge test folds together
    all_test_anndatas = merge_anndatas_inner_join(all_folds_test_anndatas)

    return all_test_anndatas
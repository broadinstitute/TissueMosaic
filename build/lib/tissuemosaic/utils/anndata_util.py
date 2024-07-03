from typing import List, Any
import anndata as Anndata
import numpy as np
from tissuemosaic.utils.dict_util import concatenate_list_of_dict

# Set of simple helper functions to manipulate anndatas

def merge_anndatas_inner_join(anndata_list: List[Anndata.AnnData],
                              merge_uns: bool=False):
    """
    Merge a list of anndata objects using an inner join operation.

    Args:
        anndata_list (list): List of anndata objects to be merged.

    Returns:
        anndata.AnnData: The merged anndata object.
    """
    # Check if the anndata_list is not empty
    if not anndata_list:
        raise ValueError("Input anndata_list is empty.")

    # Determine the common feature names across all anndata objects
    common_feature_names = anndata_list[0].var_names
    for ad in anndata_list[1:]:
        common_feature_names = common_feature_names.intersection(ad.var_names)

    # Filter observations for each anndata object using the common feature names
    filtered_anndata_list = []
    for ad in anndata_list:
        ad_filtered = ad[:, common_feature_names]
        filtered_anndata_list.append(ad_filtered)

    # Concatenate the list of filtered anndata objects along axis 0 (rows)
    merged_anndata = Anndata.concat(filtered_anndata_list, axis=0, uns_merge='same')

    # Combine .uns entries
    if merge_uns and merged_anndata.uns is not None:
        combined_uns = {}
        list_of_patch_properties_dicts = []
        list_of_image_properties_dicts = []
        properties_dicts_exists = False
        for i, adata in enumerate(anndata_list):
            for key, value in adata.uns.items():
                if key not in combined_uns:
                    combined_uns[key] = value
                else:
                    pass
                if key == 'sparse_image_state_dict':
                    # try:
                    patch_key = list(value['patch_properties_dict'].keys())[0]
                    value['patch_properties_dict']['patch_sample_id'] = i * np.ones(value['patch_properties_dict'][patch_key].shape[0])

                    list_of_patch_properties_dicts.append(value['patch_properties_dict'])
                    list_of_image_properties_dicts.append(value['image_properties_dict'])
                    properties_dicts_exists = True
                    # except:
                        # pass

        # ignore image properties dict for now
        if properties_dicts_exists:
            combined_patch_properties_dict = concatenate_list_of_dict(list_of_patch_properties_dicts)
            # combined_image_properties_dict = concatenate_list_of_dict(list_of_image_properties_dicts)
            combined_uns['sparse_image_state_dict']['patch_properties_dict'] = combined_patch_properties_dict
            # combined_uns['sparse_image_state_dict']['image_properties_dict'] = combined_image_properties_dict
            combined_uns['sparse_image_state_dict']['image_properties_dict'] = {}

        merged_anndata.uns = combined_uns

    return merged_anndata

def balance_anndata(adata, label_key):
    """
    Balances the classes in an anndata object stored in obs under 'label_key' by downsampling the majority class.

    Args:
        adata: the anndata object to be balanced
        label_key: the key in adata.obs where the class labels are stored

    Returns:
        anndata.AnnData: The balanced anndata object.
    """

    # Assuming you have an AnnData object `adata` with the class label column in adata.obs['label']

    # Calculate class counts
    class_counts = adata.obs[label_key].value_counts()

    # Identify majority and minority classes
    majority_class_label = class_counts.idxmax()
    minority_class_label = class_counts.idxmin()

    # Separate majority and minority classes
    adata_majority = adata[adata.obs[label_key] == majority_class_label].copy()
    adata_minority = adata[adata.obs[label_key] == minority_class_label].copy()

    # Downsample majority class
    np.random.seed(123)  # for reproducibility
    random_subset_indices = np.random.choice(adata_majority.shape[0], size=len(adata_minority), replace=False)
    adata_majority_downsampled = adata_majority[random_subset_indices]

    # Combine minority class with downsampled majority class
    adata_balanced = adata_minority.concatenate(adata_majority_downsampled)
    
    return adata_balanced
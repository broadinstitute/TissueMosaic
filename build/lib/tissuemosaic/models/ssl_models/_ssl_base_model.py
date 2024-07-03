import numpy
import torch
import pandas
from typing import Dict, List, Sequence, Any
from neptune.types import File
from pytorch_lightning import LightningModule

from sklearn.base import is_regressor, is_classifier
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import RidgeClassifierCV, RidgeCV

from tissuemosaic.data.dataset import MetadataCropperDataset
from tissuemosaic.plots.plot_images import show_raw_all_channels
from tissuemosaic.plots.plot_embeddings import plot_embeddings
from tissuemosaic.utils.nms_util import NonMaxSuppression
from tissuemosaic.utils.dict_util import (
    concatenate_list_of_dict,
    subset_dict)

import numpy as np

from tissuemosaic.utils.validation_util import (
    SmartPca,
    SmartUmap)

def classify_and_regress(
        input_dict: dict,
        feature_keys: List[str],
        regress_keys: List[str] = None,
        classify_keys: List[str] = None,
        regressor: "sklearn_like_regressor" = None,
        classifier: "sklearn_like_classifier" = None,
        n_splits: int = 5,
        n_repeats: int = 1,
        return_regressor: bool = False,
        return_classifier: bool = False,
        verbose: bool = False) -> [pandas.DataFrame, pandas.DataFrame]:
    """
    Train a Classifier and a Regressor to use some features to classify/predict other annotations.

    Args:
        input_dict: dict with both the features and the annotations
        regressor: the regressor to train
        classifier: the classifier to train
        feature_keys: keys corresponding to the independent variables in the :attr:`input_dict`.
        regress_keys: keys corresponding to the variables to regress in the :attr:`input_dict`.
        classify_keys: keys corresponding to the variables to classify in the :attr:`input_dict`.
        n_splits: int, number of splits for RepeatedKFold (regressor) or RepeatedStratifiedKFold (classifier).
            If n_splits is 5 (defaults) then train_test_split is 80% - 20%.
        n_repeats: int, number of repeats for RepeatedKFold (regressor) or RepeatedStratifiedKFold (classifier).
            The total number of trained model is n_plists * n_repeats.
        verbose: if True (default is False) prints some intermediate statements

    Returns:
        A dataframe. Each row is a different X,y combination with the metrics describing the quality of the
        regression/classification.
    """
    if regress_keys is not None:
        assert is_regressor(regressor) or regressor.is_regressor, "Please pass in a regressor"

    if classify_keys is not None:
        assert is_classifier(classifier) or classifier.is_classifier, "Please pass in a classifier"

    assert isinstance(n_splits, int) and isinstance(n_repeats, int) and n_splits >= 1 and n_repeats >= 1, \
        "Error. n_splits = {0} and n_repeats = {1} must be integers >= 1.".format(n_splits, n_repeats)
    assert n_splits > 1 or (n_splits == 1 and n_repeats == 1), \
        "Misconfiguration error. It does not make sense to have n_splits == 1 and n_repeats != 1"

    assert isinstance(feature_keys, list), \
        "Feature_keys need to be a list. Received {0}".format(type(feature_keys))
    assert regress_keys is None or isinstance(regress_keys, list), \
        "Regress_keys need to be a list. Received {0}".format(type(regress_keys))
    assert classify_keys is None or isinstance(classify_keys, list), \
        "Classify_keys need to be a list. Received {0}".format(type(classify_keys))

    assert set(feature_keys).issubset(input_dict.keys()), \
        "Feature keys are not present in input dictionary."
    assert regress_keys is None or set(regress_keys).issubset(set(input_dict.keys())), \
        "Regress keys are not present in input dictionary."
    assert classify_keys is None or set(classify_keys).issubset(set(input_dict.keys())), \
        "Classify keys are not present in input dictionary."

    def _manual_shuffle(_X, _y):
        assert _X.shape[0] == _y.shape[0]
        random_index = numpy.random.permutation(_y.shape[0])
        return _X[random_index], _y[random_index]

    def _preprocess_to_numpy(x, len_shape: int):
        """ convert the features into a 2D numpy tensor (n, p) and the targets into a 1D numpy tensor (n) """
        assert isinstance(len_shape, int) and (len_shape == 1 or len_shape == 2)
        if isinstance(x, torch.Tensor):
            x = x.flatten(end_dim=-len_shape)
            assert len(x.shape) == len_shape
            return x.cpu().numpy()
        elif isinstance(x, numpy.ndarray):
            assert len(x.shape) == len_shape
            return x
        elif isinstance(x, list):
            assert len_shape == 1
            return numpy.array(x)

    def _do_regression(_X, _y, x_key, y_key):
        _X, _y = _manual_shuffle(_X, _y)
        _tmp_dict = {}

        if n_splits > 1:
            rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
            for train_index, test_index in rkf.split(_X, _y):
                _X_train, _X_test, _y_train, _y_test = _X[train_index], _X[test_index], _y[train_index], _y[test_index]
                regressor.fit(_X_train, _y_train)

                _tmp_dict["x_key"] = _tmp_dict.get("x_key", []) + [x_key]
                _tmp_dict["y_key"] = _tmp_dict.get("y_key", []) + [y_key]
                _tmp_dict["r2_train"] = _tmp_dict.get("r2_train", []) + [regressor.score(_X_train, _y_train)]
                _tmp_dict["r2_test"] = _tmp_dict.get("r2_test", []) + [regressor.score(_X_test, _y_test)]
            _df_tmp = pandas.DataFrame(_tmp_dict, index=numpy.arange(rkf.get_n_splits()))
        elif n_splits == 1 and n_repeats == 1:
            regressor.fit(_X, _y)
            _tmp_dict = {
                "x_key": x_key,
                "y_key": y_key,
                "r2": regressor.score(_X, _y)
            }
            _df_tmp = pandas.DataFrame(_tmp_dict, index=[0])
        else:
            raise Exception("Does not make sense to have n_splits = {0} and n_repeats = {1}".format(n_splits,
                                                                                                    n_repeats))
        return _df_tmp

    def _do_classification(_X, _y, x_key, y_key):
        _X, _y = _manual_shuffle(_X, _y)
        _tmp_dict = {}

        
        if n_splits > 1:

            rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
            for train_index, test_index in rkf.split(_X, _y):
                _X_train, _X_test, _y_train, _y_test = _X[train_index], _X[test_index], _y[train_index], _y[test_index]
                classifier.fit(_X_train, _y_train)

                _tmp_dict["x_key"] = _tmp_dict.get("x_key", []) + [x_key]
                _tmp_dict["y_key"] = _tmp_dict.get("y_key", []) + [y_key]
                
                _tmp_dict["accuracy_train"] = _tmp_dict.get("accuracy_train", []) + [classifier.score(_X_train, _y_train)]
                _tmp_dict["accuracy_test"] = _tmp_dict.get("accuracy_test", []) + [classifier.score(_X_test, _y_test)]
            _df_tmp = pandas.DataFrame(_tmp_dict, index=numpy.arange(rkf.get_n_splits()))

        elif n_splits == 1 and n_repeats == 1:
            classifier.fit(_X, _y)
            _tmp_dict = {
                "x_key": x_key,
                "y_key": y_key,
                "accuracy": classifier.score(_X, _y)
            }
            _df_tmp = pandas.DataFrame(_tmp_dict, index=[0])
        else:
            raise Exception("Does not make sense to have n_splits = {0} and n_repeats = {1}".format(n_splits,
                                                                                                    n_repeats))
        return _df_tmp

    # loop over everything to make the predictions
    df = None
    for feature_key in feature_keys:
        X_all = _preprocess_to_numpy(input_dict[feature_key], len_shape=2)

        if classify_keys is not None:
            for kc in classify_keys:
                if verbose:
                    print("{0} classify {1}".format(feature_key, kc))
                y_all = _preprocess_to_numpy(input_dict[kc], len_shape=1)
                tmp_df = _do_classification(X_all, y_all, x_key=feature_key, y_key=kc)
                df = tmp_df if df is None else df.merge(tmp_df, how='outer')

        if regress_keys is not None:
            for kr in regress_keys:
                if verbose:
                    print("{0} regress {1}".format(feature_key, kr))
                y_all = _preprocess_to_numpy(input_dict[kr], len_shape=1)
                tmp_df = _do_regression(X_all, y_all, x_key=feature_key, y_key=kr)
                df = tmp_df if df is None else df.merge(tmp_df, how='outer')

    if return_regressor:
        return df, regressor
    if return_classifier:
        return df, classifier
    
    return df

##TODO: finish this function
def train_test_split_dict(world_dict: dict, 
                          remove_overlap: bool, 
                          val_iomin_threshold: float=False,
                          stratify: bool=True,
                          res: int=0.4, 
                          train_size: float = 0.8,
                          test_size: float = 0.2,
                          random_state: int = 0):
    """
    Utility function to split dictionary into train and test dictionaries (that are optionally split spatially)
    It takes a dictionary with patch-level features and annotations
    
    Args:
        world_dict: a dictionary with path-level features and annotations.
            The dict must contains the keys "patches_xywh" and "classify_tissue_label".
            It may contain additional keys starting in "feature_", "classify_" or "regress_".
        remove_overlap: whether to conduct train/test split spatially (for example if patch tiling strategy to generate patch features was 'random')
        val_iomin_threshold: threshold for the Intersection over Minimum. It must be in [0.0, 1.0).
        stratify: whether to stratify by patch NCV clusters. Stratifies by tissue label by default.
        res: Leiden cluster resolution for determining patch NCV clusters
        train_size: size of train set
        test_size: set of test set
    """
    
    # compute the patch_to_patch overlap just one at the beginning
    assert {"patches_xywh", "classify_tissue_label"}.issubset(world_dict.keys())
    patches = world_dict["patches_xywh"]
    
    print(world_dict['patches_xywh'].shape)
    
    if remove_overlap:
        initial_score = torch.rand_like(patches[:, 0].float())
        tissue_ids = world_dict["classify_tissue_label"]
        nms_mask_n, overlap_nn = NonMaxSuppression.compute_nm_mask(
            score=initial_score,
            ids=tissue_ids,
            patches_xywh=patches,
            iom_threshold=val_iomin_threshold)
        binarized_overlap_nn = (overlap_nn > val_iomin_threshold).float()

        # create a dictionary with only non-overlapping patches to test kn-regressor/classifier
        nms_mask_n = NonMaxSuppression.perform_nms_selection(mask_overlap_nn=binarized_overlap_nn,
                                                             score_n=torch.rand_like(initial_score),
                                                             possible_n=torch.ones_like(initial_score).bool())
        world_dict_subset = subset_dict(input_dict=world_dict, mask=nms_mask_n)
        world_dict = world_dict_subset
    
    print(world_dict['patches_xywh'].shape)
    
    ## Cluster Patch NCVs
    if stratify:
        assert "ncv" in self._patch_properties_dict.keys(), \
            "Compute Patch NCV first."

        ## Cluster patch NCVs
        patch_ncv = self._patch_properties_dict["ncv"]

        assert np.array_equal(self._patch_properties_dict["ncv_patch_xywh"], self._patch_properties_dict[feature_xywh]), \
            "NCV patch xywh does not match feature xywh"

        smart_umap = SmartUmap(n_neighbors=25, preprocess_strategy='z_score', n_components=2, min_dist=0.5, metric='cosine')
        embeddings_umap = smart_umap.fit_transform(patch_ncv)

        umap_graph = smart_umap.get_graph()
        smart_leiden = SmartLeiden(graph=umap_graph)

        leiden_clusters = smart_leiden.cluster(resolution=res, partition_type='RBC')
        
    ## Split patches into train/test, stratifying by NCV clusters
    ## write custom train_test_split function in utils
    
    all_indices = np.arange(world_dict['patches_xywh'].shape[0])

    train_indices, test_indices = train_test_split(all_indices, stratify=world_dict['classify_tissue_label'], random_state=random_state)

    train_mask = torch.tensor(np.isin(all_indices, train_indices))
    test_mask = torch.tensor(np.isin(all_indices, test_indices))
    
    train_features_dict = subset_dict(world_dict, train_mask)
    test_features_dict = subset_dict(world_dict, test_mask)
    # ##TODO: see why this isn't working as expected
    # if stratify:
    #     try:
    #         train_patch_xywh, test_patch_xywh = train_test_split(self._patch_properties_dict[feature_xywh], train_size=train_size, test_size=test_size, stratify=leiden_clusters,random_state=random_state)
    #     except ValueError: #ValueError as ve: ##TODO: specify for the appropriate exception here
    #         # raise Exception("ValueError: " + str(ve))
    #         raise Exception("Not enough samples in each cluster to split the data. Try a smaller res")
    # else:
    #     train_patch_xywh, test_patch_xywh = train_test_split(self._patch_properties_dict[feature_xywh], train_size=train_size, test_size=test_size,
    #                                                         random_state=random_state)
        
    return train_features_dict, test_features_dict
        

    


def knn_classification_regression(world_dict: dict, val_iomin_threshold: float):
    """
    Utility function to perform knn-based classification and regression.
    It takes a dictionary with patch-level features and annotations.
    A set of (weakly) overlapping patches (with Intersection over Minimum smaller than the assigned threshold)
    is selected. A knn classifier/regressor is trained to
    predict the various annotations starting from the features.
    This process is repeated multiple times for all feature-annotation pairs to produce a confidence interval.

    Args:
        world_dict: a dictionaries with path-level features and annotations.
            The dict must contains the keys "patches_xywh" and "classify_tissue_label".
            It may contain additional keys starting in "feature_", "classify_" or "regress_".
        val_iomin_threshold: threshold for the Intersection over Minimum. It must be in [0.0, 1.0).
    """

    # compute the patch_to_patch overlap just one at the beginning
    assert {"patches_xywh", "classify_tissue_label"}.issubset(world_dict.keys())
    patches = world_dict["patches_xywh"]
    initial_score = torch.rand_like(patches[:, 0].float())
    tissue_ids = world_dict["classify_tissue_label"]
    nms_mask_n, overlap_nn = NonMaxSuppression.compute_nm_mask(
        score=initial_score,
        ids=tissue_ids,
        patches_xywh=patches,
        iom_threshold=val_iomin_threshold)
    binarized_overlap_nn = (overlap_nn > val_iomin_threshold).float()
    # print("# non-overlapping patches->", nms_mask_n.sum())

    # figure out the keys for the features, regression and classification
    feature_keys, regress_keys, classify_keys = [], [], []
    for key in world_dict.keys():
        if key.startswith("regress"):
            regress_keys.append(key)
        elif key.startswith("classify"):
            classify_keys.append(key)
        elif key.startswith("pca_") or key.startswith("umap_") or key.startswith("feature"):
            feature_keys.append(key)

    # define regressor and classifier
    def exclude_self(d):
        # This has shape: d.shape = (n_points, n_neighbours)
        # This function relies on the fact that the distances are sorted in increasing order
        w = numpy.ones_like(d)
        w[:, 0] = 0.0
        return w

    kn_kargs = {
        "n_neighbors": 5,
        "weights": exclude_self,
    }

    regressor = KNeighborsRegressor(**kn_kargs)
    classifier = KNeighborsClassifier(**kn_kargs)

    # loop over subset made of non-overlapping patches
    df_tot = None
    for n in range(10): #range(20): #oom error
        
        # create a dictionary with only non-overlapping patches to test kn-regressor/classifier
        nms_mask_n = NonMaxSuppression.perform_nms_selection(mask_overlap_nn=binarized_overlap_nn,
                                                             score_n=torch.rand_like(initial_score),
                                                             possible_n=torch.ones_like(initial_score).bool())
        world_dict_subset = subset_dict(input_dict=world_dict, mask=nms_mask_n)

        df_tmp = classify_and_regress(
            input_dict=world_dict_subset,
            feature_keys=feature_keys,
            regress_keys=regress_keys,
            classify_keys=classify_keys,
            regressor=regressor,
            classifier=classifier,
            n_repeats=1,
            n_splits=1,
            verbose=False,
        )
        df_tot = df_tmp if df_tot is None else df_tot.merge(df_tmp, how='outer')

    df_tot["combined_key"] = df_tot["x_key"] + "_" + df_tot["y_key"]
    df_mean = df_tot.groupby("combined_key").mean()
    df_std = df_tot.groupby("combined_key").std()
    return df_mean, df_std


def linear_classification_regression(world_dict: dict, val_iomin_threshold: float):
    """
    Utility function to perform linear classification and regression.
    It takes a dictionaries with path-level features and annotations.
    A set of (weakly) overlapping patches (with Intersection over Minimum smaller than the assigned threshold)
    is selected. A linear classifier/regressor is trained to
    predict the various annotations starting from the features.
    This process is repeated multiple times for all feature-annotation pairs to produce a confidence interval.

    Args:
        world_dict: a dictionaries with path-level features and annotations.
            The dict must contains the keys "patches_xywh" and "classify_tissue_label".
            It may contain additional keys starting in "feature_", "classify_" or "regress_".
        val_iomin_threshold: threshold for the Intersection over Minimum. It must be in [0.0, 1.0).
    """

    # compute the patch_to_patch overlap just one at the beginning
    assert {"patches_xywh", "classify_tissue_label"}.issubset(world_dict.keys())
    patches = world_dict["patches_xywh"]
    initial_score = torch.rand_like(patches[:, 0].float())
    tissue_ids = world_dict["classify_tissue_label"]
    nms_mask_n, overlap_nn = NonMaxSuppression.compute_nm_mask(
        score=initial_score,
        ids=tissue_ids,
        patches_xywh=patches,
        iom_threshold=val_iomin_threshold)
    binarized_overlap_nn = (overlap_nn > val_iomin_threshold).float()
    # print("# non-overlapping patches->", nms_mask_n.sum())

    # figure out the keys for the features, regression and classification
    feature_keys, regress_keys, classify_keys = [], [], []
    for key in world_dict.keys():
        if key.startswith("regress"):
            regress_keys.append(key)
        elif key.startswith("classify"):
            classify_keys.append(key)
        elif key.startswith("pca_") or key.startswith("umap_") or key.startswith("feature"):
            feature_keys.append(key)

    # define regressor and classifier
    ridge_kargs = {
        "alphas": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    }

    regressor = RidgeCV(**ridge_kargs)
    classifier = RidgeClassifierCV(**ridge_kargs)

    # loop over subset made of non-overlapping patches
    df_tot = None
    for n in range(10): #range(20): # oom error
        # create a dictionary with only non-overlapping patches to test kn-regressor/classifier
        nms_mask_n = NonMaxSuppression.perform_nms_selection(mask_overlap_nn=binarized_overlap_nn,
                                                             score_n=torch.rand_like(initial_score),
                                                             possible_n=torch.ones_like(initial_score).bool())
        world_dict_subset = subset_dict(input_dict=world_dict, mask=nms_mask_n)

        df_tmp = classify_and_regress(
            input_dict=world_dict_subset,
            feature_keys=feature_keys,
            regress_keys=regress_keys,
            classify_keys=classify_keys,
            regressor=regressor,
            classifier=classifier,
            n_repeats=1,
            n_splits=5,
            verbose=False,
        )
        df_tot = df_tmp if df_tot is None else df_tot.merge(df_tmp, how='outer')

    df_tot["combined_key"] = df_tot["x_key"] + "_" + df_tot["y_key"]
    df_mean = df_tot.groupby("combined_key").mean()
    df_std = df_tot.groupby("combined_key").std()
    return df_mean, df_std  


class SslModelBase(LightningModule):
    """
    Base class for the self-supervised learning (ssl) models (Vae, Dino, Barlow, Simclr).
    This base class is responsible for the validation (which is common to all ssl models) and some logging.
    The child classes need to implement :meth:`head_and_backbone_embeddings_step`, :meth:`forward` and
    :meth:`training_step`.
    """
    def __init__(self, val_iomin_threshold: float, run_classify_regress: bool):
        super(SslModelBase, self).__init__()
        self.val_iomin_threshold = val_iomin_threshold
        self.run_classify_regress = run_classify_regress
        self.neptune_run_id = None

    def head_and_backbone_embeddings_step(self, x) -> (torch.Tensor, torch.Tensor):
        # must be overwritten by child class
        # this generates both head and backbone embeddings
        raise NotImplementedError

    def forward(self, x) -> torch.Tensor:
        # must be overwritten by child class
        # this is the stuff that will generate the backbone embeddings
        raise NotImplementedError

    def training_step(self, batch, batch_idx) -> dict:
        # must be overwritten by child class
        raise NotImplementedError

    def get_metadata_to_regress(self, metadata) -> dict:
        try:
            return self.trainer.datamodule.get_metadata_to_regress(metadata)
        except AttributeError:
            return dict()

    def get_metadata_to_classify(self, metadata) -> dict:
        try:
            return self.trainer.datamodule.get_metadata_to_classify(metadata)
        except AttributeError:
            return dict()

    @property
    def n_global_crops(self):
        return self.trainer.datamodule.n_global_crops

    @property
    def n_local_crops(self):
        return self.trainer.datamodule.n_local_crops

    @property
    def trsfm_train_local(self):
        return self.trainer.datamodule.trsfm_train_local

    @property
    def trsfm_train_global(self):
        return self.trainer.datamodule.trsfm_train_global

    @property
    def trsfm_test(self):
        return self.trainer.datamodule.trsfm_test

    def __log_example_images__(self, which_loaders: str, n_examples: int = 10, n_cols: int = 5):
        if which_loaders == "val":
            loaders = self.trainer.datamodule.val_dataloader()
            log_name = "val_imgs"
        elif which_loaders == "train":
            loaders = self.trainer.datamodule.train_dataloader()
            log_name = "train_imgs"
        elif which_loaders == "predict":
            loaders = self.trainer.datamodule.predict_dataloader()
            log_name = "predict_imgs"
        else:
            raise Exception("Invalid value for which_loaders. Expected 'val' or 'train' or 'predict'. \
            Received={0}".format(which_loaders))

        if not isinstance(loaders, Sequence):
            loaders = [loaders]

        for idx_dataloader, loader in enumerate(loaders):
            indeces = torch.randperm(n=loader.dataset.__len__())[:n_examples]
            list_imgs, _, _ = loader.load(index=indeces)
            list_imgs = list_imgs[:n_examples]

            tmp_ref = self.trsfm_test(list_imgs)
            tmp_ref_plot = show_raw_all_channels(tmp_ref, n_col=n_cols, show_colorbar=True)
            self.logger.run[log_name + "/ref_" + str(idx_dataloader)].log(File.as_image(tmp_ref_plot))

            if which_loaders == 'train':
                tmp_global = self.trsfm_train_global(list_imgs)
                tmp_global_plot = show_raw_all_channels(tmp_global, n_col=n_cols, show_colorbar=True)
                self.logger.run[log_name+"/global"].log(File.as_image(tmp_global_plot))

                tmp_local = self.trsfm_train_local(list_imgs)
                tmp_local_plot = show_raw_all_channels(tmp_local, n_col=n_cols, show_colorbar=True)
                self.logger.run[log_name + "/local"].log(File.as_image(tmp_local_plot))

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            self.__log_example_images__(n_examples=10, n_cols=5, which_loaders="train")

    def on_predict_start(self) -> None:
        if self.global_rank == 0:
            self.__log_example_images__(n_examples=10, n_cols=5, which_loaders="predict")

    def validation_step(self, batch, batch_idx, dataloader_idx: int = -1) -> Dict[str, torch.Tensor]:
        list_imgs: List[torch.sparse.Tensor]
        list_labels: List[int]
        list_metadata: List[MetadataCropperDataset]
        list_imgs, list_labels, list_metadata = batch

        # Compute the embeddings
        img = self.trsfm_test(list_imgs)
        z, y = self.head_and_backbone_embeddings_step(img)

        # Collect the xywh for the patches in the validation
        w, h = img.shape[-2:]
        patch_x = torch.tensor([metadata.loc_x for metadata in list_metadata], dtype=z.dtype, device=z.device)
        patch_y = torch.tensor([metadata.loc_y for metadata in list_metadata], dtype=z.dtype, device=z.device)
        patch_w = w * torch.ones_like(patch_x)
        patch_h = h * torch.ones_like(patch_x)
        patches_xywh = torch.stack([patch_x, patch_y, patch_w, patch_h], dim=-1)

        # Create the validation dictionary. Note that all the entries are torch.Tensors
        val_dict = {
            "features_bbone": y,
            "features_head": z,
            "patches_xywh": patches_xywh
        }
        
            
        # Add to this dictionary the things I want to classify and regress
        dict_classify = concatenate_list_of_dict([self.get_metadata_to_classify(metadata)
                                                  for metadata in list_metadata])
        
        for k, v in dict_classify.items():
            val_dict["classify_"+k] = torch.tensor(v, device=self.device)

        dict_regress = concatenate_list_of_dict([self.get_metadata_to_regress(metadata)
                                                 for metadata in list_metadata])
        for k, v in dict_regress.items():
            val_dict["regress_" + k] = torch.tensor(v, device=self.device)

        return val_dict

    def validation_epoch_end(self, list_of_val_dict) -> None:
        """ You can receive a list_of_valdict or, if you have multiple val_datasets a list_of_list_of_valdict """
        print("inside validation epoch end")

        if isinstance(list_of_val_dict[0], dict):
            list_dict = [concatenate_list_of_dict(list_of_val_dict)]
        elif isinstance(list_of_val_dict[0], list):
            list_dict = [concatenate_list_of_dict(tmp_list) for tmp_list in list_of_val_dict]
        else:
            raise Exception("In validation epoch end. I received an unexpected input")

        for loader_idx, total_dict in enumerate(list_dict):
            print("rank {0} dataloader_idx {1}".format(self.global_rank, loader_idx))

            # gather dictionaries from the other processes and flatten the extra dimension.
            world_dict = self.all_gather(total_dict, sync_grads=False)
            all_keys = list(world_dict.keys())
            for key in all_keys:
                if len(world_dict[key].shape) == 1 + len(total_dict[key].shape):
                    world_dict[key] = world_dict[key].flatten(end_dim=1)
            print("done dictionary. rank {0}".format(self.global_rank))

            # DO operations ONLY on rank 0.
            # ADD "rank_zero_only=True" to avoid deadlocks on synchronization.
            if self.global_rank == 0:

                # plot the UMAP colored by all available annotations
                smart_pca = SmartPca(preprocess_strategy='z_score')
                smart_umap = SmartUmap(n_neighbors=25, preprocess_strategy='raw',
                                       n_components=2, min_dist=0.5, metric='euclidean')

                print("starting with embeddings")
                embedding_keys = []
                annotation_keys = []
                umap_keys = []
                all_keys = list(world_dict.keys())
                
                print("all keys:")
                print(all_keys)
                
                for k in all_keys:
                    if k.startswith("feature"):
                        embedding_keys.append(k)
                        # print("working on feature", k)
                        input_features = world_dict[k]
                        embeddings_pca = smart_pca.fit_transform(input_features, n_components=0.95)
                        embeddings_umap = smart_umap.fit_transform(embeddings_pca)
                        world_dict['pca_' + k] = embeddings_pca
                        world_dict['umap_' + k] = embeddings_umap
                        umap_keys.append('umap_' + k)
                    elif k.startswith("regress") or k.startswith("classify"):
                        annotation_keys.append(k)
                print("done with embeddings")

                print("starting to make umaps")
                all_files = []
                
                for umap_key in umap_keys:
                    fig_tmp = plot_embeddings(
                        input_dictionary=world_dict,
                        embedding_key=umap_key,
                        annotation_keys=annotation_keys,
                        n_col=2,
                        sup_title="{0} epoch= {1}".format(umap_key, self.current_epoch)
                    )
                    all_files.append(File.as_image(fig_tmp))
                print("done making umaps")
                
                print(embedding_keys)
                print(annotation_keys)
                print(umap_keys)
                print(all_keys)

                print("starting to log the umaps")
                for file_tmp, key_tmp in zip(all_files, embedding_keys):
                    self.logger.run["maps/" + key_tmp].log(file_tmp)
                print("printed the embeddings")

                if self.run_classify_regress:
                    # knn classification/regression
                    print("starting knn classification/regression")
                    df_mean_knn, df_std_knn = knn_classification_regression(world_dict, self.val_iomin_threshold)
                    # print("df_mean_knn ->", df_mean_knn)

                    for row in df_mean_knn.itertuples():
                        for k, v in row._asdict().items():
                            if isinstance(v, float) and numpy.isfinite(v):
                                name = "kn/" + row.Index + "/" + k + "/mean"
                                self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

                    for row in df_std_knn.itertuples():
                        for k, v in row._asdict().items():
                            if isinstance(v, float) and numpy.isfinite(v):
                                name = "kn/" + row.Index + "/" + k + "/std"
                                self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

                    # linear classification/regression
                    print("starting linear classification/regression")
                    df_mean_linear, df_std_linear = linear_classification_regression(world_dict, self.val_iomin_threshold)
                    # print("df_mean_linear ->", df_mean_linear)

                    for row in df_mean_linear.itertuples():
                        for k, v in row._asdict().items():
                            if isinstance(v, float) and numpy.isfinite(v):
                                name = "linear/" + row.Index + "/" + k + "/mean"
                                self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

                    for row in df_std_linear.itertuples():
                        for k, v in row._asdict().items():
                            if isinstance(v, float) and numpy.isfinite(v):
                                name = "linear/" + row.Index + "/" + k + "/std"
                                self.log(name=name, value=v, batch_size=1, rank_zero_only=True)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        self.neptune_run_id = checkpoint.get("neptune_run_id", None)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """ Loading and resuming is handled automatically. Here I am dealing only with the special variables """
        checkpoint["neptune_run_id"] = getattr(self.logger, "_run_short_id", None)

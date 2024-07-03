Models
======

.. _Mlp Classification and Regression:

Mlp Classification and Regression
---------------------------------
We have implemented a Mlp Classifier and Mlp Regressor with an interface similar
to the one in scikit-learn. Our implementation is much faster than the one in scikit-learn
since it is based on pytorch and runs on GPUs. Moreover, we have reimplemented the method described in the
`Unsupervised Label Noise Modeling and Loss Correction <https://arxiv.org/abs/1904.11238>`_ for classification
with noisy labels. The use-case for the MlpRegressor and MlpClassifier is to check whether the extracted features
can be used to recapitulate biological annotations. For example, can the features be used to classify the
the tissue-condition ("wild-type" vs "disease")? Or, can the features be used to regress the Moran's I score?

.. autoclass:: tissuemosaic.models.classifier_regressor.scikit_learn_interface.BaseEstimator

.. autoclass:: tissuemosaic.models.classifier_regressor.scikit_learn_interface.MlpRegressor
   :show-inheritance:
   :members: fit, predict, score, is_classifier, is_regressor

.. autoclass:: tissuemosaic.models.classifier_regressor.scikit_learn_interface.MlpClassifier
   :show-inheritance:
   :members: fit, predict, score, predict_proba, predict_log_proba, is_classifier, is_regressor


.. _Self Supervised Models:

Self Supervised Models
----------------------
We have implemented multiple self-supervised learning (ssl) models.
All these models ingest image patches. The data augmentation strategy and loss function depends on
the ssl framework chosen. *After training, these models can be used to compute features for new image patches.*
All ssl models inherit from the base class :class:`tissuemosaic.models.ssl_models.SslModelBase` which
is responsible for the validation (which is common to all ssl models) and logging.

.. autoclass:: tissuemosaic.models.ssl_models._ssl_base_model.SslModelBase

.. autoclass:: tissuemosaic.models.ssl_models.barlow.BarlowModel
   :members: add_specific_args, get_default_params

.. autoclass:: tissuemosaic.models.ssl_models.dino.DinoModel
   :members: add_specific_args, get_default_params

.. autoclass:: tissuemosaic.models.ssl_models.simclr.SimclrModel
   :members: add_specific_args, get_default_params

.. autoclass:: tissuemosaic.models.ssl_models.vae.VaeModel
   :members: add_specific_args, get_default_params


.. _Patch Analyzers:

Patch Analyzers
---------------
We have implemented two classes :class:`tissuemosaic.models.patch_analyzer.patch_analyzer.Composition` and
:class:`tissuemosaicmodels.patch_analyzer.patch_analyzer.SpatialAutocorrelation` which can be used to extract
annotations from image patches. Together with the other models described in `Self Supervised Models`_
and `Mlp Classification and Regression`_
these allow to answer interesting questions such as: "Can the patch embedding be used to predict the
cellular-composition of a patch?" or "Can the patch embedding be used to predict the Moran's I score of a patch?".

.. autoclass:: tissuemosaic.models.patch_analyzer.patch_analyzer.Composition
   :members: __call__

.. autoclass:: tissuemosaic.models.patch_analyzer.patch_analyzer.SpatialAutocorrelation
   :members: __call__
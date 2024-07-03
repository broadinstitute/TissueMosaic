Data
====

DataModule
----------
The datamodule encapsulates all the data-related functionalities.
It defines both the pre-processing and data augmentation strategies and
it is ultimately responsible for the definition of the train/test/validation data loaders.
It is a self contained piece of code that ensures reproducibility of all the steps related to
the data manipulation process.

For most users it suffices to use the predefined class
:class:`tissuemosaic.data.datamodule.AnndataFolderDM`. This is the simplest way to
create a datamodule starting from a folder containing anndata objects in `.h5ad` format.
More advanced users can subclass either :class:`tissuemosaic.data.datamodule.SslDM`
or :class:`tissuemosaic.data.datamodule.SparseSslDM` to have extra flexibility.

Our datamodules include the definition of
the cropping strategy (both at train and test time)
and the data-augmentation strategy.
In the :class:`tissuemosaicmodels.ssl_model.dino.DinoModel` self supervised learning framework,
the model is trained using multiple global *and* local crops from each image.
Accordingly the datamodule accounts for the definition of different augmentation for gloabl and local crops.
Other model, such as :class:`tissuemosaic.models.ssl_model.vae.VaeModel`,
:class:`tissuemosaic.models.ssl_model.simclr.SimclrModel` and
:class:`tissuemosaic.models.ssl_model.barlow.BarlowModel` do not use local crops.

.. autoclass:: tissuemosaic.data.datamodule.SslDM

.. autoclass:: tissuemosaic.data.datamodule.SparseSslDM
   :show-inheritance:
   :members: add_specific_args, global_size, local_size, n_global_crops, n_local_crops,
    cropper_test, cropper_train, trsfm_test, trsfm_train_global, trsfm_train_local

.. autoclass:: tissuemosaic.data.datamodule.AnndataFolderDM
   :show-inheritance:
   :members: add_specific_args, ch_in, anndata_to_sparseimage, get_metadata_to_classify, get_metadata_to_regress

SparseImage
-----------
The :class:`SparseImage` is *the most important concept* in the *TissueMosaic* library.
It has easy interoperability with `Anndata <https://anndata.readthedocs.io/en/latest/>`_ which is a
data-structure specifically designed for transcriptomic data.
Contrary to Anndata, which stores the data in the form of a panda Dataframe, :class:`SparseImage` stores the
data in a sparse torch tensor for fast (GPU enabled) processing.

:class:`SparseImage` keeps information at three level of description:
1. the spot-level description. This is similar to Anndata. Cell-level annotations are stored at this level.
2. the patch-level description. For example when an image-patch is processed by a self-supervised learning model
the resulting embedding (which describes property of the entire patch) is stored at this level of description.
3. the image-level description which contains image-level properties.

:class:`SparseImage` provides built-in methods for transferring information between different levels of description.
For example a collection of patch-level properties can be *glued* together to obtain image-level properties
(note that we can deal with overlapping patches) and image-level properties can be evaluated at discrete
location to obtain spot-level properties.

Finally, :class:`SparseImage` provides two methods
:meth:`tissuemosaic.data.sparse_image.SparseImage.compute_ncv` and
:meth:`tissuemosaic.data.sparse_image.SparseImage.compute_patch_features` and to easily extract
information about the cellular micro-environment.


.. autoclass:: tissuemosaic.data.sparse_image.SparseImage
   :members:


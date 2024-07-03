Miscellaneous Utilities
=======================
These are function
Balblabla

.. _dictionary utilities:

Dictionary utilities
--------------------
These are utilities for manipulating dictionaries.
See `notebook4 <https://github.com/broadinstitute/TissueMosaic/blob/main/notebooks/notebook4.ipynb>`_
for an example of why/how to use them.

.. automodule:: tissuemosaic.utils.dict_util
   :members: are_dicts_equal, transfer_annotations_between_dict, inspect_dict,
    subset_dict, subset_dict_non_overlapping_patches,
    flatten_dict, sort_dict_according_to_indices, concatenate_list_of_dict

.. _validation utilities:

Validation utilities
--------------------
These are utilities used during validation to analyze the embeddings.
See `notebook4 <https://github.com/broadinstitute/TissueMosaic/blob/main/notebooks/notebook4.ipynb>`_
for an example of why/how to use them.

.. autoclass:: tissuemosaic.utils.validation_util.SmartUmap
   :members:

.. autoclass:: tissuemosaic.utils.validation_util.SmartLeiden
   :members:

.. autoclass:: tissuemosaic.utils.validation_util.SmartPca
   :members:

.. autoclass:: tissuemosaic.utils.validation_util.SmartScaler
   :members:

.. automodule:: tissuemosaic.utils.validation_util
   :members: get_percentile, inverse_one_hot, get_z_score, compute_distance_embedding

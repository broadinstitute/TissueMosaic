from typing import List, Any
import numpy
import torch
from sklearn.neighbors import KDTree
from .nms_util import NonMaxSuppression


# Set of simple helper functions to manipulate dictionaries


def are_dicts_equal(
        dict1: dict,
        dict2: dict,
        keys_to_include: List[str] = None,
        keys_to_exclude: List[str] = None) -> bool:
    """ Compare two dictionaries. Returns true if all entries are identical

    Args:
        dict1: first dictionary to compare
        dict2: second dictionary to compare
        keys_to_include: list of keys to use for the comparison.
           If None (defaults) the union of the keys in the two dictionary is used.
        keys_to_exclude: list of keys to exclude. If None (defaults) no keys are excluded.

    Returns:
        result: True if all the entries corresponding to :attr:'keys_to_include' are identical.

    Note:
        float(1.0) if considered different from int(1)
    """
    def _equal(v1, v2):
        if type(v1) != type(v2):
            return False
        else:
            bool_tmp = (v1 == v2)
            if isinstance(bool_tmp, torch.Tensor):
                return torch.all(bool_tmp).item()
            elif isinstance(bool_tmp, numpy.ndarray):
                return numpy.all(bool_tmp)
            else:
                return bool_tmp

    assert isinstance(dict1, dict) and isinstance(dict2, dict)

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    if keys_to_include is None:
        keys_to_include = keys1.union(keys2)

    if keys_to_exclude is not None:
        keys_to_include = set(keys_to_include) - set(keys_to_exclude)

    assert keys_to_include.issubset(keys1.union(keys2)), \
        "Error. Some of the keys used in the comparison are not present neither in dict1 nor dict2."

    if keys_to_include.issubset(keys1) ^ keys_to_include.issubset(keys2):
        # same keys are present only in one dict and not the other. The dicts are different
        return False

    # Finally I can loop over all the keys_to_include
    for k in keys_to_include:
        if not _equal(dict1[k], dict2[k]):
            return False
    return True


def transfer_annotations_between_dict(
        source_dict: dict,
        dest_dict: dict,
        annotation_keys: List[Any],
        anchor_key: Any,
        metric: str = 'euclidean',) -> dict:
    """
    Transfer the annotations from the source dictionary to the destination dictionary.
    For each element in the destination dictionary it findis the closests element in the source dictionary and copies
    the annotations from there. Closeness is defined as the metric distance between the anchor_elements.

    Args:
        source_dict: source dictionary from which the annotations will be read
        dest_dict: destination dictionary where the annotation will be written
        annotation_keys: List of keys. It is assumed that these keys are present in the source_dictionary
        anchor_key: The key of the element to be used to measure distances.
            It must be present in BOTH source and destination dictionaries.
        metric: the distance metric to measure distance between elements in the source and destination dictionaries.
            It defaults to 'euclidian'.

    Returns:
        dict: The updated destination dictionary
    """
    assert set(annotation_keys).issubset(set(source_dict.keys())), \
        "Some annotation_keys are missing in the source dictionary. {0} vs {1}".format(set(annotation_keys),
                                                                                       set(source_dict.keys()))
    assert anchor_key in source_dict.keys() and anchor_key in dest_dict.keys(), \
        "The anchor_key need to be present in both the source and destination dictionary"

    def _to_numpy(v):
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        elif isinstance(v, numpy.ndarray):
            return v
        else:
            raise Exception("Expected torch.Tensor or numpy.ndarray but received {0}".format(type(v)))

    def _select_index(values, index):
        if isinstance(values, torch.Tensor):
            return values[index]
        elif isinstance(values, numpy.ndarray):
            return values[index]
        elif isinstance(values, list):
            tmp_numpy = numpy.array(values)
            return tmp_numpy[index].tolist()

    anchors_source = _to_numpy(source_dict[anchor_key])
    anchors_destination = _to_numpy(dest_dict[anchor_key])

    kdt = KDTree(anchors_source, metric=metric)
    dist_kdt, index_kdt = kdt.query(anchors_destination, k=2, return_distance=True)

    # If the annotation were built by aggregating the source_dictionary then the first distance should be exactly ZERO.
    # This is a nice check for debugging.
    # print(dist_kdt)

    # copy the annotation from the nearest neighbour only
    for key in annotation_keys:
        dest_dict[key] = _select_index(values=source_dict[key], index=index_kdt[:, 0])
    return dest_dict


def inspect_dict(d, prefix: str = ''):
    """
    Inspect the content of the dictionary

    Args:
        d: the dictionary to inspect
        prefix: used recursively in case of nested dictionary. Do not set it directly.
    """
    for k, v in d.items():
        if isinstance(v, list):
            print(prefix, k, type(v), len(v))
        elif isinstance(v, torch.Tensor):
            print(prefix, k, type(v), v.shape, v.device)
        elif isinstance(v, numpy.ndarray):
            print(prefix, k, type(v), v.shape)
        elif isinstance(v, dict):
            print(prefix, k, type(v))
            inspect_dict(v, prefix=prefix+"-->")
        else:
            print(prefix, k, type(v))


def subset_dict(input_dict: dict, mask: torch.Tensor):
    """
    Subset all the elements of a dictionary according to a mask

    Args:
        input_dict: dictionary with multiple entries in the form of list, numpy.arrau or torch.Tensors
            with the same leading dimensions, `(N)`
        mask: boolean tensor of shape `(N)`.

    Returns:
        output_dict: a new dictionary with the subset values
    """

    assert mask.dtype == torch.bool
    new_dict = dict()
    for k, v in input_dict.items():
        if isinstance(v, numpy.ndarray):
            new_dict[k] = v[mask.cpu().numpy()]
        elif isinstance(v, list):
            new_dict[k] = numpy.array(v)[mask.cpu().numpy()].tolist()
        elif isinstance(v, torch.Tensor):
            new_dict[k] = v[mask.to(device=v.device)]
    return new_dict


def subset_dict_non_overlapping_patches(
        input_dict: dict,
        key_tissue: str,
        key_patch_xywh: str = "patches_xywh",
        iom_threshold: float = 0.0) -> dict:
    """
    Subset a dictionary containing overlapping patches to a smaller dictionary containing only
    (weakly) overlapping ones.

    Args:
        input_dict: the dictionary to subset.
        key_tissue: the dictionary key corresponding to the tissue identifier.
        key_patch_xywh: the dictionary key corresponding to the coordinates (i.e. x,y,w,h) of the patches.
        iom_threshold: Threshold value for Intersection Over Minimum (IoM).
            If two patches have :math:`\\text{IoM} > \\text{threshold}` only one will survive
            the filtering process.
            Set :attr:'iom_threshold' = 0 to have a collection of strictly non-overlapping patches.

    Returns:
        output_dict: Dictionary containing only patches with overlap less than threshold.

    Note:
        The original dictionary will NOT be overwritten.
    """

    assert key_tissue in input_dict.keys(), \
        "key_tissue = {0} in not present in the input_dictionary.".format(key_tissue)

    assert key_patch_xywh in input_dict.keys(), \
        "key_patch_xywh = {0} in not present in the input_dictionary.".format(key_patch_xywh)

    nms_mask_n, overlap_nn = NonMaxSuppression.compute_nm_mask(
        score=torch.rand_like(torch.Tensor(input_dict[key_patch_xywh][:, 0]).float()),
        ids=input_dict[key_tissue],
        patches_xywh=input_dict[key_patch_xywh],
        iom_threshold=iom_threshold)
    return subset_dict(input_dict=input_dict, mask=nms_mask_n)


def flatten_dict(input_dict: dict, separator: str = '_', prefix: str = ''):
    """ Flatten a (possibly nested) dictionary

    Args:
        input_dict: the input dictionary to flatten
        separator: string used to merge nested keys. It defaults to "_"
        prefix: used in the recursive calls. Do not set manually
    """
    return {prefix + separator + k if prefix else k: v
            for kk, vv in input_dict.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(input_dict, dict) else {prefix: input_dict}


def sort_dict_according_to_indices(input_dict: dict, list_of_indices: List[int]) -> dict:
    """
    Sort dictionaries w.r.t. a list of indices.

    Args:
        input_dict: the dictionary to sort
        list_of_indices: the indices to use in the sorting.

    Returns:
        output_dict: the sorted dictionary.

    Example:
         >>> input_dict = {'key': ['b', 'c', 'a']}
         >>> list_of_indices = [3, 1, 2]
         >>> output_dict = sort_dict_according_to_indices(input_dict, list_of_indices)
         >>> print(output_dict) # will be a,b,c
    """
    sorted_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            # note that I have to do left sorting
            y = torch.zeros_like(v)
            y[list_of_indices] = v
            sorted_dict[k] = y
        elif isinstance(v, numpy.ndarray):
            # note that I have to do left sorting
            y = numpy.zeros_like(v)
            y[list_of_indices] = v
            sorted_dict[k] = y
        elif isinstance(v, list):
            sorted_dict[k] = [x for _, x in sorted(zip(list_of_indices, v), key=lambda pair: pair[0])]
        else:
            raise Exception("Expected tensor or list. Received = {0}".format(type(v)))
    return sorted_dict


def concatenate_list_of_dict(list_of_dict) -> dict:
    """
    Concatenate dictionary with the same set of keys

    Args:
        list_of_dict: list of dictionary to concatenate

    Returns:
        output_dict: the concatenated dictionary
    """
    # check that all dictionaries have the same set of keys
    for i in range(len(list_of_dict)-1):
        keys1 = set(list_of_dict[i].keys())
        keys2 = set(list_of_dict[i+1].keys())
        assert keys1 == keys2, "ERROR, Some dictionary contains different keys: {0} vs {1}".format(keys1, keys2)

    total_dict = {}
    for mydict in list_of_dict:
        for k, v in mydict.items():

            if isinstance(v, list):
                if k in total_dict.keys():
                    total_dict[k] = total_dict[k] + v
                else:
                    total_dict[k] = v

            elif isinstance(v, torch.Tensor):
                if k in total_dict.keys():
                    total_dict[k] = torch.cat((total_dict[k], v), dim=0)
                else:
                    total_dict[k] = v
                    
            elif isinstance(v, numpy.ndarray):
                if k in total_dict.keys():
                    total_dict[k] = numpy.concatenate((total_dict[k], v), axis=0)
                else:
                    total_dict[k] = v

            elif isinstance(v, int) or isinstance(v, float):
                if k in total_dict.keys():
                    total_dict[k] = total_dict[k] + [v]
                else:
                    total_dict[k] = [v]

            else:
                raise Exception("ERROR: Unexpected in concatenate_list_of_dict. \
                Received {0}, {1}".format(type(v), v))

    return total_dict

import torch
import numpy
from tissuemosaic.misc_utils.dict_util import (
    are_dicts_equal)


def test_are_dict_equal():
    dict1 = {'a': 1, 'b': 1.0, 'c': [1, 2, 3], 'd': torch.arange(3), 'e': numpy.arange(3)}
    dict2 = {'a': 1, 'b': 1.0, 'c': [1, 2, 3], 'd': torch.arange(3), 'e': numpy.arange(3)}
    assert are_dicts_equal(dict1, dict2)

    dict1 = {'a': 1, 'b': 1.0, 'c': [1, 2, 3], 'd': torch.arange(3), 'e': numpy.arange(3)}
    dict2 = {'a': 1.0, 'b': 1.0, 'c': [1, 2, 3], 'd': torch.arange(3), 'e': numpy.arange(3)}
    assert not are_dicts_equal(dict1, dict2)

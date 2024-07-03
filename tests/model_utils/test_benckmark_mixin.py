import pytest
import torch
import numpy
import pandas
from tissuemosaic.model_utils.benckmark_mixin import classify_and_regress
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


@pytest.mark.parametrize("n_splits, n_repeats", [(1, 1), (2, 1), (2, 2)])
def test_classify_regress_knn(n_splits, n_repeats, capsys):
    """ Test that knn classifier regressor works """

    n, p = 23, 12
    my_dict = {
        "feature_a": torch.randn((n, p)),
        "feature_b": torch.randn((n, 2*p)),
        "label_a": torch.randint(low=0, high=5, size=[n]),
        "label_b": ["wt" if i < 0.5*n else "dis" for i in range(n)],
        "label_c": torch.zeros(n),
        "regress_a": torch.randn(n),
        "regress_b": numpy.random.randn(n)
    }

    feature_keys, classify_keys, regress_keys = [], [], []
    for k in my_dict.keys():
        if k.startswith("feature"):
            feature_keys.append(k)
        elif k.startswith("label"):
            classify_keys.append(k)
        elif k.startswith("regress"):
            regress_keys.append(k)

    def exclude_self(d):
        w = numpy.ones_like(d)
        w[:, 0] = 0.0
        return w

    kn_kargs = {
        "n_neighbors": 5,
        "weights": exclude_self,
    }

    df_tot = classify_and_regress(
        input_dict=my_dict,
        feature_keys=feature_keys,
        regress_keys=regress_keys,
        classify_keys=classify_keys,
        regressor=KNeighborsRegressor(**kn_kargs),
        classifier=KNeighborsClassifier(**kn_kargs),
        n_splits=n_splits,
        n_repeats=n_repeats)

    df_tot["combined_key"] = df_tot["x_key"] + "_" + df_tot["y_key"]
    assert isinstance(df_tot, pandas.DataFrame)
    assert df_tot.shape[0] == n_repeats * n_splits * len(feature_keys) * (len(regress_keys) + len(classify_keys))

#    with capsys.disabled():
#        # inside this context the stdout will not be captured
#        print(n_repeats, n_splits)
#        df_mean = df_tot.groupby("combined_key").mean()
#        for row in df_mean.itertuples():
#            for k, v in row._asdict().items():
#                if isinstance(v, float) and numpy.isfinite(v):
#                    name = "kn/" + row.Index + "/" + k + "/mean"
#                    print(name, v)
#
#        df_std = df_tot.groupby("combined_key").std()
#        for row in df_std.itertuples():
#            for k, v in row._asdict().items():
#                if isinstance(v, float) and numpy.isfinite(v):
#                    name = "kn/" + row.Index + "/" + k + "/std"
#                    print(name, v)


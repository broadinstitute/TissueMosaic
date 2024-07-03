from anndata import AnnData
import pandas as pd
from scanpy.plotting import embedding
import numpy
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from typing import List


def plot_embeddings(
        input_dictionary: dict,
        embedding_key: str,
        annotation_keys: List[str],
        sup_title: str = None,
        n_col: int = 3,
        cmap: Colormap = 'inferno',
        num_categorical: int = 10,
        **kwargs) -> plt.Figure:
    """
    Takes a dictionary with embeddings and multiple annotations and make a multi-panel figure with each panel showing
    one annotation.

    Args:
        input_dictionary: dictionary with input data
        embedding_key: str corresponding to the embeddings in input_dictionary.
            Embedding have shape (n_sample, latent_dim). Only the first two latent dimensions will be used for plotting.
        annotation_keys: List[str] corresponding to annotations in input_dictionary.
        sup_title: the title (if any) for the figure
        n_col: how many columns to have in the multi-panel figure
        cmap: the color map to use for the continuous variable. The categorical variable will have a different cmap.

    Returns:
        fig: multi-panel figure with the embeddings
    """

    assert set(annotation_keys + [embedding_key]).issubset(input_dictionary.keys()), \
        "Either embeddings or annotation keys are missing from the input dictionary"

    def _is_categorical(_x) -> bool:
        is_float = (
                isinstance(_x[0], float) or
                isinstance(_x[0], numpy.float16) or
                isinstance(_x[0], numpy.float32) or
                isinstance(_x[0], numpy.float64)
        )
        is_many = (_x.shape[0] > num_categorical) ## change this to a parameter
        is_continuous = (is_many and is_float)
        is_categorical = not is_continuous
        return is_categorical

    # make a copy of the dict with the torch to numpy conversion
    cloned_dict = {}
    for k, v in input_dictionary.items():
        if isinstance(v, torch.Tensor):
            cloned_dict[k] = v.detach().cpu().numpy()
        elif isinstance(v, list):
            cloned_dict[k] = numpy.array(v)
        elif isinstance(v, numpy.ndarray):
            cloned_dict[k] = v

    # create dataframe with annotations
    df = pd.DataFrame(cloned_dict, columns=annotation_keys)
    for k in annotation_keys:
        vec = numpy.unique(df[k].to_numpy())
        if _is_categorical(vec):
            df[k] = df[k].astype("category")

    # create anndata with annotations and embeddings
    adata = AnnData(obs=df)
    
    adata.obsm[embedding_key] = cloned_dict[embedding_key]

    # leverage anndata embedding capabilities
    fig = embedding(adata,
                    basis=embedding_key,
                    color=annotation_keys,
                    return_fig=True,
                    show=False,
                    ncols=n_col,
                    cmap=cmap,
                    **kwargs)
    if sup_title:
        _ = fig.suptitle(sup_title)

    # close figure and return
    plt.close(fig)
    return fig

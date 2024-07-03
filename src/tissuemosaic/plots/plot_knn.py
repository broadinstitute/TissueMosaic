import torch
import matplotlib
from matplotlib import pyplot as plt
from typing import Tuple, Union, List
from tissuemosaic.utils.validation_util import compute_distance_embedding
from .plot_images import _show_raw_all_channels, _get_color_tensor


def plot_knn_examples(
        input_dict: dict,
        embedding_key: str,
        image_key: str,
        n_neighbors: int = 3,
        examples: Union[int, List[int]] = 6,
        metric: str = "euclidean",
        cmap: "matplotlib.colors.ListedColormap" = plt.cm.viridis,
        figsize: Tuple[int, int] = None,
        plot_histogram: bool = False,
        max_distance: float = None,
        **kargs):
    """
    NEED TO BE REARRANGED b/c INTERFACE HAS CHANGED

    Args:
        input_dict: dictionary with the images of the crops and their embeddings
        embedding_key: str, key corresponding to the embeddings in the input_dict
            (or distance_matrix if :attr:'metric' == 'precomputed')
        image_key: str, key corresponding to the images in the input_dict
        n_neighbors: int, how many nearest neighbour to plot
        examples: either a list of int with the index of the example whose nearest neighbour to show or an int.
            If an int, the example will be selected at random.
        metric: which metric to use. Either "euclidean", "cosine" or "contrastive"
        cmap: the colormap to use for plotting
        figsize: size of the image. Default to None
        plot_histogram: if true, the histogram of distances is shown
        max_distance: float, if given distances larger than this value will be excluded from the histogram.
        kargs: additional parameter which will be passed to matplotlib.hist() function.
            For example 'bins=50', 'density=True'.

    Return:
        a matplotlib figure with example fo knn neighbours.
    """
    # TODO: need to be rearranged b/c interface has changed
    raise NotImplementedError

    assert embedding_key in input_dict.keys(), "Embedding_key = {0} is not present in input_dict.".format(embedding_key)
    assert image_key in input_dict.keys(), "Image_key = {0} is not present in input_dict.".format(image_key)
    assert metric == "euclidean" or "metric" == 'cosine'
    assert len(input_dict[embedding_key]) == len(input_dict[image_key])

    embeddings = input_dict[embedding_key]

    # get some random samples
    if isinstance(examples, int):
        num_examples = examples
        samples_idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:num_examples].long()
    elif isinstance(examples, list) and isinstance(examples[0], int):
        num_examples = len(examples)
        samples_idx = torch.tensor(examples).to(embeddings.device).long()
    else:
        raise Exception("examples is neither a int nor a list of int")

    # compute pairwise-distances using broadcasting
    distances = compute_distance_embedding(
        ref_embeddings=embeddings[samples_idx],
        other_embeddings=embeddings,
        metric=metric)

    distances_nn, index_nn = torch.topk(distances, k=n_neighbors, largest=False, sorted=True, dim=-1)

    # get the colors
    if input_dict[image_key].shape[-3] == 3:
        # if channels == 3 does not need to multiply by color tensor
        colors = None
    else:
        ch = input_dict[image_key].shape[-3]
        if cmap is None:
            colors = _get_color_tensor('tab20', ch)
        else:
            colors = _get_color_tensor(cmap, ch)
        colors = colors.to(input_dict[image_key].device).float()

    # make the images 3 channels
    tensors = input_dict[image_key][index_nn]  # torch.Size([n_examples, n_neighbours, ch, w, h])
    if colors is None:
        imgs = tensors.detach().clone().cpu()
    else:
        imgs = torch.einsum('...cwh,cj->...jwh', tensors, colors).detach().clone().cpu()

    # plot the randomly picked examples and their neighbours
    nrow = num_examples
    ncol = n_neighbors + 1 if plot_histogram else n_neighbors
    figsize = (4 * ncol, 4 * nrow) if figsize is None else figsize
    fig, ax = plt.subplots(ncols=ncol, nrows=nrow, figsize=figsize)
    for r in range(num_examples):
        for c in range(n_neighbors):
            ax_curr = ax[r, c]
            dist = distances_nn[r, c]
            _ = _show_raw_all_channels(
                tensor=imgs[r, c],
                ax=ax_curr,
                title="dist = {0:.4}".format(dist))
            ax_curr.set_axis_off()

        if plot_histogram:
            ax_curr = ax[r, -1]
            dist_tmp = distances[r].flatten().detach()
            if max_distance is not None:
                dist_hist = dist_tmp[dist_tmp < max_distance].cpu().numpy()
            else:
                dist_hist = dist_tmp.cpu().numpy()
            ax_curr.hist(dist_hist, **kargs)
            ax_curr.set_title("Hist distances")

    fig.tight_layout()
    plt.close(fig)
    return fig

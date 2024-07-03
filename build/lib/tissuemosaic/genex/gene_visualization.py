import numpy
import torch
import matplotlib.pyplot as plt


def plot_gene_hist(cell_types_n, value1_ng, value2_ng=None, bins=20) -> plt.Figure:
    """
    Plot the per cell-type histogram. If :attr:`value2_ng` is defined the two histogram are interlieved.

    Args:
        cell_types_n: tensor of shape N with the cell type labels (with K distinct values)
        value1_ng: the first quantity to whose histogram is computed lot of shape (N,G)
        value2_ng: the second quantity to plot of shape (N,G) (optional)
        bins: number of bins in the histogram

    Returns:
        fig: A figure with G rows and K columns where K is the number of distinct cell types.
    """
    def _to_torch(_x):
        if isinstance(_x, torch.Tensor):
            return _x
        elif isinstance(_x, numpy.ndarray):
            return torch.tensor(_x)
        else:
            raise Exception("Expected torch.tensor or numpy.ndarray. Received {0}".format(type(_x)))

    # TODO: use seaborn instead
    #   fig, ax = plt.subplots(figsize=(12, 6))
    #   _ = seaborn.histplot(data=df, x="eps", hue="cell_type", bins=200, ax=ax, multiple="dodge")

    raise NotImplementedError
    value2_ng = value1_ng if value2_ng is None else value2_ng
    value1_torch_ng = _to_torch(value1_ng)
    value2_torch_ng = value1_torch_ng if value2_ng is None else _to_torch(value1_ng)

    assert len(cell_types_n.shape) == 1
    assert len(value1_ng.shape) >= 2
    assert cell_types_n.shape[0] == value1_ng.shape[-2]
    assert value1_torch_ng.shape == value2_torch_ng.shape
    assert value1_torch_ng.dtype == value2_torch_ng.dtype

    ctypes = torch.unique(cell_types_n)
    genes = value1_ng.shape[-1]

    nrows = genes
    ncols = len(ctypes)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4 * ncols, 4 * nrows))

    for r in range(genes):
        for c, c_type in enumerate(ctypes):
            mask = (cell_types_n == c_type)
            if r == 0:
                _ = axes[r, c].set_title("cell_type {0}".format(c_type))

            if value2_ng is None:
                v1 = value1_torch_ng[mask][r]
                if v1.dtype == torch.long:
                    y1 = torch.bincount(v1)
                    x1 = torch.arange(y1.shape[0])
                    barWidth = 0.9 * (x1[1] - x1[0])
                    _ = axes[r, c].bar(x1, y1, width=barWidth)
                else:
                    y1, x1 = numpy.histogram(v1, bins=bins, density=True)
                    barWidth = 0.9 * (x1[1] - x1[0])
                    _ = axes[r, c].bar(x1[:-1], y1, width=barWidth)

            else:
                v1 = value1_torch_ng[mask][r]
                v2 = value2_torch_ng[mask][r]
                a1, b1 = v1.min(), v1.max()
                a2, b2 = v2.min(), v2.max()
                a = min(a1.item(), a2.item())
                b = max(b1.item(), b2.item(), 1)
                myrange = (a, b+1)

                if v1.dtype == torch.long:
                    y1 = torch.bincount(v1, minlength=int(myrange[1]))
                    y2 = torch.bincount(v2, minlength=int(myrange[1]))
                    x = torch.arange(y1.shape[0])
                    barWidth = 0.4 * (x[1] - x[0])
                    _ = axes[r, c].bar(x, y1, width=barWidth)
                    _ = axes[r, c].bar(x + barWidth, y2, width=barWidth)
                else:
                    y1, x1 = numpy.histogram(v1, range=myrange, bins=bins, density=True)
                    y2, x2 = numpy.histogram(v2, range=myrange, bins=bins, density=True)
                    barWidth = 0.4 * (x1[1] - x1[0])
                    _ = axes[r, c].bar(x1[:-1], y1, width=barWidth)
                    _ = axes[r, c].bar(x2[:-1] + barWidth, y2, width=barWidth)

    plt.close()
    return fig


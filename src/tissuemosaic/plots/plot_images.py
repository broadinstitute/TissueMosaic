from typing import Optional, Tuple, List, Union

import numpy
import torch
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torchvision.transforms import CenterCrop as trsfm_center_crop
from torchvision.transforms import Compose as trsfm_compose
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def _get_color_tensor(_cmap, _ch):
    cm = plt.get_cmap(_cmap, _ch)
    x = numpy.linspace(0.0, 1.0, _ch)
    colors_np = cm(x)
    color = torch.Tensor(colors_np)[:, :3]
    assert color.shape[0] == _ch
    return color


def _minmax_scale_tensor(tensor, in_range: Tuple[float, float] = None):
    """ Clamp tensor in_range and transform to (0,1) range """
    if in_range is None:
        in_range_min, in_range_max = torch.min(tensor), torch.max(tensor)
    else:
        in_range_min, in_range_max = in_range

    dist = in_range_max - in_range_min
    scale = 1.0 if dist == 0.0 else 1.0 / dist
    return tensor.clamp(min=in_range_min,
                        max=in_range_max).add_(other=in_range_min,
                                               alpha=-1.0).mul_(other=scale).clamp_(min=0.0, max=1.0)


def pad_and_crop_and_stack(x: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """
    Takes a list of tensor and returns a single batched_tensor. It is useful for visualization.

    Args:
        x: a list of tensor with the same channel dimension but possibly different width and heigth
        pad_value: float, the value used in padding the images. Defaults to padding with black colors

    Returns:
        tensor: A single batch tensor of shape :math:`(B, c, \\text{width}_\\text{max}, \\text{height}_\\text{max})`
    """
    widths = [tmp.shape[-2] for tmp in x]
    heigths = [tmp.shape[-1] for tmp in x]

    w_min = min(widths)
    w_max = max(widths)
    h_min = min(heigths)
    h_max = max(heigths)

    pad_w = w_max - w_min
    pad_h = h_max - h_min

    padder = torch.nn.ConstantPad2d((pad_w, pad_w, pad_h, pad_h), value=pad_value)
    cropper = trsfm_center_crop(size=(w_max, h_max))
    transform = trsfm_compose([padder, cropper])

    imgs_batched = torch.stack([transform(tmp) for tmp in x], dim=0)
    return imgs_batched


def show_batch(
        tensor: torch.Tensor,
        cmap: str = None,
        n_col: int = 4,
        n_padding: int = 10,
        title: str = None,
        pad_value: int = 1,
        normalize: bool = True,
        normalize_range: Tuple[float, float] = None,
        figsize: Tuple[float, float] = None) -> plt.Figure:
    """
    Visualize a torch tensor of shape: :math:`(*,  \\text{ch}, \\text{width}, \\text{height})`.
    It works for any number of leading dimensions.

    Args:
        tensor: the torch.Tensor to plot
        cmap: the color map to use. If None, it defaults to 'gray' for 1 channel images, RGB for 3 channels images and
            'tab20' for images with more that 3 channels.
        n_col: int, number of columns in the image grid
        n_padding: int, padding between images in the grid
        title: str, the tile on the image
        pad_value: float, pad_value
        normalize: bool, if tru normalize the tensor in normalize_range
        normalize_range: tuple, if not specified it is set to (min_image, max_image)
        figsize: size of the figure
    """
    assert len(tensor.shape) >= 4  # *, ch, width, height
    tensor = tensor.flatten(end_dim=-4)  # -1, ch, width, height
    ch = tensor.shape[-3]

    if ch > 3:
        cmap = 'tab20' if cmap is None else cmap
        colors = _get_color_tensor(cmap, ch)
        images = torch.einsum('...cwh,cj->...jwh', tensor, colors.to(device=tensor.device).float())
    elif ch == 3:
        images = tensor
    else:
        images = tensor[..., :1, :, :]

    images = images.cpu().to(dtype=torch.float32)  # upgrade to full precision if working in half precision.
    n_images = images.shape[-4]
    n_row = int(numpy.ceil(float(n_images) / n_col))

    # Always normalize the image in (0,1) either using min_max of tensor or normalize_range
    grid = make_grid(images, n_col, n_padding, normalize=normalize, value_range=normalize_range,
                     scale_each=False, pad_value=pad_value)

    figsize = (4 * n_col, 4 * n_row) if figsize is None else figsize

    fig = plt.figure(figsize=figsize)
    plt.imshow(grid.detach().permute(1, 2, 0).squeeze(-1).numpy())
    # plt.axis("off")
    if isinstance(title, str):
        plt.title(title)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _show_raw_one_channel(
        tensor: torch.Tensor,
        ax: "matplotlib.axes.Axes",
        cmap: str,
        in_range: Union[str, Tuple[float, float]] = 'image',
        title: Optional[str] = None):

    # normalization
    if in_range == 'image':
        tensor = _minmax_scale_tensor(tensor, in_range=(torch.min(tensor).item(), torch.max(tensor).item()))
    else:
        tensor = _minmax_scale_tensor(tensor, in_range=in_range)

    _ = ax.imshow(numpy.asarray(to_pil_image(tensor)), cmap=cmap)
    if title is not None:
        ax.set_title(title)


def _show_raw_all_channels(
        tensor: torch.Tensor,
        ax: "matplotlib.axes.Axes",
        title: Optional[str] = None):

    assert len(tensor.shape) == 3

    # normalization
    tensor = _minmax_scale_tensor(tensor, in_range=(torch.min(tensor).item(), torch.max(tensor).item()))

    _ = ax.imshow(numpy.asarray(to_pil_image(tensor)))
    if title is not None:
        ax.set_title(title)


def show_raw_one_channel(
        data: Union[torch.Tensor, List[torch.Tensor]],
        n_col: int = 4,
        cmap: str = None,
        in_range: Union[str, Tuple[float, float]] = 'image',
        scale_each: bool = True,
        figsize: Tuple[float, float] = None,
        titles: List[str] = None,
        sup_title: str = None,
        show_axis: bool = True,
        seed: int=None) -> plt.Figure:
    """
    Visualize a torch tensor of shape :math:`(*, \\text{width}, \\text{height})`.
    or a list of tensor of shape :math:`(\\text{width}, \\text{height})`.
    Each leading dimension is shown separately.

    Args:
        data: A torch.tensor of shape :math:`(*, \\text{width}, \\text{height})` or list
            of tensor of shape :math:`(\\text{width}, \\text{height})`.
        n_col: number of columns to plot the data
        cmap: the matplotlib color map to use. If None use 'gray' colormap.
        in_range: Either a tuple specifying a a min and max value (for clamping) or a string 'image'.
            If 'image' the min and max value are computed form the image itself.
            Value are clamped in_range and then transformed to range (0.0, 1.0) for visualization.
        scale_each: bool if true each leading dimension is scaled by itself. It has effect only if in_range = 'image'
        figsize: Optional, the tuple with the width and height of the rendered figure
        titles: list with the titles for each small image
        sup_title: str, the title for the entire image
        show_axis: If True (defaults) show the axis.

    Returns:
        fig: A figure with `(*)` panels. Each panel is a rendering of a tensor of
            shape :math:`(\\text{width}, \\text{height})`
    """

    if isinstance(data, list):
        tmp = [len(tensor.shape) == 2 for tensor in data]
        assert all(tmp)
        n_max = len(data)
        data = [tmp.float() for tmp in data]
        if in_range == 'image' and not scale_each:
            mins = [torch.min(tensor) for tensor in data]
            maxs = [torch.max(tensor) for tensor in data]
            in_range = min(mins), max(maxs)

    elif isinstance(data, torch.Tensor):
        data = data.flatten(end_dim=-3).float()  # shape: (*, w, h)
        n_max = data.shape[0]
        if in_range == 'image' and not scale_each:
            in_range = torch.min(data), torch.max(data)
    else:
        raise Exception("Expected Union[tensor, List[tensor]]. Received {0}".format(type(data)))

    assert titles is None or (isinstance(titles, list) and len(titles) == n_max)
    ncols = min(n_col, n_max)
    nrows = int(numpy.ceil(n_max / ncols))
    figsize = (4*ncols, 4*nrows) if figsize is None else figsize

    if nrows == 1:
        fig, ax = plt.subplots(ncols=ncols, figsize=figsize)
    else:
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    n = 0
    for r in range(nrows):
        for c in range(ncols):
            if nrows == 1 and ncols == 1:
                ax_curr = ax
            elif nrows == 1:
                ax_curr = ax[c]
            else:
                ax_curr = ax[r, c]

            if n < n_max:
                _show_raw_one_channel(
                    tensor=data[n],
                    ax=ax_curr,
                    cmap='gray' if cmap is None else cmap,
                    in_range=in_range,
                    title=None if titles is None else titles[n],
                )
                n += 1
                if show_axis:
                    ax_curr.set_axis_on()
                else:
                    ax_curr.set_axis_off()
            else:
                ax_curr.set_axis_off()

    if sup_title:
        fig.suptitle(sup_title)

    plt.close(fig)
    return fig


def show_raw_all_channels( 
        data: Union[torch.Tensor, List[torch.Tensor]],
        n_col: int = 4,
        cmap: str = None,
        figsize: Tuple[float, float] = None,
        titles: List[str] = None,
        sup_title: str = None,
        show_colorbar: Optional[bool] = None,
        legend_colorbar: List[str] = None,
        show_axis: bool = True,
        show_ticks: bool = True,
        col_titles: List[str] = None,
        row_titles: List[str] = None,
        fontsize: int = 20,
        labelpad: int = 10,
        scalebar_size: int = None,
        scale_factor: float = 0.65,
        pixel_size: float = 4.0) -> plt.Figure:
    """
    Visualize a torch tensor of shape: :math:`(*, \\text{ch}, \\text{width}, \\text{height})`
    or a list of tensors of shape :math:`(\\text{ch}, \\text{width}, \\text{height})`.

    Args:
        data: A torch.tensor of shape :math:`(*, \\text{ch}, \\text{width}, \\text{height})`
            or a list of tensor of shape :math:`(\\text{ch}, \\text{width}, \\text{height})`
        n_col: number of columns to plot the data
        cmap: the matplotlib color map to use. Defaults to RBG (if data has 3 channels) or 'tab20' otherwise.
        figsize: Optional, the tuple with the width and height of the rendered figure
        titles: list with the titles for each panel
        sup_title: the title for the entire image
        show_colorbar: bool, if yes show the color bar
        legend_colorbar: legend for the colorbar
        show_axis: If True (default) show the axis

    Returns:
        fig: A figure with `*` panels each panel is a rendering of a tensors of shape
            :math:`(\\text{ch}, \\text{width}, \\text{height})`
    """
    if isinstance(data, torch.Tensor) and len(data.shape) == 3:
        data = data.unsqueeze(dim=0)

    if isinstance(data, torch.Tensor):
        data = data.detach().clone().cpu().float()
    elif isinstance(data, list):
        data = [tmp.detach().clone().cpu().float() for tmp in data]
    else:
        raise Exception("Expected either a tensor or a list of tensors. Received {0}".format(type(data)))

    # check the images have all the same channels
    chs = [data[n].shape[-3] for n in range(0, len(data))]
    check = [ch == chs[0] for ch in chs]
    assert all(check), "The images have different number of channels {0}".format(chs)

    # extract the channels and the colors to use
    ch = chs[0]
    assert legend_colorbar is None or len(legend_colorbar) == ch
    if cmap is None:
        if ch == 3:
            colors = torch.eye(3)
        else:
            colors = _get_color_tensor('tab20', ch)
    else:
        colors = _get_color_tensor(cmap, ch)
    colors = colors.to(data[0].device).float()

    if isinstance(data, torch.Tensor):
        imgs = torch.einsum('...cwh,cj->...jwh', data, colors).detach().clone().cpu()
    else:
        imgs = [torch.einsum('...cwh,cj->...jwh', img, colors).detach().clone().cpu() for img in data]

    # set the canvas
    n_max = imgs.shape[0] if isinstance(data, torch.Tensor) else len(data)
    ncols = min(n_col, n_max)
    nrows = int(numpy.ceil(n_max / ncols))
    figsize = (4*ncols, 4*nrows) if figsize is None else figsize

    assert titles is None or (isinstance(titles, list) and len(titles) == n_max)

    if nrows == 1:
        fig, axes = plt.subplots(ncols=ncols, figsize=figsize)
    else:
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    n = 0
    for r in range(nrows):
        for c in range(ncols):
            if nrows == 1 and ncols == 1:
                ax_curr = axes
            elif nrows == 1:
                ax_curr = axes[c]
            else:
                ax_curr = axes[r, c]

            if n < n_max:
                _show_raw_all_channels(
                    tensor=imgs[n],
                    ax=ax_curr,
                    title=None if titles is None else titles[n],
                )
                n += 1
                if show_axis:
                    ax_curr.set_axis_on()
                else:
                    ax_curr.set_axis_off()
                if not show_ticks:
                    ax_curr.set_xticks([])
                    ax_curr.set_yticks([])
                    # ax_curr.tick_params(left = False, bottom = False)
            else:
                ax_curr.set_axis_off()

    if sup_title:
        fig.suptitle(sup_title)

    show_colorbar = legend_colorbar is not None if show_colorbar is None else show_colorbar
    if show_colorbar:
        discrete_cmp = ListedColormap(colors.numpy())
        normalizer = matplotlib.colors.BoundaryNorm(
            boundaries=numpy.linspace(-0.5, ch - 0.5, ch + 1),
            ncolors=ch,
            clip=True)

        scalar_mappable = matplotlib.cm.ScalarMappable(norm=normalizer, cmap=discrete_cmp)
        cbar = fig.colorbar(scalar_mappable, ticks=numpy.arange(ch), ax=axes)

        if legend_colorbar is None:
            legend_colorbar = numpy.arange(ch).tolist()
        cbar.ax.set_yticklabels(legend_colorbar)

    if col_titles is not None:
        assert ncols == len(col_titles) 
        for c in range(ncols):
            axes[0, c].set_title(col_titles[c], fontsize=fontsize, pad=labelpad)

    if row_titles is not None:
        assert nrows == len(row_titles) 
        for r in range(ncols):
            axes[r, 0].set_ylabel(row_titles[r], fontsize=fontsize, labelpad=labelpad)

    if scalebar_size is not None:
        for c in range(ncols):
            scalebar_len = 1/scale_factor * scalebar_size / pixel_size
            scalebar = AnchoredSizeBar(axes[0, c].transData,
                            scalebar_len, '', 'lower left', 
                            pad=0.1,
                            color='white',
                            frameon=False,
                            size_vertical=2)
            axes[0, c].add_artist(scalebar)    

    plt.close(fig)
    return fig

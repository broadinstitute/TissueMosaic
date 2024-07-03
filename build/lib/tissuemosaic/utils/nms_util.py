import torch
import numpy
from typing import Union, List, Any


class NonMaxSuppression:
    """
    Given a set of bounding box defined over possibly different tissue
    Use Intersection_over_Minimum criteria to filter out overlapping proposals.
    """

    @staticmethod
    @torch.no_grad()
    def compute_nm_mask(score:  Union[torch.Tensor, numpy.ndarray],
                        ids: Union[torch.Tensor, numpy.ndarray, List[Any]],
                        patches_xywh:  Union[torch.Tensor, numpy.ndarray],
                        iom_threshold: float) -> (torch.Tensor, torch.Tensor):
        """
        Filter the proposals according to their score and their Intersection over Minimum.

        Args:
            score: score used to sort the proposals of shape (N)
            ids: vector or list of shape (N) with the (tissue) id.
                IoMIN is always zero between patches with different (tissue) ids.
            patches_xywh: coordinates with the proposals of shape (N, 4) where 4 stand for x,y,w,h.
            iom_threshold: threshold of Intersection over Minimum. If IoM is larger than this value the proposals
                will be suppressed during NMS. Only the proposal with larger score will survive.

        Returns:
            (nms_mask_n, iomin_nn) where nms_mask_n is a boolean tensor of shape (N) with True
            if the proposal survived NMS and iomin_nn with the value of the IoMIN among all possible pairs.
        """

        def _to_numpy(_x):
            if isinstance(_x, torch.Tensor):
                return _x.detach().cpu().numpy()
            elif isinstance(_x, numpy.ndarray):
                return _x
            elif isinstance(_x, list):
                return numpy.array(_x)

        def _to_torch(_x):
            if isinstance(_x, torch.Tensor):
                return _x
            elif isinstance(_x, numpy.ndarray):
                return torch.from_numpy(_x)
            else:
                raise Exception("Expected a torch.tensor or a numpy.ndarray. Received {0}".format(type(_x)))

        # the tissue ids can be a list of string. Therefore I can not convert to torch tensor directly.
        ids_numpy = _to_numpy(ids)
        assert len(patches_xywh.shape) == 2 and patches_xywh.shape[-1] == 4
        assert score.shape == ids_numpy.shape == patches_xywh[:, 0].shape

        # this is O(N^2) algorithm (all boxes compared to all other boxes) but it is very simple
        x, y, w, h = _to_torch(patches_xywh).unbind(dim=-1)
        overlap_measure_tmp_nn = NonMaxSuppression._compute_iomin(x=x, y=y, w=w, h=h)

        mask_same_id_nn_numpy = (ids_numpy == ids_numpy[:, None])
        mask_same_id_nn = _to_torch(mask_same_id_nn_numpy).to(device=overlap_measure_tmp_nn.device)
        overlap_measure_nn = overlap_measure_tmp_nn * mask_same_id_nn  # if ids are different IoMIN = 0

        binarized_overlap_nn = (overlap_measure_nn > iom_threshold).float()
        nms_mask_n = NonMaxSuppression.perform_nms_selection(mask_overlap_nn=binarized_overlap_nn,
                                                              score_n=score,
                                                              possible_n=torch.ones_like(score).bool())
        return nms_mask_n, overlap_measure_nn

    @staticmethod
    def perform_nms_selection(mask_overlap_nn: torch.Tensor,
                              score_n: torch.Tensor,
                              possible_n: torch.Tensor) -> torch.Tensor:
        """
        Given a set of n proposals and the (n x n) binarized mask which describes if two proposals are
        mutually exclusive it performs the greedy NMS in parallel (if possible).

        Args:
        mask_overlap_nn: Binarized overlap matrix with 1 if IoMIN > threshold and 0 otherwise, i.e 1 means that
            two proposals are incompatible, 0 means that they are compatible.
        score_n: Score of the proposal. Higher score proposal have precedence.
        possible_n: Vector with 1 if the proposal can be chosen and 0 otherwise.

        Note:
            The algorithm terminates when there are no more suitable proposals
            (because they have all been suppressed by higher scoring ones).

        Returns:
             mask_nms_n: A tensor with the same shape as :attr:'score_n'. The entries are 1 if that proposal
             has been selected (i.e. survived NMS) and 0 otherwise.
        """
        # reshape
        score_1n = score_n.unsqueeze(-2)
        possible_1n = possible_n.unsqueeze(-2)
        idx_n1 = torch.arange(start=0, end=score_n.shape[-1], step=1, device=score_n.device).view(-1, 1).long()
        selected_n1 = torch.zeros_like(score_n).unsqueeze(dim=-1)

        # Greedy algorithm in a loop
        n_iter = 0
        while possible_1n.sum() > 0:
            n_iter += 1
            score_mask_nn = mask_overlap_nn * (score_1n * possible_1n)
            index_n1 = torch.max(score_mask_nn, keepdim=True, dim=-1)[1]
            selected_n1 += possible_1n.transpose(dim0=-1, dim1=-2) * (idx_n1 == index_n1)
            blocks_1n = torch.sum(mask_overlap_nn * selected_n1, keepdim=True, dim=-2)
            possible_1n *= (blocks_1n == 0)
        mask_selected_n = selected_n1.squeeze(dim=-1).bool()
        # print("DEBUG nms performed in ", n_iter)
        # print("DEBUG nms. Mask ", mask_selected_n.shape, mask_selected_n.sum(), mask_selected_n.dtype)
        return mask_selected_n

    @staticmethod
    def _unroll_and_compare(x_tmp: torch.Tensor, label: str) -> torch.Tensor:
        """ Given a vector of size: (*, n) creates an output of size (*, n, n)
            obtained by comparing all vector entries with all other vector entries
            The comparison is either: MIN,MAX """
        if label == "MAX":
            y_tmp = torch.max(x_tmp.unsqueeze(dim=-1), x_tmp.unsqueeze(dim=-2))
        elif label == "MIN":
            y_tmp = torch.min(x_tmp.unsqueeze(dim=-1), x_tmp.unsqueeze(dim=-2))
        else:
            raise Exception("label is unknown. It is ", label)
        return y_tmp

    @staticmethod
    def _compute_iomin(
            x: torch.Tensor,
            y: torch.Tensor,
            w: torch.Tensor,
            h: torch.Tensor) -> torch.Tensor:
        """
        Given x,y,w,h compute the Intersection over Min Area (IoMin) among all possible pairs.

        Args:
            x: torch.Tensor of shape: (n) with the x-coordinate
            y: torch.Tensor of shape: (n) with the y-coordinate
            w: torch.Tensor of shape: (n) with the width
            h: torch.Tensor of shape: (n) with the height

        Returns:
            A matrix of shape (n, n) with the IoMIN
        """

        assert x.shape == y.shape == w.shape == h.shape

        # compute x1,x3,y1,y3 and area
        x1 = x
        x3 = x + w
        y1 = y
        y3 = y + h
        area = w * h

        min_area_nn = NonMaxSuppression._unroll_and_compare(area, "MIN")
        xi1_nn = NonMaxSuppression._unroll_and_compare(x1, "MAX")
        yi1_nn = NonMaxSuppression._unroll_and_compare(y1, "MAX")
        xi3_nn = NonMaxSuppression._unroll_and_compare(x3, "MIN")
        yi3_nn = NonMaxSuppression._unroll_and_compare(y3, "MIN")

        intersection_area_nn = torch.clamp(xi3_nn - xi1_nn, min=0) * torch.clamp(yi3_nn - yi1_nn, min=0)
        return intersection_area_nn / min_area_nn

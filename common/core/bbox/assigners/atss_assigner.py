import warnings
import mindspore as ms
from mindspore import ops, nn, Tensor

from common.core.bbox.iou2d_calculator import BboxOverlaps2D


class ATSSAssigner(nn.Cell):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    If ``alpha`` is not None, it means that the dynamic cost
    ATSSAssigner is adopted, which is currently only used in the DDOD.

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 alpha=None,
                 ignore_iof_thr=-1):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.iou_calculator = BboxOverlaps2D()
        self.ignore_iof_thr = ignore_iof_thr

    """Assign a corresponding gt bbox or background to each bbox.

    Args:
        topk (int): number of bbox selected in each level.
        alpha (float): param of cost rate for each proposal only in DDOD.
            Default None.
        iou_calculator (dict): builder of IoU calculator.
            Default dict(type='BboxOverlaps2D').
        ignore_iof_thr (int): whether ignore max overlaps or not.
            Default -1 (1 or -1).
    """

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def construct(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               gt_valids=None,
               cls_scores=None,
               bbox_preds=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        If ``alpha`` is not None, and ``cls_scores`` and `bbox_preds`
        are not None, the overlaps calculation in the first step
        will also include dynamic cost, which is currently only used in
        the DDOD.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO. Default None.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes. Default None.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4. Default None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner,  please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time. '

        # if self.alpha is None:
            # ATSSAssigner
        overlaps = self.iou_calculator(bboxes, gt_bboxes)
            # if cls_scores is not None or bbox_preds is not None:
            #     warnings.warn(message)
        # else:
        #     # Dynamic cost ATSSAssigner in DDOD
        #     assert cls_scores is not None and bbox_preds is not None, message
        #
        #     # compute cls cost for bbox and GT
        #     cls_cost = ops.sigmoid(cls_scores[:, gt_labels])
        #
        #     # compute iou between all bbox and gt
        #     overlaps = self.iou_calculator(bbox_preds, gt_bboxes)
        #
        #     # make sure that we are in element-wise multiplication
        #     assert cls_cost.shape == overlaps.shape
        #
        #     # overlaps is actually a cost matrix
        #     overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

        # assign 0 by default
        assigned_gt_inds = ops.full((num_bboxes + 1, ),
                                             num_gt,
                                             dtype=ms.int64)
        assigned_gt_inds = ops.stop_gradient(assigned_gt_inds)

        # if num_gt == 0 or num_bboxes == 0:
        #     # No ground truth or boxes, return empty assignment
        #     max_overlaps = overlaps.new_zeros((num_bboxes, ))
        #     if num_gt == 0:
        #         # No truth, assign everything to background
        #         assigned_gt_inds[:] = 0
        #     if gt_labels is None:
        #         assigned_labels = None
        #     else:
        #         assigned_labels = ops.full((num_bboxes,),
        #                                     -1,
        #                                     dtype=ms.int64)
        #         assigned_labels = ops.stop_gradient(assigned_labels)
        #     return AssignResult(
        #         num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = ops.stack((gt_cx, gt_cy), axis=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = ops.stack((bboxes_cx, bboxes_cy), axis=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
        #         and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
        #     ignore_overlaps = self.iou_calculator(
        #         bboxes, gt_bboxes_ignore, mode='iof')
        #     ignore_max_overlaps, _ = ignore_overlaps.max(axis=1)
        #     ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
        #     distances[ignore_idxs, :] = INF
        #     assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            # selectable_k = int(ops.minimum(Tensor(self.topk), bboxes_per_level))
            selectable_k = self.topk
            
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = ops.cat(candidate_idxs, axis=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, ops.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).broadcast_to((num_gt, num_bboxes)).view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).broadcast_to((num_gt, num_bboxes)).view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = ops.stack([l_, t_, r_, b_], axis=1).min(axis=1) > 0.01

        is_pos = ops.logical_and(is_pos, is_in_gts)
        is_pos = ops.logical_and(is_pos, gt_valids.bool())

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = ops.full((overlaps.size + 1,), -INF, dtype=ms.float32)

        index = candidate_idxs.view(-1) * is_pos.astype(candidate_idxs.dtype).view(-1) + -1 * (1 - is_pos.astype(candidate_idxs.dtype).view(-1))
        overlaps_inf[index] = overlaps.t().view(-1)[index]
        overlaps_inf = overlaps_inf[:-1]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps = overlaps_inf.max(axis=1)
        argmax_overlaps = overlaps_inf.argmax(axis=1)
        assigned_gt_inds[:-1] = ops.where(max_overlaps != -INF, argmax_overlaps + 1, assigned_gt_inds[:-1])
        # assigned_gt_inds[:-1][max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        assigned_labels = ops.full((num_bboxes + 1,),
                                    -1,
                                    dtype=assigned_gt_inds.dtype)
        assigned_labels = ops.stop_gradient(assigned_labels)
        pos_mask = (assigned_gt_inds > 0).squeeze()
        aranges = ops.arange(num_bboxes, dtype=ms.int32)

        inds = ops.where(pos_mask[:-1], aranges, num_bboxes)
        # if pos_mask.sum() > 0:
        assigned_labels[inds] = gt_labels[assigned_gt_inds[inds] - 1]
        assigned_labels = assigned_labels[:-1]

        assigned_gt_inds = assigned_gt_inds[:-1]
        return assigned_gt_inds, max_overlaps, assigned_labels

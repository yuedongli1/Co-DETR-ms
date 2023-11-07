from mindspore import nn, ops, Tensor

from common.core.bbox.transforms import generalized_box_iou, box_cxcywh_to_xyxy

from mindspore.scipy.ops import LinearSumAssignment
from mindspore.scipy.utils import _mstype_check, _dtype_check
from mindspore.common import dtype as mstype


class NetLsap(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = LinearSumAssignment().set_device(device_target='CPU')

    def construct(self, cost_matrix, maximize, dimension_limit):
        func_name = 'linear_sum_assignment'
        _mstype_check(func_name, cost_matrix, mstype.TensorType, 'cost_matrix')
        _mstype_check(func_name, dimension_limit,
                      mstype.TensorType, 'dimension_limit')
        _mstype_check(func_name, maximize, mstype.TensorType, 'maximize')
        _dtype_check(func_name, cost_matrix, [mstype.float32, mstype.float64])
        _dtype_check(func_name, dimension_limit, [mstype.int64])
        _dtype_check(func_name, maximize, [mstype.bool_])
        return self.op(cost_matrix, dimension_limit, maximize)


class HungarianAssigner(nn.Cell):
    """HungarianMatcher which computes an assignment between targets and predictions.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class (float): The relative weight of the classification error
            in the matching cost. Default: 1.
        cost_bbox (float): The relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: 1.
        cost_giou (float): This is the relative weight of the giou loss of
            the bounding box in the matching cost. Default: 1.
        cost_class_type (str): How the classification error is calculated.
            Choose from ``["ce_cost", "focal_loss_cost"]``. Default: "focal_loss_cost".
        alpha (float): Weighting factor in range (0, 1) to balance positive vs
            negative examples in focal loss. Default: 0.25.
        gamma (float): Exponent of modulating factor (1 - p_t) to balance easy vs
            hard examples in focal loss. Default: 2.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_class_type: str = "focal_loss_cost",
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class_type = cost_class_type
        self.alpha = alpha
        self.gamma = gamma
        self.linear_sum_assignment = NetLsap()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        assert cost_class_type in {
            "ce_cost",
            "focal_loss_cost",
        }, "only support ce loss or focal loss for computing class cost"

    def construct(self, outputs, bs_tgt_labels, bs_tgt_bboxes, bs_tgt_valids):
        """Forward function for `HungarianMatcher` which performs the matching.

        Args:
            outputs (Dict[str, torch.Tensor]): This is a dict that contains at least these entries:

                - ``"pred_logits"``: Tensor of shape (bs, num_queries, num_classes) with the classification logits.
                - ``"pred_boxes"``: Tensor of shape (bs, num_queries, 4) with the predicted box coordinates.

            targets (List[Dict[str, torch.Tensor]]): This is a list of targets (len(targets) = batch_size),
                where each target is a dict containing:

                - ``"labels"``: Tensor of shape (num_target_boxes, ) (where num_target_boxes is the number of
                                ground-truth objects in the target) containing the class labels.  # noqa
                - ``"boxes"``: Tensor of shape (num_target_boxes, 4) containing the target box coordinates.

        Returns:
            list[torch.Tensor]: A list of size batch_size, containing tuples of `(index_i, index_j)` where:

                - ``index_i`` is the indices of the selected predictions (in order)
                - ``index_j`` is the indices of the corresponding selected targets (in order)

            For each batch element, it holds: `len(index_i) = len(index                                                                                                                                                                                                                                                                                                                                                                                               _j) = min(num_queries, num_target_boxes)`
        """
        pred_logits, pred_boxes = outputs
        # (bs, num_box)   (bs, num_box, 4)   (bs, num_box)

        bs, num_queries = pred_logits.shape[:2]
        num_pad_box = bs_tgt_labels.shape[1]

        # Flatten batch to compute the cost matrices in a batch
        if self.cost_class_type == "ce_cost":
            out_prob = (
                ops.reshape(pred_logits, (bs * num_queries, -1)).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
        elif self.cost_class_type == "focal_loss_cost":
            out_prob = (
                ops.sigmoid(ops.reshape(pred_logits, (bs * num_queries, -1)))
            )  # [batch_size * num_queries, num_classes]
        else:
            raise NotImplementedError(f'support only ce_cost and focal_loss_cost, '
                                      f'but got class_type {self.cost_class_type}')

        out_bbox = ops.reshape(pred_boxes, (bs * num_queries, -1))  # [batch_size * num_queries, 4]

        # Flatten batch
        tgt_ids = bs_tgt_labels.reshape(-1)  # (bs*num_box,)
        tgt_bbox = bs_tgt_bboxes.reshape(bs * num_pad_box, -1)  # (bs*num_box, 4)

        # Compute the classification cost.
        if self.cost_class_type == "ce_cost":
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]
        elif self.cost_class_type == "focal_loss_cost":
            alpha = self.alpha
            gamma = self.gamma
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = ops.gather(pos_cost_class - neg_cost_class, tgt_ids, axis=1)
            # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            raise NotImplementedError(f'support only ce_cost and focal_loss_cost, '
                                      f'but got class_type {self.cost_class_type}')

        # Compute the L1 cost between boxes
        cost_bbox = ops.cdist(out_bbox, tgt_bbox, p=1.0)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix, batch * batch
        weighted_cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # to check, .cpu() removed
        weighted_cost_matrix = weighted_cost_matrix.view(bs, num_queries, -1)  # (bs, num_query, sum_instance)

        # TODO to test, hungarian matcher does not need gradient
        weighted_cost_matrix = ops.stop_gradient(weighted_cost_matrix)

        # split_sections = ops.cumsum(Tensor(sizes, dtype=ms.int32), axis=0)[:-1]
        split_weights = ops.split(weighted_cost_matrix, num_pad_box, axis=-1)

        src_idx = []
        tgt_idx = []
        for i in range(bs):
            src_id, tgt_id = self.linear_sum_assignment(split_weights[i][i], Tensor(False), bs_tgt_valids[i].sum().long())
            src_id = ops.stop_gradient(src_id[0])
            tgt_id = ops.stop_gradient(tgt_id[0])
            src_idx.append(src_id)
            tgt_idx.append(tgt_id)
        return ops.stack(src_idx), ops.stack(tgt_idx)

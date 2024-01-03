import math
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.initializer import HeUniform

from .roi_extractor import RoIExtractor
from ..label_assignment import BBoxAssigner, bbox2delta


def delta2bbox(deltas, boxes, weights=(10.0, 10.0, 5.0, 5.0), max_shape=None):
    """Decode deltas to boxes.
    Note: return tensor shape [n,1,4]
    """
    clip_scale = 4

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0:1] / wx
    dy = deltas[:, 1:2] / wy
    dw = deltas[:, 2:3] / ww
    dh = deltas[:, 3:4] / wh
    # Prevent sending too large values into ops.exp()
    dw = ops.minimum(dw, clip_scale)
    dh = ops.minimum(dh, clip_scale)

    pred_ctr_x = dx * ops.expand_dims(widths, 1) + ops.expand_dims(ctr_x, 1)
    pred_ctr_y = dy * ops.expand_dims(heights, 1) + ops.expand_dims(ctr_y, 1)
    pred_w = ops.exp(dw) * ops.expand_dims(widths, 1)
    pred_h = ops.exp(dh) * ops.expand_dims(heights, 1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = ops.stack(pred_boxes, axis=-1)

    if max_shape is not None:
        h, w = max_shape
        x1 = ops.clip_by_value(pred_boxes[..., 0], 0, w - 1)
        y1 = ops.clip_by_value(pred_boxes[..., 1], 0, h - 1)
        x2 = ops.clip_by_value(pred_boxes[..., 2], 0, w - 1)
        y2 = ops.clip_by_value(pred_boxes[..., 3], 0, h - 1)
        pred_boxes = ops.stack((x1, y1, x2, y2), -1)

    return pred_boxes


class RCNNBBoxTwoFCHead(nn.Cell):
    """
    RCNN bbox head with Two fc layers to extract feature

    Args:
        in_channel (int): Input channel which can be derived by from_config
        out_channel (int): Output channel
        resolution (int): Resolution of input feature map, default 7
    """

    def __init__(self, in_channel=256, out_channel=1024, resolution=7):
        super(RCNNBBoxTwoFCHead, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc6 = nn.Dense(
            in_channel * resolution * resolution,
            out_channel,
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.fc7 = nn.Dense(
            out_channel,
            out_channel,
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.relu = nn.ReLU()

    def construct(self, rois_feat):
        b, n, c, _, _ = rois_feat.shape
        rois_feat = rois_feat.reshape(b * n, -1)
        fc6 = self.fc6(rois_feat)
        fc6 = self.relu(fc6)
        fc7 = self.fc7(fc6)
        fc7 = self.relu(fc7)
        return fc7

bbox_head_types = {'RCNNBBoxTwoFCHead': RCNNBBoxTwoFCHead}


def build_bbox_head(cfg):
    obj_cls = bbox_head_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

roi_extractor_types = {'RoIExtractor': RoIExtractor}


def build_roi_extractor(cfg):
    obj_cls = roi_extractor_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

bbox_assigner_types = {'BBoxAssigner': BBoxAssigner}


def build_bbox_assigner(cfg):
    obj_cls = bbox_assigner_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


def get_head(cfg, resolution=7):
    if cfg.name == "RCNNBBoxTwoFCHead":
        return RCNNBBoxTwoFCHead(in_channel=cfg.in_channel, out_channel=cfg.out_channel, resolution=resolution)
    else:
        raise InterruptedError(f"Not support bbox_head: {cfg.name}")


class ROIHead(nn.Cell):
    """RCNN bbox head"""

    def __init__(self, bbox_head, roi_extractor, bbox_assigner, num_classes=80, with_mask=False):
        super(ROIHead, self).__init__()
        self.head = build_bbox_head(bbox_head)
        self.roi_extractor = build_roi_extractor(roi_extractor)
        self.bbox_assigner = build_bbox_assigner(bbox_assigner)
        self.num_classes = num_classes
        self.cls_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.loc_loss = nn.SmoothL1Loss(reduction="none")
        self.bbox_cls = nn.Dense(
            self.head.out_channel,
            self.num_classes + 1,
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.bbox_delta = nn.Dense(
            self.head.out_channel,
            4 * self.num_classes,
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.onehot = nn.OneHot(depth=self.num_classes)
        self.with_mask = with_mask

    def construct(self, feats,
                  rois, rois_mask, gts, gt_masks=None):
        """
        feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_mask (Tensor): The number of RoIs in each image
        gts (Tensor): The ground-truth
        """
        if self.with_mask:
            gt_classes, gt_bboxes, gt_masks, fg_masks, valid_masks, select_rois, pos_rois = self.bbox_assigner(
                rois, rois_mask, gts, gt_masks
            )
        else:
            pos_rois = None
            gt_classes, gt_bboxes, fg_masks, valid_masks, select_rois = self.bbox_assigner(rois, rois_mask, gts)
        batch_size, rois_num, _ = select_rois.shape
        rois_feat = self.roi_extractor(feats, select_rois, valid_masks)
        feat = self.head(rois_feat)

        pred_cls = self.bbox_cls(feat).astype(ms.float32)
        pred_delta = self.bbox_delta(feat).astype(ms.float32)

        # bbox cls
        loss_bbox_cls = self.cls_loss(pred_cls, gt_classes.reshape(-1))
        loss_bbox_cls = ops.select(
            valid_masks.reshape((-1)).astype(ms.bool_), loss_bbox_cls, ops.zeros_like(loss_bbox_cls)
        )
        loss_bbox_cls = loss_bbox_cls.sum() / (valid_masks.astype(pred_cls.dtype).sum() + 1e-4)

        # bbox reg
        reg_target = bbox2delta(select_rois.reshape((-1, 4)), gt_bboxes.reshape((-1, 4)))
        reg_target = ops.tile(ops.expand_dims(reg_target, 1), (1, self.num_classes, 1))
        reg_target = ops.stop_gradient(reg_target)
        cond = ops.logical_and(gt_classes < self.num_classes, gt_classes >= 0)
        reg_class = ops.select(cond, gt_classes, ops.zeros_like(gt_classes)).reshape(-1)
        reg_class_weight = ops.expand_dims(self.onehot(reg_class), -1)
        reg_class_weight = ops.stop_gradient(
            reg_class_weight * fg_masks.reshape((-1, 1, 1)).astype(reg_class_weight.dtype))
        loss_bbox_reg = self.loc_loss(pred_delta.reshape(-1, self.num_classes, 4), reg_target)
        loss_bbox_reg = loss_bbox_reg * reg_class_weight
        loss_bbox_reg = loss_bbox_reg.sum() / (valid_masks.astype(pred_delta.dtype).sum() + 1e-4)
        if self.with_mask:
            mask_weights = reg_class_weight.reshape(batch_size, rois_num, self.num_classes) # B, N, 80, 1, 1
            mask_weights = mask_weights[:, :pos_rois.shape[1], :].reshape(-1, self.num_classes)
            return loss_bbox_reg, loss_bbox_cls, pos_rois, gt_masks, mask_weights, valid_masks
        return loss_bbox_reg, loss_bbox_cls

    def predict(self, feats, rois, rois_mask):
        batch_size, rois_num, _ = rois.shape
        rois_feat = self.roi_extractor(feats, rois, rois_mask)
        feat = self.head(rois_feat)
        pred_cls = self.bbox_cls(feat).astype(ms.float32)
        pred_cls = pred_cls.reshape((batch_size, rois_num, -1))
        pred_cls = ops.softmax(pred_cls, axis=-1)
        pred_delta = self.bbox_delta(feat).astype(ms.float32).reshape((batch_size, rois_num, self.num_classes, 4))
        rois = ops.tile(rois[:, :, :4].reshape((batch_size, rois_num, 1, 4)), (1, 1, self.num_classes, 1))
        # rois = rois.reshape((-1, rois.shape[-1]))[:, :4]
        pred_loc = delta2bbox(pred_delta.reshape((-1, 4)), rois.reshape((-1, 4)))  # true box xyxy
        pred_loc = pred_loc.reshape((batch_size, rois_num, self.num_classes * 4))

        pred_cls_mask = ops.tile(
            rois_mask.astype(ms.bool_).reshape(batch_size, rois_num, 1), (1, 1, pred_cls.shape[-1])
        )
        pred_cls = ops.select(pred_cls_mask, pred_cls, ops.ones_like(pred_cls) * -1).reshape(
            (batch_size, rois_num, self.num_classes + 1)
        )
        return ops.concat((pred_loc, pred_cls), axis=-1)
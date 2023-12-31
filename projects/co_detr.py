import numpy as np

from mindspore import nn, Tensor, ops

from common.models.backbones.builder import build_backbone
from common.models.necks.builder import build_neck
from projects.heads.builder import build_head


class CoDETR(nn.Cell):
    def __init__(self,
                 backbone,
                 neck=None,
                 query_head=None,
                 rpn_head=None,
                 roi_head=None,
                 bbox_head=None,
                 ):
        super().__init__()
        self.backbone = build_backbone(backbone)
        
        if neck is not None:
            self.neck = build_neck(neck)
        
        if query_head is not None:
            self.query_head = build_head(query_head)

        if bbox_head is not None:
            self.bbox_head = build_head(bbox_head)

        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

        if roi_head is not None:
            self.roi_head = build_head(roi_head)

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_bbox_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def construct(self,
                  images, img_masks, gt_label=None, boxes_xywhn=None, boxes_xyxy=None, gt_valid=None, dn_valid=None, img_shape=None):
        query_loss, bbox_loss, rpn_cls_loss, rpn_reg_loss, roi_reg_loss, roi_cls_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        gts = ops.concat([gt_label[:, :, None].float(), boxes_xyxy], -1)
        # extract features with backbone
        features = self.backbone(images, img_masks)

        multi_level_feats = self.neck(features)  # list[b, embed_dim, h, w], len=num_level
        output = self.query_head(multi_level_feats, images, img_masks, gt_label, boxes_xywhn, gt_valid, dn_valid)
        if self.training:
            query_loss, multi_level_feats = output
            if self.with_bbox_head:
                bbox_loss, atss_pos_coords = self.bbox_head(multi_level_feats, img_shape,
                                                      boxes_xyxy, gt_label, gt_valid)
            if self.with_rpn_head:
                rois, rois_mask, rpn_cls_loss, rpn_reg_loss = self.rpn_head(multi_level_feats,
                                                                            gts, img_shape)
            if self.with_roi_head:
                roi_reg_loss, roi_cls_loss = self.roi_head(multi_level_feats,
                                                           rois, rois_mask, gts, gt_valid)
            return query_loss + bbox_loss + rpn_cls_loss + rpn_reg_loss + roi_reg_loss + roi_cls_loss

        return output

    def construct_(self,
                  images, img_masks, gt_label=None, boxes_xywhn=None, boxes_xyxy=None, gt_valid=None, dn_valid=None, img_shape=None):
        query_loss, bbox_loss = 0.0, 0.0
        batch_size, _, h, w = images.shape
        # extract features with backbone
        features = self.backbone(images)

        multi_level_feats = self.neck(features)  # list[b, embed_dim, h, w], len=num_level

        if self.with_query_head:
            query_loss, multi_level_feats = self.query_head(multi_level_feats, images, img_masks, gt_label, boxes_xywhn, gt_valid, dn_valid)

        if self.with_bbox_head:
            feat0, feat1, feat2, feat3, feat4, feat5 = multi_level_feats
            bbox_loss, atss_pos_coords = self.bbox_head(feat0, feat1, feat2, feat3, feat4, feat5, img_shape, boxes_xyxy, gt_label, gt_valid)

        # return query_loss + bbox_loss
        return bbox_loss


co_detr_types = {'CoDETR': CoDETR}


def build_co_detr(cfg):
    obj_cls = co_detr_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

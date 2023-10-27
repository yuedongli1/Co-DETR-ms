from mindspore import nn
from common.models.backbones.builder import build_backbone
from common.models.necks.builder import build_neck
from projects.heads.builder import build_head


class CoDETR(nn.Cell):
    def __init__(self, backbone, neck=None, query_head=None):
        super().__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if query_head is not None:
            self.query_head = build_head(query_head)

    def construct(self, images, img_masks, targets=None):
        batch_size, _, h, w = images.shape
        # extract features with backbone
        features = self.backbone(images)

        multi_level_feats = self.neck(features)  # list[b, embed_dim, h, w], len=num_level
        box_cls, box_pred = self.query_head(multi_level_feats, images, img_masks, targets)
        return box_cls, box_pred

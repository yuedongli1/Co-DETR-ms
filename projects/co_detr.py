from mindspore import nn
from common.models.backbones.builder import build_backbone
from common.models.necks.builder import build_neck
from projects.heads.builder import build_head


class CoDETR(nn.Cell):
    def __init__(self,
                 backbone,
                 neck=None,
                 query_head=None,
                 ):
        super().__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        head_idx = 0

        if query_head is not None:
            self.query_head = build_head(query_head)
            head_idx += 1

    def construct(self, images, img_masks, gt_label=None, gt_box=None, gt_valid=None, dn_valid=None):
        batch_size, _, h, w = images.shape
        # extract features with backbone
        features = self.backbone(images)

        multi_level_feats = self.neck(features)  # list[b, embed_dim, h, w], len=num_level
        output = self.query_head(multi_level_feats, images, img_masks, gt_label, gt_box, gt_valid, dn_valid)
        return output


co_detr_types = {'CoDETR': CoDETR}


def build_co_detr(cfg):
    obj_cls = co_detr_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

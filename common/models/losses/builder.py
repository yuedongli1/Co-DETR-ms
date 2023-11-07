from common.models.losses.cross_entropy_loss import CrossEntropyLoss
from common.models.losses.focal_loss import FocalLoss
from common.models.losses.iou_loss import GIoULoss

loss_types = {'CrossEntropyLoss': CrossEntropyLoss,
              'FocalLoss': FocalLoss,
              'GIoULoss': GIoULoss}


def build_loss(cfg):
    obj_cls = loss_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

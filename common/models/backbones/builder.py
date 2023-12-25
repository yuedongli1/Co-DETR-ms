from common.models.backbones.resnet import ResNet
from common.models.backbones.swin_transformer import SwinTransformer

backbone_types = {'ResNet': ResNet,
                  'SwinTransformer': SwinTransformer}


def build_backbone(cfg):
    obj_cls = backbone_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

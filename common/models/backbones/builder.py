from common.models.backbones.resnet import ResNet

backbone_types = {'ResNet': ResNet}


def build_backbone(cfg):
    obj_cls = backbone_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

from common.core.bbox.coder.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder

coder_types = {'DeltaXYWHBBoxCoder': DeltaXYWHBBoxCoder}


def build_bbox_coder(cfg):
    obj_cls = coder_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

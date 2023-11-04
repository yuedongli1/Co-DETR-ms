from common.core.bbox.assigners.hungarian_assigner import HungarianAssigner

assigner_types = {'HungarianAssigner': HungarianAssigner}


def build_assigner(cfg):
    obj_cls = assigner_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

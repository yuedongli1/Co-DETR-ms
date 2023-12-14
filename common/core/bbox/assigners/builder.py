from common.core.bbox.assigners.hungarian_assigner import HungarianAssigner
from common.core.bbox.assigners.atss_assigner import ATSSAssigner

assigner_types = {'HungarianAssigner': HungarianAssigner,
                  'ATSSAssigner': ATSSAssigner}


def build_assigner(cfg):
    obj_cls = assigner_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

from projects.heads.co_dino_head import CoDINOHead
from projects.heads.co_atss_head import CoATSSHead

head_types = {'CoDINOHead': CoDINOHead,
              'CoATSSHead': CoATSSHead}


def build_head(cfg):
    obj_cls = head_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

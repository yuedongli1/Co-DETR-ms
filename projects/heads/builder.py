from projects.heads.co_dino_head import CoDINOHead

head_types = {'CoDINOHead': CoDINOHead}


def build_head(cfg):
    obj_cls = head_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

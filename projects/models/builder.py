from projects.models.co_detr import CoDETR

co_detr_types = {'CoDETR': CoDETR}


def build_co_detr(cfg):
    obj_cls = co_detr_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

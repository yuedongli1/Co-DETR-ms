from projects.criterions.dino_criterion import DINOCriterion


criterion_types = {'DINOCriterion': DINOCriterion}


def build_criterion(cfg):
    obj_cls = criterion_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

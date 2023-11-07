from common.core.anchor.anchor_generator import AnchorGenerator

prior_generator_types = {'AnchorGenerator': AnchorGenerator}


def build_prior_generator(cfg):
    obj_cls = prior_generator_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

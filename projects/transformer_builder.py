from projects.transformer import CoDinoTransformer

transformer_types = {'CoDinoTransformer': CoDinoTransformer}


def build_transformer(cfg):
    obj_cls = transformer_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

from common.models.necks.channel_mapper import ChannelMapper

neck_types = {
    'ChannelMapper': ChannelMapper,
}


def build_neck(cfg, **kwargs):
    obj_cls = neck_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

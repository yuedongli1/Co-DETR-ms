from common.models.layers.position_embedding import PositionEmbeddingSine
from common.models.layers.transformer import DINOTransformerEncoder, DINOTransformerDecoder

position_embedding_types = {'PositionEmbeddingSine': PositionEmbeddingSine}

encoder_types = {'DINOTransformerEncoder': DINOTransformerEncoder}

decoder_types = {'DINOTransformerDecoder': DINOTransformerDecoder}


def build_position_embedding(cfg):
    obj_cls = position_embedding_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


def build_encoder(cfg):
    obj_cls = encoder_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


def build_decoder(cfg):
    obj_cls = decoder_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

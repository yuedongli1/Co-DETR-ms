from projects.heads.co_dino_head import CoDINOHead
from projects.heads.co_atss_head import CoATSSHead
from common.models.heads.roi_head.roi_head import ROIHead
from common.models.heads.rpn_head.rpn_head import RPNHead

head_types = {'CoDINOHead': CoDINOHead,
              'CoATSSHead': CoATSSHead,
              'ROIHead': ROIHead,
              'RPNHead': RPNHead}


def build_head(cfg):
    obj_cls = head_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)

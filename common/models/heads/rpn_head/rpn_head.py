import math
from mindspore import ops, nn
from mindspore.common.initializer import HeUniform
from .proposal_generator import ProposalGenerator
from ..label_assignment import RPNLabelAssignment

from common.core.anchor.builder import build_prior_generator

proposal_generator_types = {'ProposalGenerator': ProposalGenerator}


def build_proposal_generator(cfg):
    obj_cls = proposal_generator_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


label_assigner_types = {'RPNLabelAssignment': RPNLabelAssignment}


def build_label_assigner(cfg):
    obj_cls = label_assigner_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


class RPNFeat(nn.Cell):
    """
    Feature extraction in RPN head

    Args:
        num_layers (int): Feat numbers
        in_channel (int): Input channel
        out_channel (int): Output channel
    """

    def __init__(self, num_layers=1, num_anchors=3, in_channel=1024, out_channel=1024):
        super(RPNFeat, self).__init__()
        self.rpn_conv = nn.Conv2d(in_channel,
                                  out_channel,
                                  kernel_size=3,
                                  padding=1,
                                  pad_mode="pad",
                                  weight_init=HeUniform(math.sqrt(5)),
                                  has_bias=True,
                                  bias_init="zeros"
        )
        self.rpn_cls = nn.Conv2d(
            out_channel, num_anchors, 1, weight_init=HeUniform(math.sqrt(5)), has_bias=True, bias_init="zeros"
        )
        self.rpn_reg = nn.Conv2d(
            out_channel, 4 * num_anchors, 1, weight_init=HeUniform(math.sqrt(5)), has_bias=True, bias_init="zeros"
        )
        self.relu = nn.ReLU()

    def construct(self, feats):
        rpn_cls_score = ()
        rpn_bbox_pred = ()
        for i, feat in enumerate(feats):
            x = self.relu(self.rpn_conv(feat))
            rpn_cls_score += (self.rpn_cls(x),)
            rpn_bbox_pred += (self.rpn_reg(x),)
        return rpn_cls_score, rpn_bbox_pred


class RPNHead(nn.Cell):
    """
    Region Proposal Network

    Args:
        cfg(Config): rpn_head config
        backbone_feat_nums(int): backbone feature numbers
        in_channel(int): rpn feature conv in channel
        loss_rpn_bbox(Cell): bbox loss function Cell, default is MAELoss
    """

    def __init__(self, in_channel, feat_channel, backbone_feat_nums=1,
                 loss_rpn_bbox=None, anchor_generator=None, proposal_generator=None, label_assigner=None):
        super(RPNHead, self).__init__()
        if anchor_generator:
            self.prior_generator = build_prior_generator(anchor_generator)
        if proposal_generator:
            self.proposal_generator = build_proposal_generator(proposal_generator)
        if label_assigner:
            self.label_assigner = build_label_assigner(label_assigner)
        self.num_anchors = self.prior_generator.num_base_priors[0]
        self.rpn_feat = RPNFeat(backbone_feat_nums, self.num_anchors, in_channel, feat_channel)

        self.loss_rpn_bbox = loss_rpn_bbox
        if self.loss_rpn_bbox is None:
            self.loss_rpn_bbox = nn.SmoothL1Loss(reduction="none")

    def get_anchors(self, featmap_sizes, img_shapes):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator(
            featmap_sizes)
        # multi_level_anchors = [ops.zeros((14400, 4)), ops.zeros((3600, 4)), ops.zeros((920, 4)), ops.zeros((240, 4)),
        #                        ops.zeros((60, 4)), ops.zeros((15, 4))]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_shape in enumerate(img_shapes):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_shape)
            valid_flag_list.append(multi_level_flags)

        num_level_anchors = []
        for anchors in multi_level_anchors:
            num_level_anchors.append(anchors.shape[0])
        return multi_level_anchors, ops.cat(valid_flag_list[0]).bool(), num_level_anchors

    def construct(self, feats,
                  gts, image_shape):
        scores, deltas = self.rpn_feat(feats)
        shapes = ()
        for feat in feats:
            shapes += (feat.shape[-2:],)
        anchor_list, valid_flag_list, num_level_anchors = self.get_anchors(shapes, image_shape)
        rois, rois_mask = self.proposal_generator(scores, deltas, anchor_list, image_shape[0])
        tgt_labels, tgt_bboxes, tgt_deltas = self.label_assigner(gts, anchor_list)

        # cls loss
        score_pred = ()
        batch_size = scores[0].shape[0]
        for score in scores:
            score_pred = score_pred + (ops.transpose(score, (0, 2, 3, 1)).reshape((batch_size, -1)),)
        score_pred = ops.concat(score_pred, 1)
        valid_mask = tgt_labels >= 0
        fg_mask = tgt_labels > 0

        loss_rpn_cls = ops.SigmoidCrossEntropyWithLogits()(score_pred, fg_mask.astype(score_pred.dtype))
        loss_rpn_cls = ops.select(valid_mask, loss_rpn_cls, ops.zeros_like(loss_rpn_cls))

        # reg loss
        delta_pred = ()
        for delta in deltas:
            delta_pred = delta_pred + (ops.transpose(delta, (0, 2, 3, 1)).reshape((batch_size, -1, 4)),)
        delta_pred = ops.concat(delta_pred, 1)
        loss_rpn_reg = self.loss_rpn_bbox(delta_pred, tgt_deltas)
        fg_mask = ops.tile(ops.expand_dims(fg_mask.int(), -1), (1, 1, 4)).bool()
        loss_rpn_reg = ops.select(fg_mask, loss_rpn_reg, ops.zeros_like(loss_rpn_reg))
        loss_rpn_cls = loss_rpn_cls.sum() / (valid_mask.astype(loss_rpn_cls.dtype).sum() + 1e-4)
        loss_rpn_reg = loss_rpn_reg.sum() / (valid_mask.astype(loss_rpn_reg.dtype).sum() + 1e-4)
        return rois, rois_mask, loss_rpn_cls, loss_rpn_reg

    def predict(self, feats, image_shape):
        scores, deltas = self.rpn_feat(feats)
        shapes = ()
        for feat in feats:
            shapes += (feat.shape[-2:],)
        anchors = self.anchor_generator(shapes)
        rois, rois_mask = self.test_gen_proposal(scores, deltas, anchors, image_shape)
        return rois, rois_mask
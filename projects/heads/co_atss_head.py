import mindspore as ms
import mindspore.nn as nn

from mindspore import Tensor, ops

from common.models.layers.conv import ConvNormAct
from common.core.anchor.utils import images_to_levels
from common.core.bbox.coder.builder import build_bbox_coder
from common.core.bbox.assigners.builder import build_assigner
from common.models.losses.builder import build_loss
from common.core.bbox.samplers.builder import build_sampler
from common.core.anchor.builder import build_prior_generator


def get_norm_from_str(norm_str):
    if norm_str == 'GN':
        gn = nn.GroupNorm(num_groups=32, num_channels=256)
    else:
        raise NotImplementedError(f'require norm_str [FrozenBN], [BN], got [{norm_str}] instead')

    return gn


class Scale(nn.Cell):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = ms.Parameter(Tensor(scale, dtype=ms.float32))

    def construct(self, x):
        return x * self.scale


class CoATSSHead(nn.Cell):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 norm='GN',
                 reg_decoded_bbox=True,
                 loss_centerness=None,
                 init_cfg=None,
                 anchor_generator=None,
                 loss_cls=None,
                 bbox_coder=None,
                 loss_bbox=None,
                 assigner=None,
                 train_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        super(CoATSSHead, self).__init__()

        if isinstance(norm, str):
            norm_layer1 = get_norm_from_str(norm)
            norm_layer2 = get_norm_from_str(norm)

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        if bbox_coder:
            self.bbox_coder = build_bbox_coder(bbox_coder)
        if loss_cls:
            self.loss_cls = build_loss(loss_cls)
        if loss_bbox:
            self.loss_bbox = build_loss(loss_bbox)
        self.sampling = False

        if assigner:
            self.assigner = build_assigner(assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg)
        if loss_centerness:
            self.loss_centerness = build_loss(loss_centerness)

        if anchor_generator:
            self.prior_generator = build_prior_generator(anchor_generator)
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.reg_decoded_bbox = reg_decoded_bbox
        self.relu = nn.ReLU()
        self.cls_convs = nn.CellList()
        self.reg_convs = nn.CellList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvNormAct(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    pad_mode='pad',
                    padding=1,
                    norm_layer=norm_layer1,
                    bias=False,
                    activation=nn.ReLU()))
            self.reg_convs.append(
                ConvNormAct(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    pad_mode='same',
                    padding=0,
                    norm_layer=norm_layer2,
                    bias=False,
                    activation=nn.ReLU()))
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            pad_mode='pad',
            padding=1,
            has_bias=True)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, pad_mode='pad', padding=1, has_bias=True)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 1, 3, pad_mode='pad', padding=1, has_bias=True)
        self.scales = nn.CellList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        self.train_cfg = train_cfg

    @property
    def num_anchors(self):
        return self.prior_generator.num_base_priors[0]

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
        num_imgs = len(img_shapes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator(
            featmap_sizes)
        # multi_level_anchors = [ops.zeros((14400, 4)), ops.zeros((3600, 4)), ops.zeros((920, 4)), ops.zeros((240, 4)),
        #                        ops.zeros((60, 4)), ops.zeros((15, 4))]
        anchor_list = [ops.cat(multi_level_anchors) for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_shape in enumerate(img_shapes):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_shape)
            valid_flag_list.append(multi_level_flags)

        num_level_anchors = []
        for anchors in multi_level_anchors:
            num_level_anchors.append(anchors.shape[0])
        return anchor_list, [ops.cat(valid_flag).bool() for valid_flag in valid_flag_list], num_level_anchors

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           gt_valids,
                           img_shape,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        # inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
        #                                    img_shape,
        #                                    self.train_cfg['allowed_border'])

        # assign gt and sample anchors
        num_gt = gt_bboxes.shape[0]
        anchors = flat_anchors

        # num_level_anchors_inside = self.get_num_level_anchors_inside(
        #     num_level_anchors, valid_flags)
        num_level_anchors_inside = num_level_anchors
        assigned_gt_inds, max_overlaps, assigned_labels = self.assigner(anchors, num_level_anchors_inside,
                                                                             gt_bboxes, gt_bboxes_ignore,
                                                                             gt_labels, gt_valids)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = ops.zeros_like(anchors)
        bbox_weights = ops.zeros_like(anchors)
        labels = ops.full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=ms.int64)
        labels = ops.stop_gradient(labels)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=ms.float32)

        is_pos = assigned_gt_inds != num_gt

        # pos
        # if is_pos.any():
        pos_bbox_targets = gt_bboxes[assigned_gt_inds - 1]
        bbox_targets = ops.where(is_pos[:, None], pos_bbox_targets, bbox_targets)
        bbox_weights = ops.where(is_pos[:, None], 1.0, bbox_weights)
        labels = ops.where(is_pos, assigned_labels, labels)
        if self.train_cfg['pos_weight'] <= 0:
            label_weights = ops.where(is_pos, 1.0, label_weights)
        else:
            label_weights = ops.where(is_pos, self.train_cfg['pos_weight'], label_weights)

        # neg
        # if not is_pos.all():
        label_weights = ops.where(ops.logical_not(is_pos), 1.0, label_weights)

        return anchors, labels, label_weights, bbox_targets, bbox_weights, is_pos

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = ops.split(inside_flags, num_level_anchors, 0)
        num_level_anchors_inside = [
            flags.sum() for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    num_level_anchors,
                    gt_bboxes_list,
                    img_shapes,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    gt_valids_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_shapes)

        # anchor number of multi levels
        num_level_anchors_list = (num_level_anchors,) * num_imgs

        # compute targets for each image
        # if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        # if gt_labels_list is None:
        #     gt_labels_list = [None for _ in range(num_imgs)]
        all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, is_pos_list = \
            [None] * num_imgs, [None] * num_imgs, [None] * num_imgs, [None] * num_imgs, [None] * num_imgs, [None] * num_imgs
        for i, (anchor, valid_flag, num_level_anchors, gt_bboxes, gt_bboxes_ignore, gt_labels, gt_valids, img_shape) in \
                enumerate(zip(anchor_list, valid_flag_list, num_level_anchors_list, gt_bboxes_list, gt_bboxes_ignore_list, gt_labels_list, gt_valids_list, img_shapes)):
            anchors, labels, label_weights, bbox_targets, bbox_weights, is_pos = self._get_target_single(
                                                                                            anchor,
                                                                                            valid_flag,
                                                                                            num_level_anchors,
                                                                                            gt_bboxes[0],
                                                                                            gt_bboxes_ignore,
                                                                                            gt_labels[0],
                                                                                            gt_valids[0],
                                                                                            img_shape,
                                                                                            )
            all_anchors[i] = anchors
            all_labels[i] = labels
            all_label_weights[i] = label_weights
            all_bbox_targets[i] = bbox_targets
            all_bbox_weights[i] = bbox_weights
            is_pos_list[i] = is_pos
        # no valid anchors
        # if any([labels is None for labels in all_labels]):
        #     return None
        # sampled anchors of all images
        num_total_pos = sum([ops.maximum(inds.sum(), 1) for inds in is_pos_list])
        num_total_neg = sum([ops.maximum(inds.sum(), 1) for inds in is_pos_list])
        # split targets to a list w.r.t. multiple levels
        ori_anchors = all_anchors
        ori_labels = all_labels
        ori_bbox_targets = all_bbox_targets
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return anchors_list, labels_list, label_weights_list, \
               bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, \
               ori_anchors, ori_labels, ori_bbox_targets

    def centerness_target(self, anchors, gts, eps=1e-6):
        # only calculate pos centerness targets, otherwise there may be nan
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = ops.stack([l_, r_], axis=1)
        top_bottom = ops.stack([t_, b_], axis=1)
        centerness = ops.sqrt(
            (left_right.min(axis=-1) / (left_right.max(axis=-1) + eps)) *
            (top_bottom.min(axis=-1) / (top_bottom.max(axis=-1) + eps)))
        # assert not ops.isnan(centerness).any()
        return centerness

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
                    label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels.float(), label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        is_pos = ops.logical_and(labels >= 0, labels < bg_class_ind)

        # if is_pos.any():
        pos_bbox_targets = bbox_targets
        pos_bbox_pred = bbox_pred
        pos_anchors = anchors
        pos_centerness = centerness
        centerness_targets = self.centerness_target(
            pos_anchors, pos_bbox_targets)
        pos_decode_bbox_pred = self.bbox_coder(
            pos_anchors, pos_bbox_pred)

        # regression loss
        loss_bbox = self.loss_bbox(
            pos_decode_bbox_pred,
            pos_bbox_targets,
            weight=centerness_targets,
            avg_factor=1.0,
            mask=is_pos)

        # centerness loss
        loss_centerness = self.loss_centerness(
            pos_centerness,
            centerness_targets,
            avg_factor=num_total_samples,
            mask=is_pos)

        # else:
        #     loss_bbox = bbox_pred.sum() * 0
        #     loss_centerness = centerness.sum() * 0
        #
        #     centerness_targets = Tensor(0., bbox_targets.dtype)
        #     centerness_targets = ops.stop_gradient(centerness_targets)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             gt_valids,
             img_shapes,
             featmap_sizes,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        anchor_list, valid_flag_list, num_level_anchors = self.get_anchors(
            featmap_sizes, img_shapes)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        gt_bboxes_list = ops.split(gt_bboxes, 1)
        gt_labels_list = ops.split(gt_labels, 1)
        gt_valids_list = ops.split(gt_valids, 1)
        anchor_list, labels_list, label_weights_list, bbox_targets_list, \
        bbox_weights_list, num_total_pos, num_total_neg, ori_anchors, \
        ori_labels, ori_bbox_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            num_level_anchors,
            gt_bboxes_list,
            img_shapes,
            gt_bboxes_ignore,
            gt_labels_list,
            gt_valids_list,
            label_channels,
            True)
        # if get_group_size() > 1:
        num_total_samples = num_total_pos
        # else:
        #     num_total_samples = reduce_mean(Tensor(num_total_pos, dtype=ms.float32))
        num_total_samples = ops.maximum(num_total_samples, 1.0)
        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = (), (), (), ()
        for anchor, cls_score, bbox_pred, centerness, labels, label_weights, bbox_targets \
                in zip(anchor_list, cls_scores, bbox_preds, centernesses, labels_list, label_weights_list, bbox_targets_list):
            losses_cls_single, losses_bbox_single, loss_centerness_single, bbox_avg_factor_single = self.loss_single(
                                                                                                                anchor,
                                                                                                                cls_score,
                                                                                                                bbox_pred,
                                                                                                                centerness,
                                                                                                                labels,
                                                                                                                label_weights,
                                                                                                                bbox_targets,
                                                                                                                num_total_samples)
            losses_cls += (losses_cls_single,)
            losses_bbox += (losses_bbox_single,)
            loss_centerness += (loss_centerness_single,)
            bbox_avg_factor += (bbox_avg_factor_single,)

        bbox_avg_factor = sum(bbox_avg_factor)
        # if get_group_size() > 1:
        #     bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()

        losses_bbox = [loss_bbox / bbox_avg_factor for loss_bbox in losses_bbox]

        pos_coords = (ori_anchors, ori_labels, ori_bbox_targets)
        return sum(losses_cls) + sum(losses_bbox) + sum(loss_centerness), pos_coords

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = ()
        bbox_preds = ()
        centernesses = ()
        for i, feat in enumerate(feats):
            cls_score, bbox_pred, centerness = self.forward_single(feat, self.scales[i])
            cls_scores += (cls_score,)
            bbox_preds += (bbox_pred,)
            centernesses += (centerness,)
        return cls_scores, bbox_preds, centernesses

    def construct(self,
                      x,
                      img_shapes,
                      gt_bboxes,
                      gt_labels=None,
                      gt_valids=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        cls_scores, bbox_preds, centernesses = self.forward(x)
        featmap_sizes = [featmap.shape[2:] for featmap in cls_scores]
        losses = self.loss(cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, gt_valids, img_shapes, featmap_sizes, gt_bboxes_ignore=None)

        return losses

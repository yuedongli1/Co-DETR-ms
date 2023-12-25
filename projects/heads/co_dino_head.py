import copy
import math
from typing import List

import mindspore as ms
from mindspore import nn, ops, Tensor, jit
import mindspore.common.initializer as init
import mindspore.numpy as ms_np

from common.models.layers.mlp import MLP
from common.utils.misc import inverse_sigmoid, replace_invalid
from common.utils.torch_converter import init_like_torch
from common.models.layers.builder import build_position_embedding
from common.core.bbox.transforms import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy

from projects.transformers.builder import build_transformer
from projects.criterions.builder import build_criterion


class CoDINOHead(nn.Cell):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
    """

    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 num_queries: int,
                 aux_loss: bool = True,
                 num_dn: int = 100,
                 label_noise_ratio: float = 0.2,
                 box_noise_scale: float = 1.0,
                 transformer=None,
                 position_embedding=None,
                 criterion=None
                 ):
        super().__init__()
        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = build_transformer(transformer)

        # define position embedding module
        self.position_embedding = build_position_embedding(position_embedding)

        # # define criterion
        self.criterion = build_criterion(criterion)

        # define where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss

        # define classification head and box head
        self.class_embed = nn.Dense(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # de-noising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.num_dn = num_dn
        self.num_cdn = num_dn * 2
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        init_like_torch(self.class_embed)
        self.class_embed.bias.set_data(ops.ones(num_classes, self.class_embed.bias.dtype) * bias_value)
        self.bbox_embed.layers[-1].weight.set_data(init.initializer(init.Constant(0),
                                                                    self.bbox_embed.layers[-1].weight.shape,
                                                                    self.bbox_embed.layers[-1].weight.dtype))
        self.bbox_embed.layers[-1].bias.set_data(init.initializer(init.Constant(0),
                                                                  self.bbox_embed.layers[-1].bias.shape,
                                                                  self.bbox_embed.layers[-1].bias.dtype))

        # hack implementaion, the class_embed of the last layer of transformer.decoder serves for two stage
        num_pred = self.transformer.decoder.num_layers + 1
        self.class_embed = nn.CellList([copy.deepcopy(self.class_embed) for _ in range(num_pred)])
        self.bbox_embed = nn.CellList([copy.deepcopy(self.bbox_embed) for _ in range(num_pred)])

        bias_init_data = self.bbox_embed[0].layers[-1].bias.data
        bias_init_data[2:] = Tensor(-2.0)
        # p_type, d_type = self.bbox_embed[0].layers[-1].bias.shape, self.bbox_embed[0].layers[-1].bias.dtype
        # self.bbox_embed[0].layers[-1].bias.set_data(init.initializer(bias_init_data, p_type, d_type))

        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.downsample = nn.SequentialCell(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True),
            nn.GroupNorm(32, embed_dim)
        )

        for bbox_embed_layer in self.bbox_embed:
            bias_init_data = bbox_embed_layer.layers[-1].bias.data
            bias_init_data[2:] = Tensor(-0.0)
            p_type, d_type = bbox_embed_layer.layers[-1].bias.shape, bbox_embed_layer.layers[-1].bias.dtype
            bbox_embed_layer.layers[-1].bias.set_data(init.initializer(bias_init_data, p_type, d_type))

        # operator
        self.uniform_real = ops.UniformReal()
        self.uniform_int = ops.UniformInt()

    def construct(self, multi_level_feats, images, img_masks, gt_label=None, gt_box=None, gt_valid=None, dn_valid=None):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            images (Tensor[b, c, h, w]): batch image
            img_masks (Tensor(b, h, w)): image masks with value 1 for padding area and 0 for valid area
        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        batch_size, _, h, w = images.shape
        # extract features with backbone

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            resize_nearest = ops.ResizeNearestNeighbor(size=feat.shape[-2:])
            l_mask = ops.squeeze(resize_nearest(ops.expand_dims(img_masks, 0)), 0)
            multi_level_masks.append(l_mask)
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # de-noising preprocessing
        # prepare label query embeding
        input_query_label, input_query_bbox, attn_mask, dn_valids = None, None, None, None
        if self.training:
            input_query_label, input_query_bbox, attn_mask, dn_valids = self.cdn_preprocess(
                gt_label,
                gt_box,
                gt_valid,
                dn_valid,
                num_dn=self.num_dn,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_query=self.num_queries,
                num_classes=self.num_classes,
            )

        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (inter_states, init_reference, inter_reference, enc_state, enc_reference, enc_outputs) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,  # gt query and target
            attn_masks=[attn_mask, None],
        )

        # hack implementation for distributed training
        inter_states[0] += self.label_enc.embedding_table[0, 0] * 0.0

        # calculate output coordinates and classes
        outputs_classes = []
        outputs_coords = []

        for lvl in range(inter_states.shape[0]):
            reference = init_reference if lvl == 0 else inter_reference[lvl - 1]
            reference = inverse_sigmoid(reference)
            l_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                # for anchor contains only x,y
                assert reference.shape[-1] == 2
                tmp[..., 2] += reference
            l_coord = ops.sigmoid(tmp)
            outputs_classes.append(l_class)
            outputs_coords.append(l_coord)
        # [num_decoder_layers, bs, num_query, num_classes]
        outputs_class = ops.stack(outputs_classes)
        # [num_decoder_layers, bs, num_query, 4]
        outputs_coord = ops.stack(outputs_coords)

        # output (tuple(tuple(Tensor))) with size 3 -> last, aux, two_stage, dn_last, dn_aux
        output = [None, None, None, None, None]
        if self.training:
            outputs_class, outputs_coord, gt_query_class, gt_query_coord = \
                self.cdn_postprocess(outputs_class, outputs_coord, self.num_cdn)
            if self.num_dn > 0:
                output[3] = (gt_query_class[-1], gt_query_coord[-1])
                if self.aux_loss:
                    output[4] = (gt_query_class[:-1], gt_query_coord[:-1])

        # prepare for loss computation
        output[0] = (outputs_class[-1], outputs_coord[-1])
        if self.aux_loss:
            # len(output["aux_outputs"]) = num_decoder_layer - 1
            output[1] = (outputs_class[:-1], outputs_coord[:-1])

        # prepare two stage output
        interm_coord = enc_reference
        # hack implementaion, the class_embed of the last layer of transformer.decoder serves for two stage
        # TODO 感觉这里可以写在transformer class里，直接输出interm_class，不知作者这么写是不是为了兼容其他的detr
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output[2] = (interm_class, interm_coord)

        if self.training:
            outs = []
            num_level = len(multi_level_feats)
            start = 0
            for lvl in range(num_level):
                bs, c, h, w = multi_level_feats[lvl].shape
                end = start + h * w
                feat = enc_outputs.permute(1, 0, 2)[start:end].permute(1, 2, 0)
                start = end
                outs.append(feat.reshape(bs, c, h, w))
            outs.append(self.downsample(outs[-1]))
            loss = self.criterion(output, gt_label, gt_box, gt_valid, dn_valid)
            return loss, outs

        return output[0]

    def cdn_postprocess(self, outputs_class, outputs_coord, num_cdn):
        """
        cdn postporcess
        Args:
            outputs_class (Tensor[[num_decoder_layers, bs, num_query, 4]]): outputs class with gt and match query
            outputs_coord (Tensor[[num_decoder_layers, bs, num_query, 4]]): outputs box coordinates
             with gt and match query
            num_cdn (int): number of contrastive de-noising query
        """
        if num_cdn <= 0:
            return outputs_class, outputs_coord, None, None

        gt_query_class = outputs_class[:, :, :num_cdn, :]
        gt_query_coord = outputs_coord[:, :, :num_cdn, :]
        match_query_class = outputs_class[:, :, num_cdn:, :]
        match_query_coord = outputs_coord[:, :, num_cdn:, :]

        return match_query_class, match_query_coord, gt_query_class, gt_query_coord

    def cdn_preprocess(
            self,
            tgt_labels,
            tgt_boxes,
            tgt_valids,
            dn_valids,
            num_dn,
            label_noise_ratio,
            box_noise_scale,
            num_query,
            num_classes,
    ):
        """
        generate cdn gt query
        Args:
            targets (Tuple[Tensor]): tuple of gt label, bbox, valid mask and dn_positive_id, dn_valid_mask
            num_dn (int): positive dn number
            label_enc (nn.Embedding[num_class, embed_dim]): label embedding table
        """
        if num_dn <= 0:
            return None, None, None, None

        assert dn_valids.shape[
                   1] == num_dn, f"num_dn should be set as the same in dataset({dn_valids.shape[1]}) and model({num_dn})"
        bs, num_pad_box = tgt_labels.shape
        tgt_labels = replace_invalid(tgt_labels, tgt_valids, num_classes)
        tgt_labels_expand = ops.full((bs, num_pad_box + 1), num_classes, dtype=tgt_labels.dtype)
        tgt_labels_expand[:, :num_pad_box] = tgt_labels
        
        tgt_boxes_expand = ops.full((bs, num_pad_box + 1, 4), 0, dtype=tgt_boxes.dtype)
        tgt_boxes_expand[:, :num_pad_box, :] = tgt_boxes
        
        num_valid_box = ops.reduce_sum(tgt_valids.astype(ms.float32), 1).astype(ms.int32)  # (bs)

        dn_positive_ids = ops.expand_dims(ms_np.arange(num_dn), 0) % num_valid_box.expand_dims(1)  # (bs, num_dn)
        dn_positive_ids = replace_invalid(dn_positive_ids, dn_valids, num_pad_box)  # 012 012 012 -1
        
        known_labels = ops.gather_elements(tgt_labels_expand, 1, dn_positive_ids)  # (bs, num_dn)
        known_boxes = ops.gather_elements(tgt_boxes_expand, 1, ms_np.tile(ops.expand_dims(dn_positive_ids, -1), (1, 1, 4)))

        # add negative query
        num_cdn = num_dn * 2
        known_labels = ops.concat([known_labels, known_labels], axis=1)  # (bs, num_cdn)
        known_boxes = ops.concat([known_boxes, known_boxes], axis=1)  # (bs, num_cdn, 4)
        # cdn_valids = ops.concat([dn_valids, dn_valids], axis=1)  # (bs, num_cdn)

        if label_noise_ratio > 0:
            p = self.uniform_real(known_labels.shape)
            noise_mask = p < label_noise_ratio * 0.5
            rand_labels = self.uniform_int(known_labels.shape, Tensor(0, ms.int32), Tensor(num_classes, ms.int32))
            known_labels = ops.logical_not(noise_mask).astype(ms.int32) * known_labels \
                           + noise_mask.astype(ms.int32) * rand_labels

        if box_noise_scale > 0:
            known_box_xyxy = box_cxcywh_to_xyxy(known_boxes)
            half_wh = ops.concat([known_boxes[..., 2:] / 2, known_boxes[..., 2:] / 2], axis=-1)

            rand_sign = ops.cast(self.uniform_int(known_boxes.shape, Tensor(0, ms.int32), Tensor(2, ms.int32)) * 2 - 1,
                                 ms.float32)
            rand_part = self.uniform_real(known_boxes.shape)

            rand_part[:, num_dn:] += 1  # for positive 0-1, for negative 1-2
            rand_part *= rand_sign
            known_box_xyxy = known_box_xyxy + ops.mul(rand_part,
                                                      half_wh) * box_noise_scale  # add noise to the rect corner point
            known_box_xyxy = ops.clip_by_value(known_box_xyxy, clip_value_min=Tensor(0.0), clip_value_max=Tensor(1.0))

            known_boxes = box_xyxy_to_cxcywh(known_box_xyxy)

        input_label_embed = self.label_enc(known_labels)  # (bs, num_cdn)
        input_box_embed = inverse_sigmoid(known_boxes)  # (bs, num_cdn, 4)

        # attn mask, if True, means communication is blocked
        attn_mask = ops.zeros((bs, num_cdn + num_query, num_cdn + num_query), ms.bool_)  # all false, means no mask
        # match query cannot see gt query, left bottom gray part of the figure in the original paper
        attn_mask[:, num_cdn:, :num_cdn] = True

        # gt query from different group cannot see each other
        gt_query_coor = ops.stack(ops.meshgrid(ms_np.arange(num_dn), ms_np.arange(num_dn), indexing='ij'),
                                  -1)  # (num_dn, num_dn, 2)
        gt_query_coor = ms_np.tile(gt_query_coor.expand_dims(0), (bs, 1, 1, 1))  # (bs, num_dn, num_dn, 2)
        div = ops.floor_div(gt_query_coor, num_valid_box[:, None, None, None]).astype(ms.float32).min(axis=3).astype(
            ms.int32)  # (bs, num_dn, num_dn)
        left_top_coor = gt_query_coor - ops.expand_dims(div, -1) * num_valid_box[:, None, None, None]
        gt_query_mask = left_top_coor.max(-1) >= num_valid_box[:, None, None]  # (bs, num_dn, num_dn)
        dn_2d_invalid_mask = ops.logical_not(
            ops.logical_and(dn_valids[:, None, :], dn_valids[:, :, None]))  # (bs, num_dn, num_dn)
        gt_query_mask = ops.logical_or(gt_query_mask, dn_2d_invalid_mask)  # set true for the padding area

        temp = ops.concat([gt_query_mask, gt_query_mask], axis=1)  # (bs, num_cdn, num_dn)
        temp = ops.concat([temp, temp], axis=2)  # (bs, num_cdn, num_cdn)
        attn_mask[:, :num_cdn, :num_cdn] = temp

        return input_label_embed, input_box_embed, attn_mask, dn_valids

import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.common.initializer as init
import mindspore.numpy as ms_np

from common.models.layers.attention_deformable import MultiScaleDeformableAttention

from common.models.layers.builder import build_encoder, build_decoder


def linspace(start: Tensor, end: Tensor, num):
    num = int(num)
    res = ops.zeros(num, start.dtype)
    step = ops.div(end - start, num - 1)
    for i in range(num):
        res[i] = start + i * step
    return res


def get_valid_ratio(mask):
    """Get the valid(non-pad) ratios of feature maps of all levels."""
    _, H, W = mask.shape
    h_mask_not, w_mask_not = ~mask[:, :, 0], ~mask[:, 0, :]
    if mask.dtype != ms.float_:
        h_mask_not = ops.cast(h_mask_not, ms.float32)
        w_mask_not = ops.cast(w_mask_not, ms.float32)
    valid_H = h_mask_not.sum(1)
    valid_W = w_mask_not.sum(1)
    valid_ratio_h = valid_H.astype(ms.float32) / H
    valid_ratio_w = valid_W.astype(ms.float32) / W
    valid_ratio = ops.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def multi_2_flatten(multi_level_feats, multi_level_masks, multi_level_pos_embeds, level_embeds):
    feat_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
    ):
        bs, c, h, w = feat.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)

        feat = feat.view(bs, c, -1).transpose(0, 2, 1)  # bs, hw, c
        mask = mask.view(bs, -1)  # bs, hw
        pos_embed = pos_embed.view(bs, c, -1).transpose(0, 2, 1)  # bs, hw, c
        # FIXME level embed dose not support __getitem__ method
        # lvl_pos_embed = pos_embed
        lvl_pos_embed = pos_embed + level_embeds[lvl].view(1, 1, -1)  # multi-scale embed
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        feat_flatten.append(feat)
        mask_flatten.append(mask.int())
    feat_flatten = ops.concat(feat_flatten, 1)
    mask_flatten = ops.concat(mask_flatten, 1).bool()
    lvl_pos_embed_flatten = ops.concat(lvl_pos_embed_flatten, 1)
    # there may be slight difference of ratio values between different level due to of mask quantization
    return feat_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes


class CoDinoTransformer(nn.Cell):
    """Transformer module for DINO

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
        learnt_init_query (bool): whether to learn content query(static) or generate from two-stage proposal(dynamic)
    """

    def __init__(
            self,
            embed_dim=256,
            num_feature_levels=4,
            two_stage_num_proposals=900,
            learnt_init_query=True,
            encoder=None,
            decoder=None,
    ):
        super(CoDinoTransformer, self).__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        # self.embed_dim = self.encoder.embed_dim
        self.embed_dim = embed_dim

        self.level_embeds = ms.Parameter(init.initializer(init.Uniform(), (self.num_feature_levels, self.embed_dim)))
        # self.level_embeds = nn.Embedding(self.num_feature_levels, self.embed_dim)
        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Dense(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm((self.embed_dim,))

        self.init_weights()

    def construct(
            self,
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embed,
            attn_masks,
    ):
        """
        Args:
            multi_level_feats (List[Tensor[bs, embed_dim, h, w]]): list of multi level features from backbone(neck)
            multi_level_masks (List[Tensor[bs, h, w]]):list of masks of multi level features
            multi_level_pos_embeds (List[Tensor[bs, embed_dim, h, w]]):  list of pos_embeds multi level features
            query_embed (List[Tensor[bs, dn_number, embed_dim], Tensor[bs, dn_number, 4]]):
                len of list is 2, initial gt query for dn, including content_query and position query(reference point)
            attn_masks (List[Tensor]): attention map for dn
        """
        feat_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = \
            multi_2_flatten(multi_level_feats, multi_level_masks, multi_level_pos_embeds, self.level_embeds)

        valid_ratios = ops.stack([get_valid_ratio(m) for m in multi_level_masks], 1)  # (bs, num_level, 2)

        # reference_points for deformable-attn, range (H, W), un-normalized, flattened
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)  # (bs sum(hw), nl, 2)

        #  spatial_shapes (bs, sum(hw), num_level, 2)
        # (bs, sum(hw), c)
        memory = self.encoder(
            feat_flatten, # query
            None,  # key
            None,  # value
            lvl_pos_embed_flatten,   # query_pos
            None,  # key_pos
            None,  # attn_masks
            # to mask image input padding area·
            mask_flatten,  # query_key_padding_mask
            None,  # key_padding_mask
            # leave for deformable-attn
            reference_points,
            spatial_shapes,  # multi_scale_args
        )

        # (bs, sum(hw), c); (bs, sum(hw), 4) unsigmoided + valid
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

        reference_points, target, target_unact, topk_coords_unact = self.perform_two_stage(output_memory, output_proposals)

        # add dn part to reference points and targets
        if query_embed[1] is not None:
            reference_points = ops.concat([ops.sigmoid(query_embed[1]), reference_points], 1)
        init_reference_out = reference_points

        if query_embed[0] is not None:
            target = ops.concat([query_embed[0], target], 1)

        inter_states, inter_references = self.decoder(
            query=target,  # (bs, sum(hw)+num_cdn, embed_dims) if dn training else None (bs, sum(hw), embed_dims)
            key=memory,  # bs, sum(hw), embed_dims
            value=memory,  # bs, sum(hw), embed_dims
            query_pos=None,
            key_pos=None,
            attn_masks=attn_masks,  # (bs, sum(hw)+num_cdn, sum(hw)+num_cdn) if dn training else None
            query_key_padding_mask=None,
            # to mask input image padding area, active in cross_attention
            key_padding_mask=mask_flatten,  # bs, sum(hw)
            reference_points=reference_points,  # bs, sum(hw), 4
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,  # (bs, nlvl, 2)
            # to mask the information leakage between gt and matching queries, active in self-attention

        )

        inter_references_out = inter_references
        return (
            inter_states,
            init_reference_out,
            inter_references_out,
            target_unact,
            ops.sigmoid(topk_coords_unact),
            memory
        )

    def init_weights(self):
        for p in self.get_parameters():
            if p.dim() > 1:
                p.set_data(init.initializer(init.XavierUniform(), p.shape, p.dtype))
        for m in self.cells():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        # p_shape, d_type = self.level_embeds.shape, self.level_embeds.dtype
        # self.level_embeds.set_data(init.initializer(init.Uniform(), p_shape, d_type))

    def perform_two_stage(self, output_memory, output_proposals):
        # two-stage
        # hack implementaion, the class_embed of the last layer of transformer.decoder serves for two stage
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided. (bs, sum(hw), 4)

        topk = self.two_stage_num_proposals

        # k must be the last axis
        topk_proposals = ops.top_k(enc_outputs_class.max(-1), topk)[1]  # index (bs, k) , k=num_query
        # extract region proposal boxes
        topk_coords_unact = ops.gather_elements(
            enc_outputs_coord_unact, 1, ms_np.tile(ops.expand_dims(topk_proposals, -1), (1, 1, 4)),
        )  # unsigmoided. (bs, k, 4)
        reference_points = ops.sigmoid(ops.stop_gradient(topk_coords_unact))

        # extract region features

        target_unact = ops.gather_elements(
            output_memory, 1, ms_np.tile(ops.expand_dims(topk_proposals, -1), (1, 1, output_memory.shape[-1]))
        )
        if self.learnt_init_query:
            bs = output_memory.shape[0]
            target = ms_np.tile(self.tgt_embed.embedding_table[None], (bs, 1, 1))
        else:
            target = ops.stop_gradient(target_unact)

        return reference_points, target, target_unact, topk_coords_unact

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """
        Args:
            memory (Tensor[bs, sum(hw), c]): flattened encoder memory
            memory_padding_mask (Tensor[bs, sum(hw)]): padding_mask of memory
            spatial_shapes (List[num_layer, 2]): spatial shapes of multiscale layer
        Returns:
            Tensor[bs, sum(hw), c]: filtered memory
            Tensor[bs, sum(hw), 4]: filtered bbox proposals
        """
        bs, sum_hw, embed_dim = memory.shape
        proposals = []
        _cur = 0  # start index of the ith layer
        for lvl, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)
            mask_flatten_ = memory_padding_mask[:, _cur: (_cur + H * W)].view(bs, H, W, 1)
            h_mask_not, w_mask_not = ~mask_flatten_[:, :, 0, 0], ~mask_flatten_[:, 0, :, 0]
            if mask_flatten_.dtype != ms.float_:
                h_mask_not = ops.cast(h_mask_not, ms.float32)
                w_mask_not = ops.cast(w_mask_not, ms.float32)
            valid_h = h_mask_not.sum(1)  # (bs,)
            valid_w = w_mask_not.sum(1)  # (bs,)

            grid_y, grid_x = ops.meshgrid(
                linspace(Tensor(0, dtype=ms.float32), Tensor(H - 1, dtype=ms.float32), H),
                linspace(Tensor(0, dtype=ms.float32), Tensor(W - 1, dtype=ms.float32), W), indexing='ij')  # (h, w)

            grid = ops.concat([grid_x.expand_dims(-1), grid_y.expand_dims(-1)], -1)  # (h ,w, 2)

            scale = ops.concat([valid_w.expand_dims(-1), valid_h.expand_dims(-1)], 1).view(bs, 1, 1, 2)
            # (bs, h ,w, 2), normalized to valid range
            grid = (grid.expand_dims(0).broadcast_to((bs, -1, -1, -1)) + 0.5) / scale
            hw = ms_np.ones_like(grid) * 0.05 * (2.0 ** lvl)  # preset wh, larger wh for higher level
            proposal = ops.concat((grid, hw), -1).view(bs, -1, 4)  # (bs, hw, 4)
            proposals.append(proposal)
            _cur += H * W

        # filter proposal
        output_proposals = ops.concat(proposals, 1)  # (bs, sum(hw), 4)
        # filter those whose centers are too close to the margin or wh too small or too large
        
        output_proposals_valid = ops.logical_and(output_proposals > 0.01, output_proposals < 0.99).all(
            -1, keep_dims=True
        )  # (bs, sum(hw), 1)
        output_proposals = ops.log(output_proposals / (1 - output_proposals))  # unsigmoid
        # filter proposal in the padding area
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.expand_dims(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # also mask memory in the filtered position
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.expand_dims(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))  # channel-wise mlp
        return output_memory, output_proposals

    def get_reference_points(self, spatial_shapes, valid_ratios):
        """Get the reference points of every pixel position of every level used in decoder.

        Args:
            spatial_shapes (List): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = ops.meshgrid(
                linspace(Tensor(0.5, dtype=ms.float32),
                         Tensor(H - 0.5, dtype=ms.float32), H),
                linspace(Tensor(0.5, dtype=ms.float32),
                         Tensor(W - 0.5, dtype=ms.float32), W),
                indexing='ij'
            )  # (h, w)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)  # (bs, hw)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)  # (bs, hw)
            ref = ops.stack((ref_x, ref_y), -1)  # (bs, hw, 2)
            reference_points_list.append(ref)
        reference_points = ops.concat(reference_points_list, 1)  # (bs, sum(hw), 2)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # (bs sum(hw), nl, 2)
        return reference_points

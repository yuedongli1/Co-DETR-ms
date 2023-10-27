import copy
from typing import List, Tuple

from mindspore import nn, Tensor, ops
import mindspore.numpy as ms_np

from common.models.layers.attention import MultiheadAttention
from common.models.layers.mlp import FFN, MLP
from common.models.layers.position_embedding import get_sine_pos_embed
from common.utils.misc import inverse_sigmoid
from common.models.layers.attention_deformable import MultiScaleDeformableAttention


class BaseTransformerLayer(nn.Cell):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.

    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn (list[nn.Cell] | nn.Cell): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(
        self,
        attn: List[nn.Cell],
        ffn: nn.Cell,
        norm: nn.Cell,
        operation_order: tuple = None,
        attn_type: tuple = None,
    ):
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({"self_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        assert len(attn) == num_attn, (
            f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
            f"is not consistent with the number of attention in "
            f"operation_order {operation_order}"
        )
        assert len(attn_type) == num_attn

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.CellList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1
        self.attn_type = attn_type
        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        if not isinstance(ffn, nn.CellList):
            self.ffns = nn.CellList()
            num_ffns = operation_order.count("ffn")
            for _ in range(num_ffns):
                self.ffns.append(copy.deepcopy(ffn))
        else:
            self.ffns = ffn

        # count norm nums
        if not isinstance(norm, nn.CellList):
            self.norms = nn.CellList()
            num_norms = operation_order.count("norm")
            for _ in range(num_norms):
                self.norms.append(copy.deepcopy(norm))
        else:
            self.norms = norm

    # @ms.ms_function
    def construct(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        key_pos: Tensor = None,
        attn_masks: List[Tensor] = None,
        query_key_padding_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        reference_points: Tensor = None,
        spatial_shapes: Tuple = None,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (Tensor): The position embedding for `query`.
                Default: None.
            key_pos (Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
            multi_scale_args (Tuple[Tensor]): tuple that contains two args of multi-scale attention,
             namely, reference_points, spatial_shapes, level_start_index
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                if self.attn_type[attn_index] == 'MultiheadAttention':
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],  # None in encoder, active in decoder
                        key_padding_mask=query_key_padding_mask,  # None in decoder, active in encoder
                    )
                elif self.attn_type[attn_index] == 'MultiScaleDeformableAttention':
                    query = self.attentions[attn_index](
                        query,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_padding_mask=query_key_padding_mask,  # None in decoder, active in encoder
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                    )
                else:
                    raise NotImplementedError(f'not supported self-attetion type [{type(self.attentions[attn_index])}]')

                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                if self.attn_type[attn_index] == 'MultiheadAttention':
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                    )
                elif self.attn_type[attn_index] == 'MultiScaleDeformableAttention':
                    query = self.attentions[attn_index](
                        query,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_padding_mask=key_padding_mask,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                    )
                else:
                    raise NotImplementedError(f'not supported cross-attetion type [{type(self.attentions[attn_index])}]')

                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class DINOTransformerEncoder(nn.Cell):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            feedforward_dim: int = 1024,
            attn_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            num_layers: int = 6,
            post_norm: bool = False,
            num_feature_levels: int = 4,
    ):
        super(DINOTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.CellList(list(
                        BaseTransformerLayer(
                                attn=nn.CellList(list(
                                        MultiScaleDeformableAttention(
                                        embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        dropout=attn_dropout,
                                        num_levels=num_feature_levels,)
                                    for _ in range(1))),
                                ffn=nn.CellList(list(
                                        FFN(
                                        embed_dim=embed_dim,
                                        feedforward_dim=feedforward_dim,
                                        output_dim=embed_dim,
                                        num_fcs=2,
                                        ffn_drop=ffn_dropout,)
                                    for _ in range(1))),
                                norm=nn.CellList(list([nn.LayerNorm((embed_dim,)) for _ in range(2)])),
                                operation_order=("self_attn", "norm", "ffn", "norm"),
                                attn_type=('MultiScaleDeformableAttention',),
                            )
                    for _ in range(num_layers))
        )

        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None
        self.enable_tuple_broaden = True

    def construct(
            self,
            query,
            key,
            value,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            reference_points=None,
            spatial_shapes=None,
    ):

        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DINOTransformerDecoder(nn.Cell):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            feedforward_dim: int = 1024,
            attn_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            num_layers: int = 6,
            return_intermediate: bool = True,
            num_feature_levels: int = 4,
            look_forward_twice=True,
    ):
        super(DINOTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.CellList(list(
                        BaseTransformerLayer(
                            attn=nn.CellList([
                                MultiheadAttention(
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    attn_drop=attn_dropout,
                                ),
                                MultiScaleDeformableAttention(
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=attn_dropout,
                                    num_levels=num_feature_levels,
                                ),
                            ]),
                            ffn=nn.CellList(list(
                                    FFN(
                                        embed_dim=embed_dim,
                                        feedforward_dim=feedforward_dim,
                                        output_dim=embed_dim,
                                        ffn_drop=ffn_dropout,)
                            for _ in range(1))),
                            norm=nn.CellList(list(nn.LayerNorm((embed_dim,)) for _ in range (3))),
                            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                            attn_type=('MultiheadAttention', 'MultiScaleDeformableAttention')
                        )
                    for _ in range(num_layers)
        ))
        self.return_intermediate = return_intermediate

        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        # values of bbox_embed and class_embed are set in outer class DINO
        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm((embed_dim,))
        self.enable_tuple_broaden = True

    def construct(
            self,
            query,
            key,
            value,
            query_pos=None,  # generated in construct from reference points
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            reference_points=None,  # (bs, num_query, 4). normalized to the valid image area
            spatial_shapes=None,
            valid_ratios=None,  # (bs, num_level, 2)  non-pad area ratio in h and w direction
    ):
        """
            Returns:
                output (Tensor[bs, num_query, embed_dim]): output of each layer
                reference_points (Tensor[bs, num_query, 4|2]): output reference point of each layer
            """
        assert query_pos is None
        output = query
        bs, num_queries, _ = output.shape
        if reference_points.ndim == 2:
            reference_points = ms_np.tile(reference_points.unsqueeze(0), (bs, 1, 1))  # bs, num_query, 4

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                # normalized to the whole image (including padding area)
                # (bs, num_query, num_level, 1) * (bs, 1, num_level, 4) -> (bs, num_query, num_level, 4)
                reference_points_input = (
                        reference_points[:, :, None]
                        * ops.concat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            # reference is almost the same for all level, pick the first one
            # TODO to compact with len(reference_points) == 2
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])  # (bs, num_query, 2*embed_dim)
            query_pos = self.ref_point_head(query_sine_embed)  # (bs, num_query, embed_dim)

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                # query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,  # list of masks for all attention layers
                query_key_padding_mask=query_key_padding_mask,  # key padding masks for self attention
                key_padding_mask=key_padding_mask,  # key padding masks for cross attention
                reference_points=reference_points_input,  # (bs, num_query, num_level, 4)
                spatial_shapes=spatial_shapes,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = ops.sigmoid(new_reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)

                    new_reference_points = ops.sigmoid(new_reference_points)
                reference_points = ops.stop_gradient(new_reference_points)
            else:
                raise NotImplementedError('box_embed must be defined')

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    # both delta and refer will be supervised
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return ops.stack(intermediate), ops.stack(intermediate_reference_points)

        return output, reference_points
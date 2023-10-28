import copy
import math
from typing import List

import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.common.initializer as init

from common.models.layers.mlp import MLP
from common.utils.misc import inverse_sigmoid
from common.utils.torch_converter import init_like_torch
from common.models.layers.builder import build_position_embedding

from projects.transformer_builder import build_transformer


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
                 transformer=None,
                 position_embedding=None,
                 ):
        super().__init__()
        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = build_transformer(transformer)

        # define position embedding module
        self.position_embedding = build_position_embedding(position_embedding)

        # define classification head and box head
        self.class_embed = nn.Dense(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # de-noising
        self.label_enc = nn.Embedding(num_classes, embed_dim)

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

        for bbox_embed_layer in self.bbox_embed:
            bias_init_data = bbox_embed_layer.layers[-1].bias.data
            bias_init_data[2:] = Tensor(-0.0)
            p_type, d_type = bbox_embed_layer.layers[-1].bias.shape, bbox_embed_layer.layers[-1].bias.dtype
            bbox_embed_layer.layers[-1].bias.set_data(init.initializer(bias_init_data, p_type, d_type))

        # operator
        self.uniform_real = ops.UniformReal()
        self.uniform_int = ops.UniformInt()

    def construct(self, multi_level_feats, images, img_masks, targets=None):
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
            l_mask = ops.cast(l_mask, ms.bool_)
            multi_level_masks.append(l_mask)
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        input_query_label, input_query_bbox, attn_mask, dn_valids = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (inter_states, init_reference, inter_reference, enc_state, enc_reference) = self.transformer(
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

        return outputs_class[-1], outputs_coord[-1]

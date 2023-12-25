import torch
import mindspore as ms
from mindspore import Tensor

co_dino = torch.load('co_dino_5scale_swin_large_16e_o365tococo.pth', map_location=torch.device('cpu'))
stacte_dict = co_dino['state_dict']

ms_ckpt = []
for k, v in stacte_dict.items():
    if 'num_batches_tracked' in k:
        continue
    if 'rpn_head' in k or 'roi_head' in k or 'bbox_head' in k:
        continue
    if 'bn' in k and 'running_mean' in k:
        k = k.replace('running_mean', 'moving_mean')
    if 'bn' in k and 'running_var' in k:
        k = k.replace('running_var', 'moving_variance')
    if ('norm' in k or 'bn' in k) and 'weight' in k:
        k = k.replace('weight', 'gamma')
    if ('norm' in k or 'bn' in k) and 'bias' in k:
        k = k.replace('bias', 'beta')
    if 'backbone' in k and 'downsample.1' in k:
        if 'running_mean' in k:
            k = k.replace('running_mean', 'moving_mean')
        if 'running_var' in k:
            k = k.replace('running_var', 'moving_variance')
        if 'weight' in k:
            k = k.replace('weight', 'gamma')
        if 'bias' in k:
            k = k.replace('bias', 'beta')
    if 'neck' in k and 'convs' in k and 'gn' in k:
        if 'gn.weight' in k:
            k = k.replace('gn.weight', 'norm.gamma')
        if 'gn.bias' in k:
            k = k.replace('gn.bias', 'norm.beta')
    if 'query_head.transformer' in k and 'layers' in k and 'norms' in k:
        if 'weight' in k:
            k = k.replace('weight', 'gamma')
        if 'bias' in k:
            k = k.replace('bias', 'beta')
    if 'query_head.transformer.decoder.layers' in k and 'attentions.0.attn' in k:
        if 'in_proj_weight' in k:
            k1 = k.replace('attn.in_proj_weight', 'in_projs.0.weight')
            k2 = k.replace('attn.in_proj_weight', 'in_projs.1.weight')
            k3 = k.replace('attn.in_proj_weight', 'in_projs.2.weight')
            ms_ckpt.append({'name': k1, 'data': Tensor(torch.split(v, 256)[0].numpy())})
            ms_ckpt.append({'name': k2, 'data': Tensor(torch.split(v, 256)[1].numpy())})
            ms_ckpt.append({'name': k3, 'data': Tensor(torch.split(v, 256)[2].numpy())})
            continue
        if 'in_proj_bias' in k:
            k1 = k.replace('attn.in_proj_bias', 'in_projs.0.bias')
            k2 = k.replace('attn.in_proj_bias', 'in_projs.1.bias')
            k3 = k.replace('attn.in_proj_bias', 'in_projs.2.bias')
            ms_ckpt.append({'name': k1, 'data': Tensor(torch.split(v, 256)[0].numpy())})
            ms_ckpt.append({'name': k2, 'data': Tensor(torch.split(v, 256)[1].numpy())})
            ms_ckpt.append({'name': k3, 'data': Tensor(torch.split(v, 256)[2].numpy())})
            continue
        if 'out_proj' in k:
            k = k.replace('attn.out_proj', 'out_proj')
    if 'query_head.transformer.decoder.ref_point_head' in k:
        if '0.weight' in k:
            k = k.replace('0.weight', 'layers.0.weight')
        if '0.bias' in k:
            k = k.replace('0.bias', 'layers.0.bias')
        if '2.weight' in k:
            k = k.replace('2.weight', 'layers.1.weight')
        if '2.bias' in k:
            k = k.replace('2.bias', 'layers.1.bias')
    if 'query_head.transformer.decoder.norm.weight' in k:
        k = k.replace('weight', 'gamma')
    if 'query_head.transformer.decoder.norm.bias' in k:
        k = k.replace('bias', 'beta')
    if 'query_head.transformer.enc_output_norm.weight' in k:
        k = k.replace('weight', 'gamma')
    if 'query_head.transformer.enc_output_norm.bias' in k:
        k = k.replace('bias', 'beta')
    if 'query_head.transformer.query_embed.weight' in k:
        k = k.replace('query_embed.weight', 'tgt_embed.embedding_table')
    if 'query_head.label_embedding.weight' in k:
        k = k.replace('label_embedding.weight', 'label_enc.embedding_table')
    if 'query_head.transformer.aux_pos_trans' in k:
        continue
    if 'query_head.cls_branches' in k:
        k = k.replace('cls_branches', 'transformer.decoder.class_embed')
    if 'query_head.reg_branches' in k:
        k = k.replace('reg_branches', 'transformer.decoder.bbox_embed')
        if '0.weight' in k:
            k = k.replace('0.weight', 'layers.0.weight')
        if '0.bias' in k:
            k = k.replace('0.bias', 'layers.0.bias')
        if '2.weight' in k:
            k = k.replace('2.weight', 'layers.1.weight')
        if '2.bias' in k:
            k = k.replace('2.bias', 'layers.1.bias')
        if '4.weight' in k:
            k = k.replace('4.weight', 'layers.2.weight')
        if '4.bias' in k:
            k = k.replace('4.bias', 'layers.2.bias')
    if 'query_head.downsample.1.weight' in k:
        k = k.replace('weight', 'gamma')
    if 'query_head.downsample.1.bias' in k:
        k = k.replace('bias', 'beta')
    # if 'bbox_head' in k:
    #     k = k.replace('bbox_head.0', 'bbox_head')
    #     if 'gn.weight' in k:
    #         k = k.replace('gn.weight', 'norm.gamma')
    #     if 'gn.bias' in k:
    #         k = k.replace('gn.bias', 'norm.beta')
    ms_ckpt.append({'name': k, 'data': Tensor(v.numpy())})

ms.save_checkpoint(ms_ckpt, './co_dino_5scale_swin_large_torch.ckpt')

print('done')

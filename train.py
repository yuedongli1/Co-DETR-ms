import os
import yaml
from datetime import datetime
import numpy as np

from mindspore.communication import init, get_rank, get_group_size

import mindspore as ms
from mindspore import nn, context, ParallelMode, ops, Tensor
from mindspore.amp import DynamicLossScaler

from common.dataset.dataset import create_mindrecord, create_detr_dataset
from common.utils.utils import set_seed
from config import config
from projects.co_detr import build_co_detr

if __name__ == '__main__':
    # set context, seed
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', pynative_synchronize=True)
    set_seed(0)

    if config.distributed:
        print('distributed training start')
        init()
        rank = get_rank()
        device_num = get_group_size()
        print(f'current rank {rank}/{device_num}')
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, gradients_mean=True,
                                          parallel_mode=ParallelMode.DATA_PARALLEL)
    else:
        rank = 0
        device_num = 1

    # create dataset
    mindrecord_file = create_mindrecord(config, rank, "DETR.mindrecord", True)
    dataset = create_detr_dataset(config, mindrecord_file, batch_size=config.batch_size,
                                  device_num=device_num, rank_id=rank,
                                  num_parallel_workers=config.num_parallel_workers,
                                  python_multiprocessing=config.python_multiprocessing,
                                  is_training=True)
    ds_size = dataset.get_dataset_size()

    # load pretrained model, only load backbone
    config_file = './projects/configs/co_dino/co_dino_5scal_r50_1x_coco.yaml'
    with open(config_file, 'r') as ifs:
        cfg = yaml.safe_load(ifs)
    co_detr = build_co_detr(cfg['model'])

    pretrain_path = "./co_dino_query_bbox_head.ckpt"
    ms.load_checkpoint(pretrain_path, co_detr)
    print(f'successfully load checkpoint from {pretrain_path}')

    epoch_num = 12

    # create optimizer
    lr = 2e-4  # normal learning rate
    lr_backbone = 2e-5  # slower learning rate for pretrained backbone
    lr_drop = epoch_num - 1
    weight_decay = 1e-4
    lr_not_backbone = nn.piecewise_constant_lr(
        [ds_size * lr_drop, ds_size * epoch_num], [lr, lr * 0.1])
    lr_backbone = nn.piecewise_constant_lr(
        [ds_size * lr_drop, ds_size * epoch_num], [lr_backbone, lr_backbone * 0.1])

    backbone_params = list(filter(lambda x: 'backbone' in x.name, co_detr.trainable_params()))
    not_backbone_params = list(filter(lambda x: 'backbone' not in x.name, co_detr.trainable_params()))
    param_dicts = [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': not_backbone_params, 'lr': lr_not_backbone, 'weight_decay': weight_decay}
    ]
    optimizer = nn.AdamWeightDecay(param_dicts)

    # # set mix precision
    # dino.to_float(ms.float16)
    # for _, cell in dino.cells_and_names():
    #     if isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, HungarianMatcher)):
    #         cell.to_float(ms.float32)

    # create model with loss scale
    co_detr.set_train(True)
    # scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
    # model = TrainOneStepWithGradClipLossScaleCell(co_detr, optimizer, scale_sense, grad_clip=True, clip_value=0.1)
    # model = nn.TrainOneStepWithLossScaleCell(dino, optimizer, scale_sense)
    # model = nn.TrainOneStepCell(dino, optimizer)

    scaler = DynamicLossScaler(scale_value=2 ** 12, scale_factor=2, scale_window=1000)

    def forward_func(
            image, mask, labels, boxes_xywhn, boxes_xyxy, valid, dn_valid, img_shape, ori_shape):
        loss = co_detr(
            image, mask, labels, boxes_xywhn, boxes_xyxy, valid, dn_valid, img_shape, ori_shape)
        return scaler.scale(loss)

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters)

    # training loop
    log_loss_step = 1
    summary_loss_step = 1
    start_time = datetime.now()
    last_step_time = start_time
    print(f'prepare finished, entering training loop')
    for e_id in range(epoch_num):
        for s_id, in_data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
            # image = Tensor(np.load('img.npy'))
            # h, w = image.shape[-2:]
            # mask = ops.zeros((1, h, w))
            # labels_real = Tensor(np.load('gt_labels.npy'))
            # labels = ops.full((1, 100), -1)
            # labels[:, :4] = labels_real
            #
            # boxes_real = np.load('gt_bboxes.npy')
            # boxes_xyxy = ops.full((1, 100, 4), 0., dtype=ms.float32)
            # boxes_xyxy[:, :4, :] = Tensor(boxes_real)
            #
            # boxes_real = box_xyxy_to_cxcywh(boxes_real)
            # boxes_real[:, [0, 2]] /= w
            # boxes_real[:, [1, 3]] /= h
            # boxes_real = Tensor(boxes_real)
            # boxes_xywhn = ops.full((1, 100, 4), 0., dtype=ms.float32)
            # boxes_xywhn[:, :4, :] = boxes_real
            #
            # valid = ops.zeros((1, 100,))
            # valid[:, :4] = 1
            # dn_valid = ops.ones((1, 100), dtype=ms.bool_)
            # img_shape = (image.shape[-2:],)
            # ori_shape = (image.shape[-2:],)
            #
            # feat0 = Tensor(np.load('feat0.npy'))
            # feat1 = Tensor(np.load('feat1.npy'))
            # feat2 = Tensor(np.load('feat2.npy'))
            # feat3 = Tensor(np.load('feat3.npy'))
            # feat4 = Tensor(np.load('feat4.npy'))
            # feat5 = Tensor(np.load('feat5.npy'))

            # image, img_mask(1 for padl), gt_box, gt_label, gt_valid(True for valid)
            image = Tensor(in_data['image'])
            mask = Tensor(in_data['mask'])
            labels = Tensor(in_data['labels'])
            boxes_xywhn = Tensor(in_data['boxes_xywhn'])
            boxes_xyxy = Tensor(in_data['boxes_xyxy'])
            valid = Tensor(in_data['valid'])
            dn_valid = Tensor(in_data['dn_valid'])
            
            img_shape_tuple = ()
            ori_shape_tuple = ()
            for img_shape in in_data['img_shape']:
               img_shape_tuple += (tuple(img_shape.tolist()),)
            for ori_shape in in_data['ori_shape']:
               ori_shape_tuple += (tuple(ori_shape.tolist()),)

            loss, grads = grad_fn(
                image, mask, labels, boxes_xywhn, boxes_xyxy,
                                  valid, dn_valid, img_shape_tuple, ori_shape_tuple)

            # loss, grads = grad_fn(feat0, feat1, feat2, feat3, feat4, feat5,
            #                     image, mask, labels, boxes_xywhn, boxes_xyxy, valid, dn_valid, img_shape, ori_shape)

            grads = ops.clip_by_global_norm(grads, clip_norm=0.1)
            loss = ops.depend(loss, optimizer(grads))
            loss = scaler.unscale(loss)

            # loss, _, _ = model(*input_data)

            # put on screen
            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d - %H:%M:%S")
            past_time = (now - start_time)
            sec_per_step = (now - last_step_time).total_seconds()
            if s_id % log_loss_step == 0:
                print(f"[{now_str}] epoch[{e_id}/{epoch_num}] step[{s_id}/{ds_size}], "
                      f"loss[{loss.asnumpy():.2f}], cost-time[{past_time}], sec-per-step[{sec_per_step:.1f}s]")

            # record in summary for mindinsight
            global_s_id = s_id + e_id * ds_size
            last_step_time = now

        # save checkpoint every epoch
        print(f'saving checkpoint for epoch {e_id}')
        ckpt_path = os.path.join(config.output_dir, f'dino_epoch{e_id:03d}_rank{rank}.ckpt')
        ms.save_checkpoint(co_detr, ckpt_path)

    print(f'finish training for dino')

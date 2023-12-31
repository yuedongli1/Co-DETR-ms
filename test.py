import os
import time
from typing import Dict

import cv2
import mindspore as ms
# ms.set_context(save_graphs=True, save_graphs_path='./')
import numpy as np
from mindspore import ops, Tensor, set_seed
import mindspore.numpy as ms_np
from pycocotools.coco import COCO
from tqdm import tqdm
import yaml

from common.dataset.coco_eval import CocoEvaluator
from common.dataset.dataset import create_mindrecord, create_detr_dataset, coco_clsid_to_catid, \
    coco_id_dict
from common.core.bbox.transforms import box_cxcywh_to_xyxy, box_scale
from config import config
from projects.co_detr import build_co_detr
from common.utils.misc import _nms


def select_from_prediction(box_cls, box_pred, ori_size, num_select=300, test_cfg=None):
    nms_type = test_cfg.get('nms_type', None)
    iou_thres = test_cfg.get('iou_threshold', None)
    bs, num_query, num_class = box_cls.shape
    if nms_type:
        num_select = num_query
    # box_cls.shape: 1, 300, 80
    # box_pred.shape: 1, 300, 4
    prob = box_cls.sigmoid()
    topk_values, topk_indexes_full = ops.topk(prob.view(bs, -1), num_select)
    # (bs, num_select)
    topk_boxes_ind = ops.div(topk_indexes_full.astype(ms.float32),
                             num_class, rounding_mode="floor").astype(ms.int32)

    boxes = ops.gather_elements(box_pred, 1, ms_np.tile(topk_boxes_ind.unsqueeze(-1), (1, 1, 4)))  # (bs,num_eval,4)

    boxes_xyxy = box_cxcywh_to_xyxy(boxes)  # (bs, num_select, 4)
    boxes_xyxy_scaled = box_scale(boxes_xyxy, scale=ori_size)  # (bs, num_select, 4)

    labels = (topk_indexes_full % num_class)  # (bs, num_select)
    scores = topk_values

    # if nms_type:
    #     scores = scores[0].asnumpy()
    #     boxes_xyxy_scaled = boxes_xyxy_scaled.asnumpy()
    #     labels = labels[0].asnumpy()
    #     max_coordinate = boxes_xyxy_scaled.max()
    #     offsets = labels * (max_coordinate + 1)
    #     boxes_for_nms = (boxes_xyxy_scaled + offsets[:, None])
    #     i = _nms(boxes_for_nms, scores, iou_thres, nms_type)  # NMS for per sample
    #     boxes_xyxy_scaled = boxes_xyxy_scaled[i]
    #     labels = labels[i]
    #     scores = scores[None, :]
    #     labels = labels[None, :]
    #     boxes_xyxy_scaled = boxes_xyxy_scaled[None, :]

    return scores, labels, boxes_xyxy_scaled


def inference(model, image, mask, ori_size, num_select=300, test_cfg=None):
    # image, mask, image_id, ori_size = data
    output = model(image, mask)
    box_cls, box_pred = output
    assert len(box_cls) == len(image)
    scores, labels, boxes = select_from_prediction(box_cls, box_pred, ori_size, num_select, test_cfg)

    return scores, labels, boxes
    # return output


def visualize(pred_dict: Dict, coco_gt: COCO, save_dir, raw_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_id, res in pred_dict.items():
        img_file_info = coco_gt.loadImgs(img_id)[0]
        save_path = os.path.join(save_dir, img_file_info['file_name'])
        raw_path = os.path.join(raw_dir, img_file_info['file_name'])
        choose = res['scores'] > 0.3
        labels = ops.masked_select(res['labels'], choose).asnumpy()
        boxes = ops.masked_select(res['boxes'], choose.unsqueeze(-1)).asnumpy().reshape(-1, 4)
        scores = ops.masked_select(res['scores'], choose).asnumpy()
        image = cv2.imread(raw_path)

        for s, l, b in zip(scores, labels, boxes):
            x1, y1, x2, y2 = b
            class_name = coco_id_dict[l]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(image, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann['bbox']
            cat_id = ann['category_id']
            class_name = coco_gt.cats[cat_id]['name']
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(image, class_name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite(save_path, image)


def coco_evaluate(model, eval_dateset, eval_anno_path, test_cfg, save_dir, raw_dir):
    # coco evaluator
    coco_gt = COCO(eval_anno_path)
    coco_evaluator = CocoEvaluator(coco_gt, ('bbox', ))

    # inference
    start_time = time.time()
    num_select = 300
    ds_size = dataset.get_dataset_size()
    i = 0
    current_start_time = time.time()
    # profiler = ms.Profiler(start_profile=False)
    for i, data in enumerate(tqdm(eval_dateset.create_dict_iterator(), total=ds_size, desc=f'inferring...')):
        # if i == 1:
        #     profiler.start()
        # if i == 4:
        #     profiler.stop()
        #     break
        image_id = data['image_id'].asnumpy()  # (bs, )
        image = data['image']  # (bs, c, h, w)
        mask = data['mask']  # (bs, h, w)
        size_wh = data['ori_size'][:, ::-1]  # (bs, 2), in wh order
        scores, labels, boxes = inference(model, image, mask, size_wh, num_select, test_cfg)
        # output = inference(model, image, mask, size_wh, num_select, test_cfg)
        cat_ids = Tensor(np.vectorize(coco_clsid_to_catid.get)(labels.asnumpy()))
        res = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, cat_ids, boxes)]
        img_res = {int(idx): output for idx, output in zip(image_id, res)}
        coco_evaluator.update(img_res)
        # visualize(img_res, coco_gt, save_dir, raw_dir)
        print(f'current image cost time: {time.time() - current_start_time}s', )
        current_start_time = time.time()
    # profiler.analyse()

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    print(coco_evaluator.coco_eval.get('bbox').stats)
    print(f'cost time: {time.time() - start_time}s', )
    print("\n========================================\n")


if __name__ == '__main__':
    # set context
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend',
                   pynative_synchronize=False, max_call_depth=2000)
    rank = 0
    device_num = 1
    set_seed(0)

    config_file = './projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.yaml'
    # config_file = './projects/configs/co_dino/co_dino_5scal_r50_1x_coco.yaml'
    with open(config_file, 'r') as ifs:
        cfg = yaml.safe_load(ifs)
    eval_model = build_co_detr(cfg['model'])
    eval_model.set_train(False)

    model_path = './co_dino_5scale_swin_large.ckpt'
    # model_path = './co_dino_query_bbox_head.ckpt'
    ms.load_checkpoint(model_path, eval_model)

    ms.amp.auto_mixed_precision(eval_model, amp_level='O0')

    # evaluate coco
    mindrecord_file = create_mindrecord(config, rank, "DETR.mindrecord.eval", False)
    dataset = create_detr_dataset(config, mindrecord_file, batch_size=1,
                                  device_num=device_num, rank_id=rank,
                                  # num_parallel_workers=config.num_parallel_workers,
                                  num_parallel_workers=1,
                                  python_multiprocessing=config.python_multiprocessing,
                                  is_training=False)

    anno_json = os.path.join(config.coco_path, "annotations/instances_val2017.json")
    vis_save_dir = os.path.join(config.coco_path, 'val2017_vis')
    raw_img_dir = os.path.join(config.coco_path, 'val2017')
    coco_evaluate(eval_model, dataset, anno_json, cfg['test_cfg'], vis_save_dir, raw_img_dir)

from functools import partial
from six.moves import map, zip

import mindspore as ms
from mindspore import Tensor, ops
import time
import numpy as np


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    Args:
        x (Tensor) : tensor within range 0,1
        eps (float) :
    """
    x = ops.clip_by_value(x, clip_value_min=0, clip_value_max=1)
    x1 = ops.clip_by_value(x, clip_value_min=Tensor(eps))
    x2 = ops.clip_by_value(1 - x, clip_value_min=Tensor(eps))
    return ops.log(x1 / x2)


def replace_invalid(inputs, v_mask, value, dtype=ms.int32):
    """
    replace value of invalid index to the given value
    Args:
        inputs (Tensor)： inputs tensor
        v_mask (Tensor): mask that indicates valid index
        value (int, float): value to replace
        dtype (ms.number): output date type
    """
    res = inputs * v_mask.astype(dtype)
    res += ops.logical_not(v_mask).astype(dtype) * value  # replace invalid with given value
    return res.astype(dtype)


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = ops.full((count,), fill, dtype=data.dtype)
        ret = ops.stop_gradient(ret)
        ret[inds.type(ms.bool_)] = data
    else:
        new_size = (count, ) + data.shape[1:]
        ret = ops.full(new_size, fill, dtype=data.dtype)
        ret = ops.stop_gradient(ret)
        ret[inds.type(ms.bool_), :] = data
    return ret


def _nms(xyxys, scores, threshold, nms_type, min_score=1e-3):
    """Calculate NMS"""
    x1 = xyxys[:, 0]
    y1 = xyxys[:, 1]
    x2 = xyxys[:, 2]
    y2 = xyxys[:, 3]
    scores = scores.copy()
    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    reserved_boxes = []
    while order.size > 0:
        i = order[0]
        reserved_boxes.append(i)
        max_x1 = np.maximum(x1[i], x1[order[1:]])
        max_y1 = np.maximum(y1[i], y1[order[1:]])
        min_x2 = np.minimum(x2[i], x2[order[1:]])
        min_y2 = np.minimum(y2[i], y2[order[1:]])

        # intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
        # intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
        intersect_w = np.maximum(0.0, min_x2 - max_x1)
        intersect_h = np.maximum(0.0, min_y2 - max_y1)
        intersect_area = intersect_w * intersect_h

        ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area + 1e-6)
        indexes = np.where(ovr >= threshold)[0]
        if not indexes:
            continue
        if nms_type == 'soft_nms':
            scores[order[indexes + 1]] *= (1 - ovr[indexes])
        else:
            scores[order[indexes + 1]] = 0
        order = order[(scores >= min_score) + 1]
    return np.array(reserved_boxes)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def _box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 ([N, 4])
        box2 ([M, 4])
    Returns:
        iou ([N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0, None).prod(2)
    )
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

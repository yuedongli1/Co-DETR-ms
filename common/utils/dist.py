from mindspore import ops
from mindspore.ops import ReduceOp
from mindspore.communication.management import get_group_size


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different NPUs."""
    tensor = tensor.copy()
    return ops.AllReduce(ReduceOp.SUM)(ops.div(tensor, get_group_size()))

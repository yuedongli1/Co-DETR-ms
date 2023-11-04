import random

import mindspore as ms
import numpy


def set_seed(seed):
    ms.set_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

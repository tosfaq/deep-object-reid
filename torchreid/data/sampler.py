# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, division
import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import RandomSampler, SequentialSampler

AVAI_SAMPLERS = ['SequentialSampler', 'RandomSampler']


def build_train_sampler(data_source, train_sampler, **kwargs):
    """Builds a training sampler.
    """
    assert train_sampler in AVAI_SAMPLERS, \
        'train_sampler must be one of {}, but got {}'.format(AVAI_SAMPLERS, train_sampler)

    if train_sampler == 'SequentialSampler':
        sampler = SequentialSampler(data_source)
    elif train_sampler == 'RandomSampler':
        sampler = RandomSampler(data_source)
    else:
        raise ValueError('Unknown sampler: {}'.format(train_sampler))

    return sampler

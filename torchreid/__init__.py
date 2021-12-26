# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, print_function

try:
    import nncf
except ImportError:
    import warnings
    warnings.warn("NNCF was not found. Model optimization options will not be available.")

from torchreid import data, engine, losses, metrics, models, ops, optim, utils
from .version import __version__

__author__ = 'Kaiyang Zhou'
__homepage__ = 'https://kaiyangzhou.github.io/'
__description__ = 'Deep learning person re-identification in PyTorch'
__url__ = 'https://github.com/KaiyangZhou/deep-person-reid'

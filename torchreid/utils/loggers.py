# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2017 Tong Xiao
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=consider-using-with

from __future__ import absolute_import
from datetime import datetime
import os
import os.path as osp
import sys

from .tools import mkdir_if_missing

__all__ = ['Logger']


class Logger:
    """Writes console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os
       >>> import os.path as osp
       >>> from torchreid.utils import Logger
       >>> save_dir = 'log/resnet50-softmax-market1501'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(args.save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.was_cr = True
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        if self.was_cr:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S|')
        else:
            timestamp = ''
        self.was_cr = msg.endswith('\n')

        self.console.write(timestamp + msg)
        if self.file is not None:
            self.file.write(timestamp + msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

from __future__ import division, print_function, absolute_import
import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
import PIL
import torch
from PIL import Image

__all__ = [
    'mkdir_if_missing', 'check_isfile', 'read_json', 'write_json',
    'set_random_seed', 'download_url', 'read_image', 'collect_env_info',
    'get_model_attr', 'clip', 'StateCacher'
]


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024*duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
            (percent, progress_size / (1024*1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write('\n')


def read_image(path, grayscale=False):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.
        grayscale (bool): load grayscale image

    Returns:
        PIL image
    """

    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))

    while not got_img:
        try:
            img = Image.open(path).convert('L' if grayscale else 'RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))

    return img


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info
    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def get_model_attr(model, attr):
    if hasattr(model, 'module'):
        return getattr(model.module, attr)
    else:
        return getattr(model, attr)

def clip(lr, pretrained, backbone_name):
    if not pretrained:
        if (lr >= 1.) or (lr <= 1e-4):
            print("Fail to find lr automaticaly. Lr finder gave either too high ot too low learning rate"
                  "set lr to standart one.")
            return 0.01
        return lr
    print(lr, pretrained, backbone_name)
    if backbone_name == "EfficientNet":
        if (exponent(lr) == 3) and (lr <= 0.0035):
            clipped_lr = lr
        elif (exponent(lr) == 3) and (lr > 0.0035):
            clipped_lr = round(lr / 2, 6)
        elif (exponent(lr) >= 4) and (exponent(lr) <= 1):
            print("Fail to find lr automaticaly. LR Finder gave either too high ot too low learning rate"
                  "set lr to average one for EfficientNet: {}".format(0.003))
            return 0.003
        else:
            clipped_lr = lr / 19.6

    elif backbone_name == "MobileNetV3":
        if (lr <= 0.1 and lr > 0.02):
            k = -180.2548*(lr**2) + 104.5253*lr - 1.0182
            print(lr, k)
            clipped_lr = lr / k
        elif (lr < 0.01 or lr > 0.1):
            print("Fail to find lr automaticaly. LR Finder gave either too high ot too low learning rate"
                  "set lr to average one for MobileNetV3: {}".format(0.013))
            return 0.013
        else:
            clipped_lr = lr
    else:
        print("Unknown backbone, the results could be wrong. LR found by ")
        return lr

    print("Finished searching learning rate. Choosed {} as the best proposed.".format(clipped_lr))
    return clipped_lr

def exponent(n):
    s = '{:.16f}'.format(n).split('.')[1]
    return len(s) - len(s.lstrip('0')) + 1


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile

            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError("Given `cache_dir` is not a valid directory.")

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, "state_{}_{}.pt".format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError("Target {} was not cached.".format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError(
                    "Failed to load state in {}. File doesn't exist anymore.".format(fn)
                )
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""

        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])
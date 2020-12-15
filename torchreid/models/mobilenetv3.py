"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen,
Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang,
Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.

Original repository: https://github.com/d-li14/mobilenetv3.pytorch
"""

import math
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout, HSigmoid, HSwish

from .common import ModelInterface

__all__ = ['mobilenetv3_small', 'mobilenetv3_large']

pretrained_urls = {
    'mobilenetv3_small':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-small-55df8e1f.pth',
    'mobilenetv3_large':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-1cd25616.pth',
}


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    :param v:
    :param divisor:
    :param min_value:
    :return:
    """

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                HSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride, instance_norm=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup) if not instance_norm else nn.InstanceNorm2d(oup, affine=True),
        HSwish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        HSwish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inplaces, hidden_dim, outplaces, kernel_size, stride, use_se, use_hs, dropout_cfg=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inplaces == outplaces

        if inplaces == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),

                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),

                # pw-linear
                nn.Conv2d(hidden_dim, outplaces, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outplaces),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inplaces, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HSwish() if use_hs else nn.ReLU(inplace=True),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),

                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                HSwish() if use_hs else nn.ReLU(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, outplaces, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outplaces),
            )

        self.dropout = None
        if dropout_cfg is not None:
            self.dropout = Dropout(**dropout_cfg)

    def forward(self, x):
        y = self.conv(x)

        if self.dropout is not None:
            y = self.dropout(y)

        return x + y if self.identity else y

class MobileNetV3(ModelInterface):
    arch_settings = {
        'large': [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 1],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ],
        'small': [
            # k, t,   c, SE, NL, s
            [3,  16,  16, 1, 0, 2],  # 0
            [3,  72,  24, 0, 0, 2],  # 1
            [3,  88,  24, 0, 0, 1],  # 2
            [5,  96,  40, 1, 1, 2],  # 3
            [5, 240,  40, 1, 1, 1],  # 4
            [5, 240,  40, 1, 1, 1],  # 5
            [5, 120,  48, 1, 1, 1],  # 6
            [5, 144,  48, 1, 1, 1],  # 7
            [5, 288,  96, 1, 1, 2],  # 8
            [5, 576,  96, 1, 1, 1],  # 9
            [5, 576,  96, 1, 1, 1]   # 10
        ]
    }

    def __init__(self,
                 mode,
                 num_classes,
                 width_mult=1.0,
                 feature_dim=256,
                 loss='softmax',
                 IN_first=False,
                 IN_conv1=False,
                 dropout_cfg=None,
                 pooling_type='avg',
                 bn_eval=False,
                 bn_frozen=False,
                 **kwargs):
        super().__init__(**kwargs)

        # config definition
        assert mode in ['large', 'small']
        self.cfg = MobileNetV3.arch_settings[mode]

        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.pooling_type = pooling_type

        self.loss = loss
        self.feature_dim = feature_dim
        assert self.feature_dim is not None and self.feature_dim > 0

        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes
        assert len(self.num_classes) > 0

        self.input_IN = None
        if IN_first:
            self.input_IN = nn.InstanceNorm2d(3, affine=True)

        # building first layer
        input_channel = make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 1, IN_conv1)]

        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfg:
            output_channel = make_divisible(c * width_mult, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, dropout_cfg))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # building last several layers
        backbone_out_num_channels = make_divisible(exp_size * width_mult, 8)
        self.conv = nn.Sequential(
            conv_1x1_bn(input_channel, backbone_out_num_channels),
            SELayer(backbone_out_num_channels) if mode == 'small' else nn.Sequential()
        )

        classifier_block = nn.Linear if self.loss not in ['am_softmax'] else AngleSimpleLinear

        in_feature_dims = [self.feature_dim] * len(self.num_classes)
        out_feature_dims = [self.feature_dim] * len(self.num_classes)
        self.out_feature_dims = out_feature_dims

        self.fc, self.classifier = nn.ModuleList(), nn.ModuleList()
        for trg_id, trg_num_classes in enumerate(self.num_classes):
            self.fc.append(self._construct_fc_layer(backbone_out_num_channels, in_feature_dims[trg_id]))
            if not self.contrastive and trg_num_classes > 0:
                self.classifier.append(classifier_block(out_feature_dims[trg_id], trg_num_classes))

        self._init_weights()

    def forward(self, x, return_featuremaps=False, get_embeddings=False):
        if self.input_IN is not None:
            x = self.input_IN(x)

        y = self.features(x)
        y = self.conv(y)
        if return_featuremaps:
            return y

        glob_features = self._glob_feature_vector(y, self.pooling_type)
        embeddings = [fc(glob_features) for fc in self.fc]

        if self.training and len(self.classifier) == 0:
            return embeddings
        elif not self.training and not self.classification:
            return torch.cat(embeddings, dim=1)

        logits = [classifier(embd) for embd, classifier in zip(embeddings, self.classifier)]

        if not self.training and self.classification:
            return logits

        if len(logits) == 1:
            logits = logits[0]
        if len(embeddings) == 1:
            embeddings = embeddings[0]

        if get_embeddings:
            out_data = [logits, embeddings]
        elif self.loss in ['softmax', 'adacos', 'd_softmax', 'am_softmax']:
            out_data = [logits]
        elif self.loss in ['triplet']:
            out_data = [logits, embeddings]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        return tuple(out_data)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def train(self, train_mode=True):
        super(MobileNetV3, self).train(train_mode)

        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False

        return self

    def load_pretrained_weights(self, pretrained_dict):
        model_dict = self.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []
        for k, v in pretrained_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            warnings.warn(
                'The pretrained weights cannot be loaded, '
                'please check the key names manually '
                '(** ignored and continue **)'
            )
        else:
            print('Successfully loaded pretrained weights')
            if len(discarded_layers) > 0:
                print(
                    '** The following layers are discarded '
                    'due to unmatched keys or layer size: {}'.
                    format(discarded_layers)
                )


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)
    state_dict = torch.load('/home/prokofiev/deep-person-reid/mobilenetv3-large-1cd25616.pth')
    model.load_pretrained_weights(state_dict)


##########
# Instantiation
##########

def mobilenetv3_small(num_classes=1000, pretrained=True, download_weights=False, **kwargs):
    model = MobileNetV3(
        'small',
        num_classes,
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='mobilenetv3_small')

    return model


def mobilenetv3_large(num_classes=1000, pretrained=True, download_weights=False, **kwargs):
    model = MobileNetV3(
        'large',
        num_classes,
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='mobilenetv3_large')

    return model

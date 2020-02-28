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

import torch.nn as nn

from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout, HSigmoid, HSwish

__all__ = ['mobilenetv3_small', 'mobilenetv3_large']

pretrained_urls = {
    'mobilenetv3_small':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-small-c7eb32fe.pth',
    'mobilenetv3_large':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-657e7b3d.pth',
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


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        HSwish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        HSwish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inplaces, hidden_dim, outplaces, kernel_size, stride, use_se, use_hs, dropout_prob=None):
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
        if dropout_prob is not None and dropout_prob > 0.0:
            self.dropout = Dropout(p=dropout_prob)

    def forward(self, x):
        y = self.conv(x)

        if self.dropout is not None:
            y = self.dropout(y)

        return x + y if self.identity else y


class MobileNetV3(nn.Module):
    arch_settings = {
        'large': [
            # k,  t,  c, SE, NL, s
            [3,  16,  16, 0, 0, 1],  # 0
            [3,  64,  24, 0, 0, 2],  # 1
            [3,  72,  24, 0, 0, 1],  # 2
            [5,  72,  40, 1, 0, 2],  # 3
            [5, 120,  40, 1, 0, 1],  # 4
            [5, 120,  40, 1, 0, 1],  # 5
            [3, 240,  80, 0, 1, 2],  # 6
            [3, 200,  80, 0, 1, 1],  # 7
            [3, 184,  80, 0, 1, 1],  # 8
            [3, 184,  80, 0, 1, 1],  # 9
            [3, 480, 112, 1, 1, 1],  # 10
            [3, 672, 112, 1, 1, 1],  # 11
            [5, 672, 160, 1, 1, 1],  # 12
            [5, 672, 160, 1, 1, 2],  # 13
            [5, 960, 160, 1, 1, 1]   # 14
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

    def __init__(self, mode, num_classes, width_mult=1.0, feature_dim=512, loss='softmax', input_instance_norm=False,
                 dropout_prob=None, **kwargs):
        super(MobileNetV3, self).__init__()

        self.feature_dim = feature_dim
        self.loss = loss

        # config definition
        assert mode in ['large', 'small']
        self.cfg = MobileNetV3.arch_settings[mode]

        self.instance_norm_input = None
        if input_instance_norm:
            self.instance_norm_input = nn.InstanceNorm2d(3, affine=True)

        # building first layer
        input_channel = make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        block = InvertedResidual
        for k, exp_size, c, use_se, use_hs, s in self.cfg:
            output_channel = make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, dropout_prob))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = make_divisible(exp_size * width_mult, 8)
        self.conv = nn.Sequential(
            conv_1x1_bn(input_channel, output_channel),
            SELayer(output_channel) if mode == 'small' else nn.Sequential()
        )

        # building embedding layer
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(output_channel, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )

        # building classifier layer
        if loss not in ['am_softmax']:
            self.classifier = nn.Linear(feature_dim, num_classes)
        else:
            self.classifier = AngleSimpleLinear(feature_dim, num_classes)

        self._init_weights()

    def forward(self, x, return_featuremaps=False, get_embeddings=False):
        if self.instance_norm_input is not None:
            x = self.instance_norm_input(x)

        y = self.features(x)
        y = self.conv(y)
        if return_featuremaps:
            return y

        v = self.global_avgpool(y)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if get_embeddings:
            return v, y

        if self.loss in ['softmax', 'am_softmax']:
            return y
        elif self.loss in ['triplet']:
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

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


##########
# Instantiation
##########

def mobilenetv3_small(num_classes=1000, pretrained=True, **kwargs):
    model = MobileNetV3('small', num_classes, input_instance_norm=True, dropout_prob=0.1, **kwargs)

    if pretrained:
        warnings.warn('The imagenet pretrained weights need to be manually downloaded from {}'
                      .format(pretrained_urls['mobilenetv3_small']))

    return model


def mobilenetv3_large(num_classes=1000, pretrained=True, **kwargs):
    model = MobileNetV3('large', num_classes, input_instance_norm=True, dropout_prob=0.1, **kwargs)

    if pretrained:
        warnings.warn('The imagenet pretrained weights need to be manually downloaded from {}'
                      .format(pretrained_urls['mobilenetv3_large']))

    return model

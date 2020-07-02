"""
 Copyright (c) 2018-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import division, absolute_import
import math
import torch.nn as nn

from torchreid.losses import AngleSimpleLinear


__all__ = ['mobile_face_net_se_1x', 'mobile_face_net_se_2x']


def init_block(in_channels, out_channels, stride, activation=nn.PReLU):
    """Builds the first block of the MobileFaceNet"""
    return nn.Sequential(
        nn.BatchNorm2d(3),
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        make_activation(activation)
    )


def make_activation(activation):
    """Factory for activation functions"""
    if activation != nn.PReLU:
        return activation(inplace=True)

    return activation()


class SELayer(nn.Module):
    """Implementation of the Squeeze-Excitaion layer from https://arxiv.org/abs/1709.01507"""
    def __init__(self, inplanes, squeeze_ratio=8, activation=nn.PReLU, size=None):
        super(SELayer, self).__init__()
        assert squeeze_ratio >= 1
        assert inplanes > 0
        if size is not None:
            self.global_avgpool = nn.AvgPool2d(size)
        else:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / squeeze_ratio), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / squeeze_ratio), inplanes, kernel_size=1, stride=1)
        self.relu = make_activation(activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out


class InvertedResidual(nn.Module):
    """Implementation of the modified Inverted residual block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, outp_size=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.inv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.PReLU(),

            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, 1,
                      groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.PReLU(),

            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            SELayer(out_channels, 8, nn.PReLU, outp_size)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_block(x)

        return self.inv_block(x)


class MobileFaceNet(nn.Module):
    """Implements modified MobileFaceNet from https://arxiv.org/abs/1804.07573"""
    def __init__(self,
                 num_classes,
                 feature=False,
                 feature_dim=256,
                 width_multiplier=1.,
                 loss='softmax',
                 input_size=(128, 128),
                 **kwargs):
        super(MobileFaceNet, self).__init__()
        assert feature_dim > 0
        assert num_classes > 0
        assert width_multiplier > 0
        self.feature = feature
        self.loss = loss
        self.input_size = input_size
        print(loss, num_classes, input_size)

        # Set up of inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1]
        ]

        first_channel_num = 64
        last_channel_num = 512
        self.features = [init_block(3, first_channel_num, 2)]

        self.features.append(nn.Conv2d(first_channel_num, first_channel_num, 3, 1, 1,
                                       groups=first_channel_num, bias=False))
        self.features.append(nn.BatchNorm2d(64))
        self.features.append(nn.PReLU())

        # Inverted Residual Blocks
        in_channel_num = first_channel_num
        size_h, size_w = self.input_size
        size_h, size_w = size_h // 2, size_w // 2
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    size_h, size_w = size_h // s, size_w // s
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          s, t, outp_size=(size_h, size_w)))
                else:
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          1, t, outp_size=(size_h, size_w)))
                in_channel_num = output_channel

        # 1x1 expand block
        self.features.append(nn.Sequential(nn.Conv2d(in_channel_num, last_channel_num, 1, 1, 0, bias=False),
                                           nn.BatchNorm2d(last_channel_num),
                                           nn.PReLU()))
        self.features = nn.Sequential(*self.features)

        # Depth-wise pooling
        k_size = (self.input_size[0] // 16, self.input_size[1] // 16)
        self.dw_pool = nn.Conv2d(last_channel_num, last_channel_num, k_size,
                                 groups=last_channel_num, bias=False)
        self.dw_bn = nn.BatchNorm2d(last_channel_num)
        self.conv1_extra = nn.Conv2d(last_channel_num, feature_dim, 1, stride=1, padding=0, bias=False)

        if not self.feature:
            classifier_block = nn.Linear if self.loss not in ['am_softmax'] else AngleSimpleLinear
            self.classifier = classifier_block(feature_dim, num_classes)

        self.init_weights()

    def forward(self, x, return_featuremaps=False, get_embeddings=False):
        x = self.features(x)
        if return_featuremaps:
            return x
        x = self.dw_bn(self.dw_pool(x))
        x = self.conv1_extra(x)

        if self.feature or not self.training:
            return x

        x = x.view(x.size(0), -1)
        y = self.classifier(x)

        if get_embeddings:
            return y, x

        if self.loss in ['softmax', 'am_softmax']:
            return y
        elif self.loss in ['triplet', ]:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

    def get_input_res(self):
        return self.input_size

    def init_weights(self):
        """Initializes weights of the model before training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobile_face_net_se_1x(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = MobileFaceNet(
        num_classes=num_classes,
        width_multiplier=1.0,
        **kwargs
    )

    return model


def mobile_face_net_se_2x(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = MobileFaceNet(
        num_classes=num_classes,
        width_multiplier=1.5,
        **kwargs
    )

    return model

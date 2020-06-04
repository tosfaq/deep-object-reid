import math
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torchreid.losses import AngleSimpleLinear


__all__ = ['res2net50_v1b_26w_4s', 'res2net101_v1b_26w_4s']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    """Res2Net Network.

        Reference:
            - Shanghua et al. Res2Net: A New Multi-scale Backbone Architecture.
            - https://github.com/Res2Net/Res2Net-PretrainedModels
        """

    def __init__(self,
                 num_classes,
                 block,
                 layers,
                 baseWidth=26,
                 scale=4,
                 feature_dim=512,
                 loss='softmax',
                 **kwargs):
        super(Res2Net, self).__init__()

        self.loss = loss
        self.feature_dim = feature_dim
        assert self.feature_dim is not None and self.feature_dim > 0

        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if isinstance(num_classes, (list, tuple)):
            assert len(num_classes) == 2
            real_data_num_classes, synthetic_data_num_classes = num_classes
        else:
            real_data_num_classes, synthetic_data_num_classes = num_classes, None

        classifier_block = nn.Linear if self.loss not in ['am_softmax'] else AngleSimpleLinear
        out_num_channels = 512 * block.expansion
        self.fc = self._construct_fc_layer(out_num_channels, self.feature_dim)
        self.classifier = classifier_block(self.feature_dim, real_data_num_classes)

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    @staticmethod
    def _construct_fc_layer(input_dim, out_dim):
        layers = [
            nn.Linear(input_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        ]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _backbone(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        return y

    @staticmethod
    def _glob_feature_vector(x):
        return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

    def forward(self, x, return_featuremaps=False, get_embeddings=False):
        feature_maps = self._backbone(x)
        if return_featuremaps:
            return feature_maps

        glob_feature = self._glob_feature_vector(feature_maps)
        embeddings = self.fc(glob_feature)

        if not self.training:
            return embeddings

        logits = self.classifier(embeddings)

        if get_embeddings:
            return embeddings, logits
        elif self.loss in ['softmax', 'am_softmax']:
            return logits
        elif self.loss in ['triplet']:
            return embeddings, logits
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

    def load_pretrained_weights(self, pretrained_dict):
        model_dict = self.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in pretrained_dict.items():
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


def res2net50_v1b_26w_4s(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = Res2Net(num_classes,
                    Bottle2neck,
                    [3, 4, 6, 3],
                    baseWidth=26,
                    scale=4,
                    **kwargs)

    if pretrained and download_weights:
        state_dict = model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'])
        model.load_pretrained_weights(state_dict)

    return model


def res2net101_v1b_26w_4s(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = Res2Net(num_classes,
                    Bottle2neck,
                    [3, 4, 23, 3],
                    baseWidth=26,
                    scale=4,
                    **kwargs)

    if pretrained and download_weights:
        state_dict = model_zoo.load_url(model_urls['res2net101_v1b_26w_4s'])
        model.load_pretrained_weights(state_dict)

    return model

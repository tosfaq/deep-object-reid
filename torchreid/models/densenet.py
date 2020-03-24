"""
Code source: https://github.com/pytorch/vision
"""

from __future__ import division, absolute_import

import re
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import model_zoo

from torchreid.losses import AngleSimpleLinear


__all__ = [
    'densenet121', 'densenet161', 'densenet169', 'densenet201',
]

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False
            )
        ),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i*growth_rate, growth_rate, bn_size,
                drop_rate
            )
            self.add_module('denselayer%d' % (i+1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densely connected network.
    
    Reference:
        Huang et al. Densely Connected Convolutional Networks. CVPR 2017.

    Public keys:
        - ``densenet121``: DenseNet121.
        - ``densenet169``: DenseNet169.
        - ``densenet201``: DenseNet201.
        - ``densenet161``: DenseNet161.
    """

    def __init__(
        self,
        num_classes,
        feature_dim=512,
        loss='softmax',
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        attr_tasks=None,
        enable_attr_tasks=False,
        num_parts=None,
        **kwargs
    ):

        super(DenseNet, self).__init__()

        self.loss = loss
        self.feature_dim = feature_dim
        assert self.feature_dim is not None and self.feature_dim > 0

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        'conv0',
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False
                        )
                    ),
                    ('norm0', nn.BatchNorm2d(num_init_features)),
                    ('relu0', nn.ReLU(inplace=True)),
                    (
                        'pool0',
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    ),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features = num_features + num_layers*growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        out_num_channels = num_features

        if isinstance(num_classes, (list, tuple)):
            assert len(num_classes) == 2
            real_data_num_classes, synthetic_data_num_classes = num_classes
        else:
            real_data_num_classes, synthetic_data_num_classes = num_classes, None

        classifier_block = nn.Linear if self.loss not in ['am_softmax'] else AngleSimpleLinear
        self.num_parts = num_parts if num_parts is not None and num_parts > 1 else 0

        fc_layers, classifier_layers = [], []
        for _ in range(self.num_parts + 1):  # main branch + part-based branches
            fc_layers.append(self._construct_fc_layer(out_num_channels, self.feature_dim))
            classifier_layers.append(classifier_block(self.feature_dim, real_data_num_classes))
        self.fc = nn.ModuleList(fc_layers)
        self.classifier = nn.ModuleList(classifier_layers)

        self.aux_fc = None
        self.aux_classifier = None
        self.split_embeddings = synthetic_data_num_classes is not None
        if self.split_embeddings:
            aux_fc_layers, aux_classifier_layers = [], []
            for _ in range(self.num_parts + 1):  # main branch + part-based branches
                aux_fc_layers.append(self._construct_fc_layer(out_num_channels, self.feature_dim))
                aux_classifier_layers.append(classifier_block(self.feature_dim, synthetic_data_num_classes))
            self.aux_fc = nn.ModuleList(aux_fc_layers)
            self.aux_classifier = nn.ModuleList(aux_classifier_layers)

        self.attr_fc = None
        self.attr_classifiers = None
        if enable_attr_tasks and attr_tasks is not None and len(attr_tasks) > 0:
            attr_fc = dict()
            attr_classifier = dict()
            for attr_name, attr_num_classes in attr_tasks.items():
                attr_fc[attr_name] = self._construct_fc_layer(out_num_channels, self.feature_dim // 4)
                attr_classifier[attr_name] = AngleSimpleLinear(self.feature_dim // 4, attr_num_classes)
            self.attr_fc = nn.ModuleDict(attr_fc)
            self.attr_classifiers = nn.ModuleDict(attr_classifier)

        self._init_params()

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
        f = self.features(x)
        f = F.relu(f, inplace=True)

        return f

    @staticmethod
    def _glob_feature_vector(x, num_parts=4):
        return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

    @staticmethod
    def _part_feature_vector(x, num_parts):
        if num_parts <= 1:
            return []

        feature_vectors = F.adaptive_avg_pool2d(x, (num_parts, 1)).squeeze(dim=-1)

        return [f.squeeze(dim=-1) for f in torch.split(feature_vectors, 1, dim=-1)]

    def forward(self, x, return_featuremaps=False, get_embeddings=False, return_logits=False):
        feature_maps = self._backbone(x)
        if return_featuremaps:
            return feature_maps

        glob_feature = self._glob_feature_vector(feature_maps, num_parts=self.num_parts)
        part_features = self._part_feature_vector(feature_maps, num_parts=self.num_parts)
        features = [glob_feature] + list(part_features)

        main_embeddings = [fc(f) for f, fc in zip(features, self.fc)]
        if not self.training and not return_logits:
            return torch.cat(main_embeddings, dim=-1)

        main_logits = [classifier(embd) for embd, classifier in zip(main_embeddings, self.classifier)]
        main_centers = [classifier.get_centers() for classifier in self.classifier]

        if self.split_embeddings:
            aux_embeddings = [fc(f) for f, fc in zip(features, self.aux_fc)]
            aux_logits = [classifier(embd) for embd, classifier in zip(aux_embeddings, self.aux_classifier)]
            aux_centers = [classifier.get_centers() for classifier in self.aux_classifier]
        else:
            aux_embeddings = [None] * len(features)
            aux_logits = [None] * len(features)
            aux_centers = [None] * len(features)

        all_embeddings = dict(real=main_embeddings, synthetic=aux_embeddings)
        all_outputs = dict(real=main_logits, synthetic=aux_logits,
                           real_centers=main_centers, synthetic_centers=aux_centers)

        attr_embeddings = dict()
        if self.attr_fc is not None:
            for attr_name, attr_fc in self.attr_fc.items():
                attr_embeddings[attr_name] = attr_fc(glob_feature)

        attr_logits = dict()
        if self.attr_classifiers is not None:
            for att_name, attr_classifier in self.attr_classifiers.items():
                attr_logits[att_name] = attr_classifier(attr_embeddings[att_name])

        if get_embeddings:
            return all_embeddings, all_outputs, attr_logits

        if self.loss in ['softmax', 'am_softmax']:
            return all_outputs, attr_logits
        elif self.loss in ['triplet']:
            return all_outputs, attr_logits, all_embeddings
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)

    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
    )
    for key in list(pretrain_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pretrain_dict[new_key] = pretrain_dict[key]
            del pretrain_dict[key]

    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


"""
Dense network configurations:
--
densenet121: num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16)
densenet169: num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)
densenet201: num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32)
densenet161: num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24)
"""


def densenet121(num_classes=1000, pretrained=True, **kwargs):
    model = DenseNet(
        num_classes,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])

    return model


def densenet169(num_classes=1000, pretrained=True, **kwargs):
    model = DenseNet(
        num_classes,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet169'])

    return model


def densenet201(num_classes=1000, pretrained=True, **kwargs):
    model = DenseNet(
        num_classes,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet201'])

    return model


def densenet161(num_classes=1000, pretrained=True, **kwargs):
    model = DenseNet(
        num_classes,
        num_init_features=96,
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet161'])

    return model

from __future__ import division, absolute_import

import warnings
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout, HSwish, gumbel_sigmoid, NonLocalModule


__all__ = ['osnet_ain_x1_0']

pretrained_urls = {
    'osnet_ain_x1_0': 'https://drive.google.com/uc?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo'
}


##########
# Basic layers
##########

class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1, out_fn=nn.ReLU, use_in=False):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if use_in else nn.BatchNorm2d(out_channels)
        self.out_fn = out_fn() if out_fn is not None else None

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.out_fn(y) if self.out_fn is not None else y
        return y


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1, out_fn=nn.ReLU):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.out_fn = out_fn() if out_fn is not None else None

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.out_fn(y) if self.out_fn is not None else y
        return y


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class LightConvStream(nn.Module):
    """Lightweight convolution stream."""

    def __init__(self, in_channels, out_channels, depth):
        super(LightConvStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(
            depth
        )
        layers = []
        layers += [LightConv3x3(in_channels, out_channels)]
        for i in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


##########
# Building blocks for spatial attention
##########

class ResidualAttention(nn.Module):
    def __init__(self, in_channels, gumbel=True, reduction=4.0, reg_weight=1.0):
        super(ResidualAttention, self).__init__()

        self.gumbel = gumbel
        self.reg_weight = reg_weight
        assert self.reg_weight > 0.0

        internal_channels = int(in_channels / reduction)
        self.spatial_logits = nn.Sequential(
            Conv1x1(in_channels, internal_channels, out_fn=None),
            HSwish(),
            Conv3x3(internal_channels, internal_channels, groups=internal_channels, out_fn=None),
            HSwish(),
            Conv1x1(internal_channels, 1, out_fn=None),
        )

    def forward(self, x, return_extra_data=False):
        logits = self.spatial_logits(x)

        if self.gumbel and self.training:
            soft_mask = gumbel_sigmoid(logits)
        else:
            soft_mask = torch.sigmoid(logits)

        out = (1.0 + soft_mask) * x

        if return_extra_data:
            return out, dict(logits=logits)
        else:
            return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, gumbel=False):
        super(ChannelAttention, self).__init__()

    def forward(self, x, return_extra_data=False):
        batch, channels, _, _ = x.size()
        m = x.view(batch, channels, -1)
        m_tr = m.permute(0, 2, 1)

        scale = float(channels ** (-0.5))
        attention = F.softmax(scale * torch.matmul(m, m_tr), dim=2)

        y = torch.matmul(attention, m).view_as(x)
        out = x + y

        return out


##########
# Building blocks for omni-scale feature learning
##########

class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, dropout_prob=None, **kwargs):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

        self.dropout = None
        if dropout_prob is not None and dropout_prob > 0.0:
            self.dropout = Dropout(p=dropout_prob)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        x1 = self.conv1(x)

        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)

        x3 = self.conv3(x2)
        if self.dropout is not None:
            x3 = self.dropout(x3)

        out = x3 + identity

        return F.relu(out)


class OSBlockINin(nn.Module):
    """Omni-scale feature learning block with instance normalization."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, dropout_prob=None, **kwargs):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

        self.dropout = None
        if dropout_prob is not None and dropout_prob > 0.0:
            self.dropout = Dropout(p=dropout_prob)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        x1 = self.conv1(x)

        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)

        x3 = self.conv3(x2)
        x3 = self.IN(x3)  # IN inside residual
        if self.dropout is not None:
            x3 = self.dropout(x3)

        out = x3 + identity

        return F.relu(out)


##########
# Network architecture
##########

class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        channels,
        attentions=None,
        nonlocal_blocks=None,
        dropout_probs=None,
        feature_dim=512,
        loss='softmax',
        conv1_IN=False,
        attr_tasks=None,
        enable_attr_tasks=False,
        num_parts=None,
        **kwargs
    ):
        super(OSNet, self).__init__()

        num_blocks = len(blocks)
        assert num_blocks == len(channels) - 1

        self.loss = loss
        self.feature_dim = feature_dim
        assert self.feature_dim is not None and self.feature_dim > 0

        self.dropout_probs = dropout_probs
        if self.dropout_probs is None:
            self.dropout_probs = [None] * num_blocks
        assert len(self.dropout_probs) == num_blocks

        self.use_attentions = attentions
        if self.use_attentions is None:
            self.use_attentions = [False] * (num_blocks + 1)
        assert len(self.use_attentions) == num_blocks + 1

        self.use_nonlocal_blocks = nonlocal_blocks
        if self.use_nonlocal_blocks is None:
            self.use_nonlocal_blocks = [False] * (num_blocks + 1)
        assert len(self.use_nonlocal_blocks) == num_blocks + 1

        if isinstance(num_classes, (list, tuple)):
            assert len(num_classes) == 2
            real_data_num_classes, synthetic_data_num_classes = num_classes
        else:
            real_data_num_classes, synthetic_data_num_classes = num_classes, None

        classifier_block = nn.Linear if self.loss not in ['am_softmax'] else AngleSimpleLinear
        self.num_parts = num_parts if num_parts is not None and num_parts > 1 else 0

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._construct_layer(blocks[0], channels[0], channels[1])
        self.nl2 = self._construct_nonlocal_layer(channels[1], self.use_nonlocal_blocks[0])
        self.att2 = self._construct_attention_layer(channels[1], self.use_attentions[0])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._construct_layer(blocks[1], channels[1], channels[2])
        self.nl3 = self._construct_nonlocal_layer(channels[2], self.use_nonlocal_blocks[1])
        self.att3 = self._construct_attention_layer(channels[2], self.use_attentions[1])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._construct_layer(blocks[2], channels[2], channels[3])
        self.nl4 = self._construct_nonlocal_layer(channels[3], self.use_nonlocal_blocks[2])
        self.att4 = self._construct_attention_layer(channels[3], self.use_attentions[2])

        out_num_channels = channels[3]
        self.conv5 = Conv1x1(channels[3], out_num_channels)
        self.nl5 = self._construct_nonlocal_layer(out_num_channels, self.use_nonlocal_blocks[3])
        self.att5 = self._construct_attention_layer(out_num_channels, self.use_attentions[3])

        # self.glob_max_fc = self._construct_fc_layer(out_num_channels, out_num_channels)
        # self.glob_cont_fc = self._construct_fc_layer(out_num_channels, out_num_channels)
        # self.glob_cat_fc = self._construct_fc_layer(2 * out_num_channels, out_num_channels)

        if self.num_parts > 1:
            self.part_self_fc = nn.ModuleList()
            self.part_rest_fc = nn.ModuleList()
            self.part_cat_fc = nn.ModuleList()
            for _ in range(self.num_parts):
                self.part_self_fc.append(self._construct_fc_layer(out_num_channels, out_num_channels))
                self.part_rest_fc.append(self._construct_fc_layer(out_num_channels, out_num_channels))
                self.part_cat_fc.append(self._construct_fc_layer(2 * out_num_channels, out_num_channels))

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
    def _construct_layer(blocks, in_channels, out_channels, dropout_probs=None):
        if dropout_probs is None:
            dropout_probs = [None] * len(blocks)
        assert len(dropout_probs) == len(blocks)

        layers = []
        layers += [blocks[0](in_channels, out_channels, dropout_prob=dropout_probs[0])]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels, dropout_prob=dropout_probs[i])]

        return nn.Sequential(*layers)

    @staticmethod
    def _construct_attention_layer(num_channels, enable):
        return ChannelAttention(num_channels) if enable else None

    @staticmethod
    def _construct_nonlocal_layer(num_channels, enable):
        return NonLocalModule(num_channels) if enable else None

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
        y = self.maxpool(y)

        y = self.conv2(y)
        if self.nl2 is not None:
            y = self.nl2(y)
        if self.att2 is not None:
            y = self.att2(y)
        y = self.pool2(y)

        y = self.conv3(y)
        if self.nl3 is not None:
            y = self.nl3(y)
        if self.att3 is not None:
            y = self.att3(y)
        y = self.pool3(y)

        y = self.conv4(y)
        if self.nl4 is not None:
            y = self.nl4(y)
        if self.att4 is not None:
            y = self.att4(y)

        y = self.conv5(y)
        if self.nl5 is not None:
            y = self.nl5(y)
        if self.att5 is not None:
            y = self.att5(y)

        return y

    def _glob_feature_vector(self, x, num_parts):
        return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        # row_parts = F.adaptive_avg_pool2d(x, (num_parts, 1)).squeeze(dim=-1)
        #
        # p_max, _ = torch.max(row_parts, dim=2)
        # p_avg = torch.mean(row_parts, dim=2)
        # p_cont = p_avg - p_max
        #
        # p_max_embd = self.glob_max_fc(p_max)
        # p_cont_embd = self.glob_cont_fc(p_cont)
        #
        # p_cat = torch.cat((p_max_embd, p_cont_embd), dim=1)
        # out = p_max_embd + self.glob_cat_fc(p_cat)
        #
        # return out

    def _part_feature_vector(self, x, num_parts):
        if num_parts <= 1:
            return []

        # gap_branch = F.adaptive_avg_pool2d(x, (num_parts, 1)).squeeze(dim=-1)
        # gmp_branch = F.adaptive_max_pool2d(x, (num_parts, 1)).squeeze(dim=-1)
        # feature_vectors = gap_branch + gmp_branch
        #
        # return [f.squeeze(dim=-1) for f in torch.split(feature_vectors, 1, dim=-1)]

        row_parts = F.adaptive_avg_pool2d(x, (num_parts, 1)).squeeze(dim=-1)
        row_parts = [f.squeeze(dim=-1) for f in torch.split(row_parts, 1, dim=-1)]

        row_outs = []
        for i in range(num_parts):
            p = row_parts[i]
            r = sum([row_parts[k] for k in range(num_parts) if k != i]) / float(num_parts - 1)

            p_embd = self.part_self_fc[i](p)
            r_embd = self.part_rest_fc[i](r)

            p_cat = torch.cat((p_embd, r_embd), dim=1)
            row_outs.append(p_embd + self.part_cat_fc[i](p_cat))

        return row_outs

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

    state_dict = torch.load(cached_file)
    model.load_pretrained_weights(state_dict)


##########
# Instantiation
##########

def osnet_ain_x1_0(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = OSNet(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin],
            [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        channels=[64, 256, 384, 512],
        # attentions=[False, False, False, True],
        # nonlocal_blocks=[False, True, True, False],
        # dropout_probs=[
        #     [None, 0.1],
        #     [0.1, None],
        #     [0.1, None]
        # ],
        conv1_IN=True,
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='osnet_ain_x1_0')

    return model

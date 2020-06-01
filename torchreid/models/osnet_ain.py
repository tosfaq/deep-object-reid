from __future__ import division, absolute_import

import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout, HSwish, GumbelSigmoid, LocalContrastNormalization


__all__ = ['osnet_ain_x1_0', 'osnet_ain2_x1_0']

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
# Attention modules
##########

class ResidualAttention(nn.Module):
    def __init__(self, in_channels, gumbel=True, reduction=4.0, residual=True):
        super(ResidualAttention, self).__init__()

        self.residual = residual

        internal_channels = int(in_channels / reduction)
        self.spatial_attention = nn.Sequential(
            Conv1x1(in_channels, internal_channels, out_fn=None),
            HSwish(),
            Conv3x3(internal_channels, internal_channels, groups=internal_channels, out_fn=None),
            HSwish(),
            Conv1x1(internal_channels, 1, out_fn=None),
            GumbelSigmoid(scale=5.0) if gumbel else nn.Sigmoid()
        )

    def forward(self, x, return_mask=False):
        soft_mask = self.spatial_attention(x)
        out = (1.0 + soft_mask) * x if self.residual else soft_mask * x

        if return_mask:
            return out, soft_mask
        else:
            return out


##########
# Building blocks for omni-scale feature learning
##########

class LCTGate(nn.Module):
    def __init__(self, channels, groups=16):
        super(LCTGate, self).__init__()

        assert channels > 0
        assert groups > 0
        self.gn = nn.GroupNorm(groups, channels, affine=True)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.gate_activation = nn.Sigmoid()

    def init_params(self):
        nn.init.ones_(self.gn.bias)
        nn.init.zeros_(self.gn.weight)

    def forward(self, x):
        y = self.global_avgpool(x)
        y = self.gn(y)
        y = self.gate_activation(y)
        out = y * x

        return out


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
        head_attention=False,
        attentions=None,
        dropout_probs=None,
        feature_dim=256,
        loss='softmax',
        input_lcn=False,
        input_IN=False,
        conv1_IN=False,
        bn_eval=False,
        bn_frozen=False,
        attr_names=None,
        attr_num_classes=None,
        **kwargs
    ):
        super(OSNet, self).__init__()

        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

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
            self.use_attentions = [False] * (num_blocks + 2)
        assert len(self.use_attentions) == num_blocks + 2

        if not isinstance(num_classes, (list, tuple)):
            num_classes = [num_classes]
        self.num_classes = num_classes
        assert len(self.num_classes) > 0

        self.input_lcn = LocalContrastNormalization(3, 5, affine=True) if input_lcn else None
        self.input_IN = nn.InstanceNorm2d(3, affine=True) if input_IN else None

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.att1 = self._construct_attention_layer(channels[0], self.use_attentions[0])
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._construct_layer(blocks[0], channels[0], channels[1])
        self.att2 = self._construct_attention_layer(channels[1], self.use_attentions[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._construct_layer(blocks[1], channels[1], channels[2])
        self.att3 = self._construct_attention_layer(channels[2], self.use_attentions[2])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._construct_layer(blocks[2], channels[2], channels[3])
        self.att4 = self._construct_attention_layer(channels[3], self.use_attentions[3])

        out_num_channels = channels[3]
        self.conv5 = Conv1x1(channels[3], out_num_channels)
        self.att5 = self._construct_attention_layer(out_num_channels, self.use_attentions[4])

        self.head_att = self._construct_head_attention(out_num_channels, enable=head_attention)

        classifier_block = nn.Linear if self.loss not in ['am_softmax'] else AngleSimpleLinear
        self.fc, self.classifier = nn.ModuleList(), nn.ModuleList()
        for trg_id, trg_num_classes in enumerate(self.num_classes):
            self.fc.append(self._construct_fc_layer(out_num_channels, self.feature_dim, dropout=False))
            if trg_num_classes > 0:
                self.classifier.append(classifier_block(self.feature_dim, trg_num_classes))

        self.use_attr = attr_names is not None and attr_num_classes is not None
        if self.use_attr:
            assert len(attr_names) == len(attr_num_classes)

            self.attr_names = []
            self.attr, self.attr_classifier = nn.ModuleDict(), nn.ModuleDict()
            attr_feature_dim = self.feature_dim // 4
            for attr_name, attr_size in zip(attr_names, attr_num_classes):
                if attr_size is None or attr_size <= 0:
                    continue

                self.attr[attr_name] = self._construct_fc_layer(out_num_channels, attr_feature_dim, dropout=False)
                self.attr_classifier[attr_name] = classifier_block(attr_feature_dim, attr_size)

                self.attr_names.append(attr_name)

            if len(self.attr) == 0:
                self.use_attr = False

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
        return ResidualAttention(num_channels, gumbel=False, residual=True) if enable else None

    @staticmethod
    def _construct_head_attention(num_channels, enable, channel_factor=8, gumbel=True, gumbel_scale=5.0):
        if not enable:
            return None

        internal_num_channels = int(float(num_channels) / float(channel_factor))

        layers = [
            Conv1x1(num_channels, internal_num_channels, out_fn=None),
            HSwish(),
            Conv3x3(internal_num_channels, internal_num_channels, groups=internal_num_channels, out_fn=None),
            HSwish(),
            Conv1x1(internal_num_channels, 1, out_fn=None),
            GumbelSigmoid(scale=gumbel_scale) if gumbel else nn.Sigmoid()
        ]

        return nn.Sequential(*layers)

    @staticmethod
    def _construct_fc_layer(input_dim, output_dim, dropout=False):
        layers = []

        if dropout:
            layers.append(Dropout(p=0.2, dist='gaussian'))

        layers.extend([
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        ])

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
            elif isinstance(m, LocalContrastNormalization):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LCTGate):
                m.init_params()

    def _backbone(self, x):
        att_maps = []

        y = self.input_lcn(x) if self.input_lcn is not None else x
        y = self.input_IN(y) if self.input_IN is not None else y

        y = self.conv1(y)
        if self.att1 is not None:
            y, att1 = self.att1(y, return_mask=True)
            att_maps.append(att1)
        y = self.pool1(y)

        y = self.conv2(y)
        if self.att2 is not None:
            y, att2 = self.att2(y, return_mask=True)
            att_maps.append(att2)
        y = self.pool2(y)

        y = self.conv3(y)
        if self.att3 is not None:
            y, att3 = self.att3(y, return_mask=True)
            att_maps.append(att3)
        y = self.pool3(y)

        y = self.conv4(y)
        if self.att4 is not None:
            y, att4 = self.att4(y, return_mask=True)
            att_maps.append(att4)

        y = self.conv5(y)
        if self.att5 is not None:
            y, att5 = self.att5(y, return_mask=True)
            att_maps.append(att5)

        return y, att_maps

    @staticmethod
    def _glob_feature_vector(x, head_att=None):
        if head_att is not None:
            att_map = head_att(x)
            with torch.no_grad():
                num_values = torch.sum(att_map, dim=(2, 3), keepdim=True)
                scale = num_values.clamp_min(1.0).pow(-1)

            y = scale * att_map * x
            out = torch.sum(y, dim=(2, 3))
        else:
            out = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
            att_map = None

        return out, att_map

    def forward(self, x, return_featuremaps=False, get_embeddings=False, get_extra_data=False):
        feature_maps, feature_att_maps = self._backbone(x)
        if return_featuremaps:
            return feature_maps

        glob_features, head_att_map = self._glob_feature_vector(feature_maps, self.head_att)
        embeddings = [fc(glob_features) for fc in self.fc]

        attr_embeddings = {}
        if self.use_attr:
            attr_embeddings = {attr_name: attr_fc(glob_features) for attr_name, attr_fc in self.attr.items()}

        if not self.training:
            return torch.cat(embeddings + [attr_embeddings[attr_name] for attr_name in self.attr_names], dim=1)

        logits = [classifier(embd) for embd, classifier in zip(embeddings, self.classifier)]

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

        if get_extra_data:
            extra_out_data = dict()
            extra_out_data['att_maps'] = [head_att_map] + feature_att_maps

            if self.use_attr:
                attr_logits = {attr_name: attr_classifier(attr_embeddings[attr_name])
                               for attr_name, attr_classifier in self.attr_classifier.items()}
                extra_out_data['attr_logits'] = attr_logits

            out_data += [extra_out_data]

        return tuple(out_data)

    def train(self, train_mode=True):
        super(OSNet, self).train(train_mode)

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
        conv1_IN=True,
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='osnet_ain_x1_0')

    return model


def osnet_ain2_x1_0(num_classes, pretrained=False, download_weights=False,
                    enable_attentions=False, **kwargs):
    model = OSNet(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin],
            [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        channels=[64, 256, 384, 512],
        attentions=[False, True, True, False, False] if enable_attentions else None,
        input_IN=True,
        conv1_IN=True,
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='osnet_ain_x1_0')

    return model

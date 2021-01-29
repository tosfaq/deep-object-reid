import torch.nn as nn
import torch
import math
from .common import (ModelInterface, round_channels, conv1x1, conv1x1_block, conv3x3_block,
                    dwconv3x3_block, dwconv5x5_block, SEBlock, HSwish)
from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout, rsc, EvalModeSetter


__all__ = ['mobilenetv3_large', 'mobilenetv3_large_075', 'mobilenetv3_small']

pretrained_urls = {
    'mobilenetv3_small':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-small-55df8e1f.pth?raw=true',
    'mobilenetv3_large':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-1cd25616.pth?raw=true',
    'mobilenetv3_large_075':
    'https://github.com/d-li14/mobilenetv3.pytorch/blob/master/pretrained/mobilenetv3-large-0.75-9632d2a8.pth?raw=true',
}

def _make_divisible(v, divisor, min_value=None):
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


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
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
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
        # h_swish() if self.loss == 'softmax' else nn.PReLU()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3PreClassifier(nn.Module):
    """
    MobileNetV3 classifier.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    dropout_cls : dict
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 loss = 'softmax',
                 dropout_cls=None):
        super().__init__()
        self.use_dropout = (dropout_cls != None)

        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.activ = (h_swish(inplace=True)
                      if loss == 'softmax'
                      else nn.PReLU())
        if self.use_dropout:
            self.dropout = Dropout(**dropout_cls)

    def forward(self, x):
        x1 = self.conv1(x)
        x2= self.activ(x1)
        if self.use_dropout:
            x2 = self.dropout(x2, x)
        return x2

class MobileNetV3(ModelInterface):
    def __init__(self,
                 cfgs,
                 mode,
                 num_classes=1000,
                 width_mult=1.,
                 in_channels=3,
                 in_size=(224, 224),
                 dropout_cls = None,
                 dropout_cfg = None,
                 pooling_type='avg',
                 bn_eval=False,
                 bn_frozen=False,
                 feature_dim=1280,
                 loss='softmax',
                 IN_first=False,
                 IN_conv1=False,
                 self_challenging_cfg=False,
                 lr_finder=None,
                 **kwargs):

        super().__init__(**kwargs)
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.in_size = in_size
        self.num_classes = num_classes
        self.input_IN = nn.InstanceNorm2d(3, affine=True) if IN_first else None
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.pooling_type = pooling_type
        self.self_challenging_cfg = self_challenging_cfg
        self.lr_finder = lr_finder

        self.loss = loss
        self.feature_dim = feature_dim
        assert self.feature_dim is not None and self.feature_dim > 0
        assert mode in ['large', 'small']
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        stride = 1 if in_size[0] < 100 else 2
        layers = [conv_3x3_bn(3, input_channel, stride)]
        # building inverted residual blocks
        block = InvertedResidual
        flag = True
        for k, t, c, use_se, use_hs, s in self.cfgs:
            if (in_size[0] < 100) and (s == 2) and flag:
                s = 1
                flag = False
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]

        if self.loss == 'softmax':
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                h_swish(),
                Dropout(**dropout_cls),
                nn.Linear(self.feature_dim, num_classes),
            )
        else:
            assert self.loss == 'am_softmax'
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.PReLU(),
                Dropout(**dropout_cls),
                AngleSimpleLinear(self.feature_dim, num_classes),
            )

        # if self.loss == 'softmax':
        #      self.classifier = nn.Sequential(
        #                         MobileNetV3PreClassifier(
        #                             in_channels=exp_size,
        #                             out_channels=num_classes,
        #                             mid_channels=self.feature_dim,
        #                             dropout_cls=dropout_cls),
        #                         conv1x1(
        #                             in_channels=self.feature_dim,
        #                             out_channels=num_classes,
        #                             bias=True) # question about bias
        #                         )
        # else:
        #     assert self.loss == 'am_softmax'
        #     self.classifier = nn.Sequential(
        #                         MobileNetV3PreClassifier(
        #                             in_channels=exp_size,
        #                             out_channels=num_classes,
        #                             mid_channels=self.feature_dim,
        #                             dropout_cls=dropout_cls)
        #                         )
        #     self.asl = AngleSimpleLinear(
        #         in_features=self.feature_dim,
        #         out_features=num_classes)

            # self.classifier = AngleSimpleLinear(
            #     in_features=exp_size,
            #     out_features=num_classes)

        self._initialize_weights()

    def forward(self, x, return_featuremaps=False, get_embeddings=False, gt_labels=None):
        if self.input_IN is not None:
            x = self.input_IN(x)

        y = self.conv(self.features(x))

        if return_featuremaps:
            return y

        glob_features = self._glob_feature_vector(y, self.pooling_type, reduce_dims=False)
        logits = self.classifier(glob_features.view(x.shape[0], -1))

        if self.training and self.self_challenging_cfg.enable and gt_labels is not None:
            glob_features = rsc(
                features = glob_features,
                scores = logits,
                labels = gt_labels,
                retain_p = 1.0 - self.self_challenging_cfg.drop_p,
                retain_batch = 1.0 - self.self_challenging_cfg.drop_batch_p
            )

            with EvalModeSetter([self.output], m_type=(nn.BatchNorm1d, nn.BatchNorm2d)):
                logits = self.classifier(glob_features.view(x.shape[0], -1))

        if not self.training and self.classification:
            return [logits]

        if get_embeddings:
            out_data = [logits, glob_features]
        elif self.loss in ['softmax', 'am_softmax']:
            if self.lr_finder.enable and self.lr_finder.lr_find_mode == 'automatic':
                out_data = logits
            else:
                out_data = [logits]
        elif self.loss in ['triplet']:
            out_data = [logits, glob_features]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        if self.lr_finder.enable and self.lr_finder.lr_find_mode == 'automatic':
            return out_data
        return tuple(out_data)

    def _initialize_weights(self):
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

def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from .model_store import load_model

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
    # if not os.path.exists(cached_file):
    gdown.download(pretrained_urls[key], cached_file)

    # state_dict = torch.load(cached_file)
    model = load_model(model, cached_file)

def mobilenetv3_large_075(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
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
    ]

    net = MobileNetV3(cfgs, mode='large', width_mult =.75, **kwargs)
    if pretrained:
        init_pretrained_weights(net, key='mobilenetv3_large_075')

    return net

def mobilenetv3_large(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
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
    ]

    net = MobileNetV3(cfgs, mode='large', width_mult = 1.5, **kwargs)
    if pretrained:
        init_pretrained_weights(net, key='mobilenetv3_large')

    return net

def mobilenetv3_small(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    net = MobileNetV3(cfgs, mode='small', width_mult = 1., **kwargs)
    if pretrained:
        init_pretrained_weights(net, key='mobilenetv3_small')

    return net

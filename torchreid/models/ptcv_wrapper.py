"""
 Copyright (c) 2020-2021 Intel Corporation
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

from functools import partial

import torch.nn as nn
from pytorchcv.model_provider import get_model

from torchreid.losses import AngleSimpleLinear
from .common import ModelInterface

__all__ = ['pytcv_wrapped_models']

AVAI_MODELS = {"mobilenetv2_large" : "mobilenetv2_w1"}

class PTCVModel(ModelInterface):
    def __init__(self,
                 model_name,
                 num_classes,
                 loss='softmax',
                 IN_first=False,
                 pooling_type='avg',
                 **kwargs):
        super().__init__(**kwargs)
        self.pooling_type = pooling_type
        self.loss = loss
        assert isinstance(num_classes, int)

        model = get_model(model_name, num_classes=1000, pretrained=self.pretrained)
        assert hasattr(model, 'features') and isinstance(model.features, nn.Sequential)
        self.features = model.features
        self.features = self.features[:-1] # remove pooling, since it can have a fixed size
        if self.loss not in ['am_softmax']:
            self.output_conv = nn.Conv2d(in_channels=model.output.in_channels, out_channels=num_classes, kernel_size=1, stride=1, bias=False)
        else:
            self.output_conv = AngleSimpleLinear(model.output.in_channels, num_classes)
            self.num_features = self.num_head_features = model.output.in_channels

        self.input_IN = nn.InstanceNorm2d(3, affine=True) if IN_first else None

    def forward(self, x, return_featuremaps=False):
        if self.input_IN is not None:
            x = self.input_IN(x)

        y = self.features(x)
        if return_featuremaps:
            return y

        glob_features = self._glob_feature_vector(y, self.pooling_type, reduce_dims=False)

        logits = self.output_conv(glob_features).view(x.shape[0], -1)

        if not self.training:
            return [logits]
        return tuple([logits])


pytcv_wrapped_models = {dor_name : partial(PTCVModel, model_name=pytcv_name) for dor_name, pytcv_name in AVAI_MODELS.items()}

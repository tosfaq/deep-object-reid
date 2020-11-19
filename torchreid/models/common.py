"""
 Copyright (c) 2020 Intel Corporation
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

import torch.nn as nn
import torch.nn.functional as F

from torchreid.ops import Dropout


class ModelInterface(nn.Module):
     def __init__(self,
                 classification=False,
                 contrastive=False,
                 pretrained=False,
                 **kwargs):
          super().__init__()

          self.classification = classification
          self.contrastive = contrastive
          self.pretrained = pretrained

     @staticmethod
     def _glob_feature_vector(x, mode):
          if mode == 'avg':
               out = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
          elif mode == 'max':
               out = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
          elif mode == 'avg+max':
               avg_pool = F.adaptive_avg_pool2d(x, 1)
               max_pool = F.adaptive_max_pool2d(x, 1)
               out = (avg_pool + max_pool).view(x.size(0), -1)
          else:
               raise ValueError(f'Unknown pooling mode: {mode}')

          return out

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

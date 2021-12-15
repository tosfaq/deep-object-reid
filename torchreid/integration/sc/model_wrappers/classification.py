# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import numpy as np
from typing import Any, Dict

try:
    from openvino.model_zoo.model_api.models.classification import Classification
    from openvino.model_zoo.model_api.models.types import BooleanValue
except ImportError as e:
    import warnings
    warnings.warn("ModelAPI was not found.")


class OteClassification(Classification):
    __model__ = 'ote_classification'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'multilabel': BooleanValue(default_value=False)
        })

        return parameters

    def postprocess(self, outputs: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        outputs = outputs[self.out_layer_name].squeeze()
        self.activate = False
        if not np.isclose(np.sum(outputs), 1.0, atol=0.01):
            self.activate = True

        if self.multilabel:
            return get_multilabel_predictions(outputs, activate=self.activate)

        return get_multiclass_predictions(outputs, activate=self.activate)


def sigmoid_numpy(x: np.ndarray):
    return 1. / (1. + np.exp(-1. * x))


def softmax_numpy(x: np.ndarray):
    x = np.exp(x)
    x /= np.sum(x)
    return x


def get_multiclass_predictions(logits: np.ndarray, activate: bool = True):

    index = np.argmax(logits)
    if activate:
        logits = softmax_numpy(logits)
    return [(index, logits[index])]


def get_multilabel_predictions(logits: np.ndarray, pos_thr: float = 0.5, activate: bool = True):
    if activate:
        logits = sigmoid_numpy(logits)
    scores = []
    indices = []
    for i in range(logits.shape[0]):
        if logits[i] > pos_thr:
            indices.append(i)
            scores.append(logits[i])

    return list(zip(indices, scores))

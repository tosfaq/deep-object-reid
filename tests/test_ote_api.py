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

import io
import os.path as osp
import random
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from subprocess import run
from typing import Optional

import numpy as np
import pytest
import torch
from bson import ObjectId
from e2e_test_system import e2e_pytest_api

from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.configuration.helper import convert, create

from torchreid.integration.sc.parameters import OTEClassificationParameters


DEFAULT_TEMPLATE_DIR = osp.join('configs', 'ote_custom_classification', 'efficientnet_b0')


@e2e_pytest_api
def test_reading_efficientnet_b0():
    parse_model_template(osp.join('configs', 'ote_custom_classification', 'efficientnet_b0', 'template.yaml'))


@e2e_pytest_api
def test_reading_mobilenet_v3_large_1():
    parse_model_template(osp.join('configs', 'ote_custom_classification', 'mobilenet_v3_large_1', 'template.yaml'))


@e2e_pytest_api
def test_configuration_yaml():
    configuration = OTEClassificationParameters()
    configuration_yaml_str = convert(configuration, str)
    configuration_yaml_converted = create(configuration_yaml_str)
    configuration_yaml_loaded = create(osp.join('torchreid', 'integration', 'sc', 'configuration.yaml'))
    assert configuration_yaml_converted == configuration_yaml_loaded

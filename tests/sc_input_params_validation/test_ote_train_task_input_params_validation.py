# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelConfiguration, ModelEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from torchreid.integration.sc.train_task import OTEClassificationTrainingTask

from .helpers import load_test_dataset


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestOTEClassificationTrainingTaskInputParamsValidation:
    @staticmethod
    def task():
        labels_list = load_test_dataset()[1]
        labels_schema = LabelSchemaEntity.from_labels(labels_list)
        model_template = parse_model_template(
            "configs/ote_custom_classification/efficientnet_b0/template.yaml"
        )

        params = create(model_template.hyper_parameters.data)
        params.learning_parameters.num_iters = 5
        params.learning_parameters.learning_rate_warmup_iters = 1
        params.learning_parameters.batch_size = 2
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template,
        )
        return OTEClassificationTrainingTask(task_environment=environment)

    @staticmethod
    def model():
        model_configuration = ModelConfiguration(
            configurable_parameters=ConfigurableParameters(
                header="header", description="description"
            ),
            label_schema=LabelSchemaEntity(),
        )
        return ModelEntity(
            train_dataset=DatasetEntity(), configuration=model_configuration
        )

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ote_classification_train_task_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationTrainingTask object initialization parameters validation

        <b>Input data:</b>
        "task_environment" non-TaskEnvironment object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEClassificationTrainingTask object initialization parameter
        """
        with pytest.raises(ValueError):
            OTEClassificationTrainingTask(task_environment="unexpected string")  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ote_classification_train_task_save_model_input_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationTrainingTask object "save_model" method input parameters validation

        <b>Input data:</b>
        OTEClassificationTrainingTask object, "model" non-ModelEntity object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "save_model" method
        """
        task = self.task()
        with pytest.raises(ValueError):
            task.save_model(output_model="unexpected string")  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_ote_classification_train_task_input_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationTrainingTask object "train" method input parameters validation

        <b>Input data:</b>
        OTEClassificationTrainingTask object, "train" method unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "train" method
        """
        task = self.task()
        correct_values_dict = {
            "dataset": DatasetEntity(),
            "output_model": self.model(),
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "dataset" parameter
            ("dataset", unexpected_str),
            # Unexpected string is specified as "output_model" parameter
            ("output_model", unexpected_str),
            # Unexpected string is specified as "train_parameters" parameter
            ("train_parameters", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=task.train,
        )

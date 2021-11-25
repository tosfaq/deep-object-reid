
from attr import attrs
from sys import maxsize

from ote_sdk.configuration.elements import (ParameterGroup,
                                            add_parameter_group,
                                            configurable_float,
                                            configurable_integer,
                                            selectable,
                                            string_attribute,
                                            )
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.model_lifecycle import ModelLifecycle

from .parameters_enums import POTQuantizationPreset, Models

@attrs
class OTEClassificationParameters(ConfigurableParameters):
    header = string_attribute("Configuration for an image classification task")
    description = header

    @attrs
    class __LearningParameters(ParameterGroup):
        header = string_attribute("Learning Parameters")
        description = header

        batch_size = configurable_integer(
            default_value=32,
            min_value=1,
            max_value=512,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        max_num_epochs = configurable_integer(
            default_value=200,
            min_value=1,
            max_value=1000,
            header="Maximum number of training epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        learning_rate = configurable_float(
            default_value=0.01,
            min_value=1e-07,
            max_value=1e-01,
            header="Learning rate",
            description="Increasing this value will speed up training convergence but might make it unstable.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

    @attrs
    class __InferenceParameters(ParameterGroup):
        header = string_attribute("Parameters for inference")
        description = header

        class_name = selectable(default_value=Models.CLASSIFICATION,
                                header="Model class for inference",
                                description="Model classes with defined pre- and postprocessing",
                                editable=False,
                                visible_in_ui=True)

        @attrs
        class __Postprocessing(ParameterGroup):
            header = string_attribute("Postprocessing")
            description = header

            topk = configurable_integer(
                header="Top k elements",
                description="First top k elements will be displayed",
                default_value=1,
                min_value=1,
                affects_outcome_of=ModelLifecycle.INFERENCE
            )

        postprocessing = add_parameter_group(__Postprocessing)

    @attrs
    class __POTParameter(ParameterGroup):
        header = string_attribute("POT Parameters")
        description = header

        stat_subset_size = configurable_integer(
            header="Number of data samples",
            description="Number of data samples used for post-training optimization",
            default_value=300,
            min_value=1,
            max_value=maxsize
        )

        preset = selectable(default_value=POTQuantizationPreset.PERFORMANCE, header="Preset",
                            description="Quantization preset that defines quantization scheme",
                            editable=False, visible_in_ui=False)

    learning_parameters = add_parameter_group(__LearningParameters)
    inference_parameters = add_parameter_group(__InferenceParameters)
    pot_parameters = add_parameter_group(__POTParameter)


from attr import attrs
from sc_sdk.configuration import ModelConfig, ModelLifecycle
from ote_sdk.configuration.elements import (ParameterGroup,
                                            add_parameter_group,
                                            boolean_attribute,
                                            configurable_boolean,
                                            configurable_float,
                                            configurable_integer,
                                            string_attribute)

@attrs
class OTEClassificationParameters(ModelConfig):
    header = string_attribute("Configuration for an object detection task")
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
            max_value=100000,
            header="Maximum number of training epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

    @attrs
    class __AlgoBackend(ParameterGroup):
        header = string_attribute("Internal Algo Backend parameters")
        description = header
        visible_in_ui = boolean_attribute(False)

        template = string_attribute("template.yaml")
        model = string_attribute("main_model.yaml")
        model_name = string_attribute("image classification model")

    learning_parameters = add_parameter_group(__LearningParameters)
    algo_backend = add_parameter_group(__AlgoBackend)

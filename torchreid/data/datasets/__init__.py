# Copyright (c) 2018-2021 Kaiyang Zhou
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import absolute_import, print_function
from copy import copy

from .image import (Classification, ClassificationImageFolder, ExternalDatasetWrapper,
                    MultiLabelClassification)

__image_datasets = {
    'classification': Classification,
    'classification_image_folder' : ClassificationImageFolder,
    'external_classification_wrapper' : ExternalDatasetWrapper,
    'multilabel_classification': MultiLabelClassification,
}


def init_image_dataset(name, custom_dataset_names=[''],
                       custom_dataset_roots=[''],
                       custom_dataset_types=[''], **kwargs):
    """Initializes an image dataset."""

    # handle also custom datasets
    avai_datasets = list(__image_datasets.keys())
    assert len(name) > 0
    if name not in avai_datasets and name not in custom_dataset_names:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {} {}'.format(name, avai_datasets, custom_dataset_names)
        )
    if name in custom_dataset_names:
        assert len(custom_dataset_names) == len(custom_dataset_types)
        assert len(custom_dataset_names) == len(custom_dataset_roots)
        i = custom_dataset_names.index(name)
        new_kwargs = copy(kwargs)
        if custom_dataset_types[i] == 'external_classification_wrapper':
            new_kwargs['data_provider'] = custom_dataset_roots[i]
        else:
            new_kwargs['root'] = custom_dataset_roots[i]
        return __image_datasets[custom_dataset_types[i]](**new_kwargs)

    return __image_datasets[name](**kwargs)

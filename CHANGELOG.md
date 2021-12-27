# Changelog

All notable changes to this project will be documented in this file.

This project is based on [deep-person-reid project by KaiyangZhou](https://github.com/KaiyangZhou/deep-person-reid).
With respect to it we made the following changes.

## \[2021-12-27\]
### Added
* ImageAMSoftmaxEngine, MultilabelEngine with new features: mutual learning, am-softmax loss support, metric losses, EMA, SAM support.
* Classification models, datasets, adapted engines for classification pipelines
* Onnx and OpenVINO export
* Multilabel classification support
* Validation script
* Multilabel classification metrics
* Augmentations, optimizers, losses, learning rate schedulers
* NNCF support

### Removed
* Reid backbones and datasets

### Deprecated
* Video data support

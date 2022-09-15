# **Multilabel Classification with Metric Learning**

This repository contains code for the paper [K. Prokofiev, V. Sovrasov - Combining Metric Learning and Attention Heads For Accurate and Efficient Multilabel Image Classification](https://arxiv.org/abs/2209.06585).
Also it contains the implementations of [MLDecoder](https://arxiv.org/abs/2111.12933),
[ASL](https://arxiv.org/abs/2009.14119) and [AAM](https://arxiv.org/abs/2209.06585) losses, [TResNet-L](https://arxiv.org/abs/2003.13630) backbone and a modified version of [ML-GCN](https://arxiv.org/abs/1904.03582).
Supported backbones table:

Model         | Input Resolution | MACs(G)* | mAP on COCO 14* | mAP on PASCAL VOC 07*
---           |---               |---        |---      |---
EfficientNetV2-S ([OI weights](https://drive.google.com/uc?export=download&id=1N0t0eShJS3L1cDiY8HTweKfPKJ55h141) )       |448x448           | 12.28    | 88.75       | 95.86
EfficientNetV2-L                 |448x448     | 49.92    | 90.10       | 96.05
TResNet-L                        |448x448     | 36.15    | 90.30       | 96.70

*with ML-Decoder and AAM loss

## Data format
Annotation for multilabel classification is stored in json format.
Root of the json file contains a list of classes available by key `classes` and
a list of images with labels available by key `images`. Annotation of a single
image is a tuple containing a relative path to the image as a first element and a list of all
the presented labels as a second element.

Annotations and [download instructions](./datasets/README.md) for COCO14, NUS WIDE, VG500 and VOC07 can be found in the `datasets` folder.

## Environment setup

In a clean python environment install an appropriate for your GPU configuration build
of [pytorch](https://pytorch.org/get-started/locally/) 1.8.2 LTS.

Install the repository requirements: `pip install -r requirements.txt`.
If you'd like to use OpenVINO tools, install the corresponding dependencies as well: `pip install -r openvino-requirements.txt`.

After that install the training framework itself: `python setup.py develop`.

## How to run training

Prior to launching a training, download the required dataset, and prepare annotation or
use one of the predefined annotation files from the `datasets` folder.
To run a training, use the following command:
```bash
python tools/main.py --config-file configs/EfficientNetV2_small_gcn.yml --gpu-num 1 custom_datasets.roots "['<data_root>/train.json', '<data_root>/train.json']" data.save_dir <work_dir>
```

## How to run evaluation
After training is done in the working directory will appear weights of the trained model:
`<work_dir>/main_model/main_model.pth.tar-1`.
To start evaluation, run the following command:
```bash
python tools/eval.py --config-file configs/EfficientNetV2_small_gcn.yml --gpu-num 1 custom_datasets.roots "['<data_root>/train.json', '<data_root>/train.json']" model.load_weights <work_dir>/main_model/main_model.pth.tar-1
```

## How to run thresholds estimation

To run the thresholds estimation process, append `test.estimate_multilabel_thresholds True` to the eval command:
```bash
python tools/eval.py --config-file configs/EfficientNetV2_small_gcn.yml --gpu-num 1 custom_datasets.roots "['<data_root>/train.json', '<data_root>/train.json']" model.load_weights <work_dir>/main_model/main_model.pth.tar-1 test.estimate_multilabel_thresholds True
```
Both the output f1 scores are supposed to increase. An `.npy` file with the thresholds is saved to `<work_dir>/thresholds.npy`. Indexes in the output array correspond to lexicographically sorted names of the input classes (python built-in function `sorted` is used).


## How to export the resulting model to the ONNX or [OpenVINO](https://docs.openvino.ai/latest/index.html) IR format

To convert the resulting model to the ONNX format, use the corresponding script:
```bash
python tools/convert_to_onnx.py --config-file configs/EfficientNetV2_small_gcn.yml --disable-dyn-axes --output-name <model_name> custom_datasets.roots "['<data_root>/train.json', '<data_root>/train.json']" model.load_weights ./output/gan_efficientnetv2_s_21k/VOC/main_model/main_model.pth.tar-1
```
To convert the model to OpenVINO IR, append `--export_ir` to the previous command:
```bash
python tools/convert_to_onnx.py --config-file configs/EfficientNetV2_small_gcn.yml --disable-dyn-axes --export_ir --output-name <model_name> custom_datasets.roots "['<data_root>/train.json', '<data_root>/train.json']" model.load_weights ./output/gan_efficientnetv2_s_21k/VOC/main_model/main_model.pth.tar-1
```
You can also run evaluation of the resulting IR model:

```bash
python tools/eval.py --config-file configs/EfficientNetV2_small_gcn.yml custom_datasets.roots "['<data_root>/train.json', '<data_root>/train.json']" model.load_weights ./model.xml
```
The output IR model produces logits, that can be converted to class probabilities by applying sigmoid activation.
Also the IR model assumes the input image pixels to be in [0,255] range with BRG channels order (such images produces cv2.imread() for instance).


## Citation
If the presented results were useful for your paper or tech report, please cite us:
```
@ARTICLE{ProkofievSovrasovCombiningML2022,
    author = {{Prokofiev}, Kirill and {Sovrasov}, Vladislav},
    title = "{Combining Metric Learning and Attention Heads For Accurate and Efficient Multilabel Image Classification}",
    journal = {arXiv e-prints},
    year = 2022,
    month = sep,
    archivePrefix = {arXiv},
    eprint = {2209.06585},
    primaryClass = {cs.CV},
}
```
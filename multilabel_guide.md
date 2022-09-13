# **Multilabel classification tutorial**

This repository supports training models in classic miltilabel classification setup.
It contains implementations of [MLDecoder](https://arxiv.org/abs/2111.12933),
[ASL](https://arxiv.org/abs/2009.14119) and [AAM]() losses, a modified version of [ML-GCN](https://arxiv.org/abs/1904.03582).
Supported backbones table:

Model         | Input Resolution | MACs(G)* | mAP on COCO 14* | mAP on PASCAL VOC 07*
---           |---               |---        |---      |---
EfficientNetV2-S ([OI weights](https://drive.google.com/uc?export=download&id=1N0t0eShJS3L1cDiY8HTweKfPKJ55h141) )       |448x448           | 12.28    | 88.75       | 95.86
EfficientNetV2-L                 |448x448     | 49.92    | 90.10       | 96.05
TResNet-L                        |448x448     | 36.15    | 90.30       | 96.70
*with ML-Decoder and AAM loss

## Data format
Annotation for multilabel classification is stored in json format.
Root of the json file contatins a list of classes available by key `classes` and
a lsit of images with labels available by key `images`. Annotation of a single
image is a tuple containing a relative path to the image as a first element and a list of all
the presented labels as a second element.

Annotations and [download instructions](./datasets/README.md) for COCO14, NUS WIDE, VG500 and VOC07 can be found in the `datasets` folder.

## How to run training

Prior to launching a training, donwload the required dataset, and prepare annotation or
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
Both the output f1 scores are supposed to increase. An .npy file with thresholds is saved to `<work_dir>/thresholds.npy`. Indexes in the output array correspond to alphabetically sorted names of the input classes.



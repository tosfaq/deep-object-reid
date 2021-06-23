# **Deep Object Reid**

Deep Object Reid is a library for deep-learning image classification and object re-identification, written in [PyTorch](https://pytorch.org/).
It is a part of [OpenVINO™ Training Extensions](https://github.com/opencv/openvino_training_extensions).

The project is based on [Kaiyang Zhou's Torchreid](https://github.com/KaiyangZhou/deep-person-reid) project.

Its features:

- multi-GPU training
- end-to-end training and evaluation
- incredibly easy preparation of reid and classification datasets
- multi-dataset training
- cross-dataset evaluation
- standard protocol used by most research papers
- highly extensible (easy to add models, datasets, training methods, etc.)
- implementations of state-of-the-art and lightweight reid/classification models
- access to pretrained reid/classification models
- advanced training techniques such as mutual learning, RSC, SAM, AugMix and many other
- visualization tools (tensorboard, ranks, activation map, etc.)
- automated learning rate search and exiting from training (no need to choose epoch number)

How-to instructions: https://github.com/openvinotoolkit/deep-object-reid/blob/ote/docs/user_guide.rst


Original tech report by Kaiyang Zhou and Tao Xiang: https://arxiv.org/abs/1910.10093.

You can find some other research projects that are built on top of Torchreid `here (https://github.com/KaiyangZhou/deep-person-reid/tree/master/projects).

Also if you are planning to perform image classification project, please, refer to [OpenVINO™ Training Extensions Custom Image Classification Templates](https://github.com/openvinotoolkit/training_extensions/tree/develop/models/image_classification/model_templates/custom-classification) to get a strong baseline for your project. The paper is comming soon.

## **What's new**

- **[June 2021]** Added new algorithms, regularization techniques and models for image classification task
- **[May 2020]** Added the person attribute recognition code used in `Omni-Scale Feature Learning for Person Re-Identification [ICCV'19](https://arxiv.org/abs/1905.00953). See ``projects/attribute_recognition/``.
- **[May 2020]** 1.2.1: Added a simple API for [feature extraction](torchreid/utils/feature_extractor.py). See the [documentation](https://kaiyangzhou.github.io/deep-person-reid/user_guide.html) for the instruction.
- **[Apr 2020]** Code for reproducing the experiments of [deep mutual learning](https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) in the [OSNet paper](https://arxiv.org/pdf/1905.00953v6.pdf) (Supp. B) has been released at ``projects/DML``.
- **[Apr 2020]** Upgraded to 1.2.0. The engine class has been made more model-agnostic to improve extensibility. See [Engine](torchreid/engine/engine.py) and [ImageSoftmaxEngine](torchreid/engine/image/softmax.py) for more details. Credit to [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).
- **[Dec 2019]** Our [OSNet paper](https://arxiv.org/pdf/1905.00953v6.pdf) has been updated, with additional experiments (in section B of the supplementary) showing some useful techniques for improving OSNet's performance in practice.
- **[Nov 2019]** ``ImageDataManager`` can load training data from target datasets by setting ``load_train_targets=True``, and the train-loader can be accessed with ``train_loader_t = datamanager.train_loader_t``. This feature is useful for domain adaptation research.


## **Installation**
---------------

Make sure `conda (https://www.anaconda.com/distribution/) is installed.

``` bash
# cd to your preferred directory and clone this repo
git clone https://github.com/openvinotoolkit/deep-object-reid.git

# create environment
cd deep-object-reid/
conda create --name torchreid python=3.7
conda activate torchreid

# install dependencies
# make sure `which python` and `which pip` point to the correct path
pip install -r requirements.txt

# install torchreid (don't need to re-build it if you modify the source code)
python setup.py develop
```

## **Get started**
-------------------------------------

You can use deep-object-reid in your project or use this repository to train [proposed models](https://github.com/openvinotoolkit/deep-object-reid/tree/ote/torchreid/models) or your own model through configuration file.

### **Use deep-object-reid in your project**

 1. Import ``torchreid``
```python
    import torchreid
```
 2. Load data manager

```python
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)
```
 3. Build model, optimizer and lr_scheduler

```python
model = torchreid.models.build_model(
    name='osnet_ain_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='am_softmax',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.001
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)
```
 4. Build engine

```python
engine = torchreid.engine.ImageAMSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)
```
5. Run training and test

```python
engine.run(
    save_dir='log/osnet_ain',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False
)
```
### **Use deep-object-reid through configuration file**

modify one of the following [config file](https://github.com/openvinotoolkit/deep-object-reid/tree/ote/configs) and run:

```bash
python tools/main.py \
--config-file $PATH_TO_CONFIG \
--root $PATH_TO_DATA
--gpu-num $NUM_GPU
```
See "tools/main.py" and "scripts/default_config.py" for more details.

## **Evaluation**

Evaluation is automatically performed at the end of training. To run the test again using the trained model, do

```bash
python tools/eval.py \
--config-file  $PATH_TO_CONFIG\
--root $PATH_TO_DATA \
model.load_weights log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250 \
test.evaluate True
```

## **Cross-domain setting**

Suppose you wanna train OSNet on DukeMTMC-reID and test its performance on Market1501, you can do

```bash
python scripts/main.py \
--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml \
-s dukemtmcreid \
-t market1501 \
--root $PATH_TO_DATA
```
Here we only test the cross-domain performance. However, if you also want to test the performance on the source dataset, i.e. DukeMTMC-reID, you can set: `-t dukemtmcreid market1501`, which will evaluate the model on the two datasets separately.


## **Datasets**

### **Image-reid datasets**

- [Market1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)
- [CUHK03](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf)
- [DukeMTMC-reID](https://arxiv.org/abs/1701.07717)
- [MSMT17](https://arxiv.org/abs/1711.08565)
- [VIPeR](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.7285&rep=rep1&type=pdf)
- [GRID](http://www.eecs.qmul.ac.uk/~txiang/publications/LoyXiangGong_cvpr_2009.pdf)
- [CUHK01](http://www.ee.cuhk.edu.hk/~xgwang/papers/liZWaccv12.pdf)
- [SenseReID](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Spindle_Net_Person_CVPR_2017_paper.pdf)
- [QMUL-iLIDS](http://www.eecs.qmul.ac.uk/~sgg/papers/ZhengGongXiang_BMVC09.pdf)
- [PRID](https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf)

### **Classification dataset**
* [Describable Textures (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
* [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
* [Oxford 102 Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [SVHN (w/o additional data)](http://ufldl.stanford.edu/housenumbers/)
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [FOOD101](https://www.kaggle.com/dansbecker/food-101)
* [SUN397](https://vision.princeton.edu/projects/2010/SUN/)
* [Birdsnap](http://thomasberg.org/)
* [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## **Models**

### **Classification models**

- [Inception-V4](https://arxiv.org/abs/1602.07261)
- [Efficient-b0](https://arxiv.org/pdf/1905.11946)
- [MobilenetV3](https://arxiv.org/abs/1905.02244)


### **ReID-specific models**

- [OSNet](https://arxiv.org/abs/1905.00953)
- [OSNet-AIN](https://arxiv.org/abs/1910.06827)

### **Face Recognition specific models**
- [MobileFaceNet](https://arxiv.org/abs/1804.07573)

## **Useful links**
- [OSNet-IBN1-Lite](https://github.com/RodMech/OSNet-IBN1-Lite) (test-only code with lite docker container)
- [Deep Learning for Person Re-identification: A Survey and Outlook](https://github.com/mangye16/ReID-Survey)
- [OpenVINO™ Training Extention](https://github.com/openvinotoolkit/training_extensions)


## **Citation**

If you find this code useful to your research, please cite the following papers.

```bash
@article{torchreid,
    title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
    author={Zhou, Kaiyang and Xiang, Tao},
    journal={arXiv preprint arXiv:1910.10093},
    year={2019}
}

@inproceedings{zhou2019osnet,
    title={Omni-Scale Feature Learning for Person Re-Identification},
    author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
    booktitle={ICCV},
    year={2019}
}

@article{zhou2019learning,
    title={Learning Generalisable Omni-Scale Representations for Person Re-Identification},
    author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
    journal={arXiv preprint arXiv:1910.06827},
    year={2019}
}
```
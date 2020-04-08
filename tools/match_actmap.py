import os.path as osp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, REMAINDER

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchreid
from torchreid.utils import check_isfile, mkdir_if_missing, load_pretrained_weights
from torchreid.data.datasets import init_image_dataset
from torchreid.data.transforms import build_transforms
from scripts.default_config import imagedata_kwargs, get_default_config, model_kwargs

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root


def build_dataset(mode='gallery', targets=None, height=192, width=256,
                  transforms=None, norm_mean=None, norm_std=None, **kwargs):
    _, transform_test = build_transforms(
        height,
        width,
        transforms=transforms,
        norm_mean=norm_mean,
        norm_std=norm_std
    )

    main_target_name = targets[0]
    dataset = init_image_dataset(
        main_target_name,
        transform=transform_test,
        mode=mode,
        verbose=False,
        **kwargs
    )

    return dataset


def build_data_loader(dataset, use_gpu=True, batch_size=300):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=use_gpu,
        drop_last=False
    )

    return data_loader


def prepare_data(cfg, mode='query'):
    data_config = imagedata_kwargs(cfg)
    dataset = build_dataset(mode=mode, **data_config)
    data_loader = build_data_loader(dataset, use_gpu=cfg.use_gpu)

    return data_loader, dataset.num_train_pids


@torch.no_grad()
def collect_features_maps(model, data_loader, use_gpu, img_mean=None, img_std=None, normalize=True):
    if img_mean is None or img_std is None:
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    all_images, all_features_maps, all_labels = [], [], []
    for batch_idx, data in enumerate(data_loader):
        imgs, labels = data[0], data[1]
        if use_gpu:
            imgs = imgs.cuda()

        try:
            outputs = model(imgs, return_featuremaps=True)
        except TypeError:
            raise TypeError(
                'forward() got unexpected keyword argument "return_featuremaps". '
                'Please add return_featuremaps as an input argument to forward(). When '
                'return_featuremaps=True, return feature maps only.'
            )

        if outputs.dim() != 4:
            raise ValueError(
                'The model output is supposed to have '
                'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                'Please make sure you set the model output at eval mode '
                'to be the last convolutional feature maps'.format(
                    outputs.dim()
                )
            )

        if normalize:
            outputs = F.normalize(outputs, p=2, dim=1)

        if use_gpu:
            imgs, labels, outputs = imgs.cpu(), labels.cpu().numpy(), outputs.cpu().numpy()

        for j in range(imgs.size(0)):
            img = imgs[j, ...]
            for t, m, s in zip(img, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            all_images.append(img_np)
            all_features_maps.append(outputs[j])
            all_labels.append(labels[j])

        if (batch_idx + 1) % 10 == 0:
            print('- done batch {}/{}'.format(batch_idx + 1, len(data_loader)))

    return all_images, all_features_maps, all_labels


def visualize_matches(images, features_maps, labels):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        class_mask = labels == label
        class_positions = np.arange(len(labels))[class_mask]

        class_images = [images[p] for p in class_positions]
        class_features_maps = [features_maps[p] for p in class_positions]

        image_a = class_images[0]
        image_b = class_images[-1]
        img_h, img_w = image_a.shape[:2]

        features_a = class_features_maps[0]
        features_b = class_features_maps[-1]

        c, h, w = features_a.shape
        act_map = np.matmul(np.transpose(features_a.reshape([c, -1])), features_b.reshape([c, -1]))

        exp_values = np.exp(act_map)
        attention_map = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        att_values = np.max(attention_map, axis=1)
        att_pos = np.argmax(attention_map, axis=1)

        top_att_values_pos = np.argsort(-att_values)[:5]

        matches = []
        for src_pos in top_att_values_pos:
            trg_pos = att_pos[src_pos]

            src_coordinate = src_pos / w, src_pos % w
            trg_coordinate = trg_pos / w, trg_pos % w

            scaled_src = int(src_coordinate[0] / float(h) * img_h), int(src_coordinate[1] / float(w) * img_w)
            scaled_trg = int(trg_coordinate[0] / float(h) * img_h), int(trg_coordinate[1] / float(w) * img_w)

            matches.append((scaled_src, scaled_trg))
            print(scaled_src, scaled_trg)

        plt.imshow(attention_map)
        plt.show()


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--root', '-r', type=str, required=True)
    parser.add_argument('opts', default=None, nargs=REMAINDER)
    args = parser.parse_args()

    assert osp.exists(args.config_file)
    assert osp.exists(args.root)

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    data_loader, num_pids = prepare_data(cfg, mode='query')

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_pids))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = model.cuda()

    images, features_maps, labels = collect_features_maps(model, data_loader, cfg.use_gpu, normalize=True)
    visualize_matches(images, features_maps, labels)


if __name__ == '__main__':
    main()

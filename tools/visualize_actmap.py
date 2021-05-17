"""Visualizes CNN activation maps to see where the CNN focuses on to extract features.

Reference:
    - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
      performance of convolutional neural networks via attention transfer. ICLR, 2017
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
"""

import os.path as osp
from argparse import REMAINDER, ArgumentDefaultsHelpFormatter, ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scripts.default_config import (get_default_config, imagedata_kwargs,
                                    model_kwargs, merge_from_files_with_base)

import torchreid
from torchreid.data.datasets import init_image_dataset
from torchreid.data.transforms import build_transforms
from torchreid.utils import (check_isfile, load_pretrained_weights,
                             mkdir_if_missing)

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


def build_data_loader(dataset, use_gpu=True, batch_size=100):
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

    pids = dataset.num_train_pids
    keys = sorted(pids.keys())
    pids = [pids[key] for key in keys]

    return data_loader, pids


@torch.no_grad()
def visualize_activation_map(model, data_loader, save_dir, width, height, use_gpu, img_mean=None, img_std=None):
    if img_mean is None or img_std is None:
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    # original images and activation maps are saved individually
    actmap_dir = osp.join(save_dir, 'actmap')
    mkdir_if_missing(actmap_dir)

    for batch_idx, data in enumerate(data_loader):
        imgs, paths = data[0], data[3]
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

        # compute activation maps
        outputs = outputs.mean(dim=1)
        b, h, w = outputs.size()
        outputs = outputs.view(b, h * w)
        outputs = F.softmax(outputs, dim=1)
        # outputs = F.normalize(outputs, p=2, dim=1)
        outputs = outputs.view(b, h, w)

        if use_gpu:
            imgs, outputs = imgs.cpu(), outputs.cpu()

        for j in range(outputs.size(0)):
            # get image name
            path = paths[j]
            imname = osp.basename(osp.splitext(path)[0])

            # RGB image
            img = imgs[j, ...]
            for t, m, s in zip(img, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(img.numpy() * 255))
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # activation map
            am = outputs[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

            # overlapped
            overlapped = img_np * 0.3 + am * 0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)

            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones((height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8)
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:, width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
            cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

        if (batch_idx + 1) % 10 == 0:
            print('- done batch {}/{}'.format(batch_idx + 1, len(data_loader)))


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--root', '-r', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='log')
    parser.add_argument('opts', default=None, nargs=REMAINDER)
    args = parser.parse_args()

    assert osp.exists(args.config_file)
    assert osp.exists(args.root)

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        merge_from_files_with_base(cfg, args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    data_loader, num_pids = prepare_data(cfg, mode='gallery')

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_pids))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = model.cuda()

    visualize_activation_map(model, data_loader, args.save_dir, cfg.data.width, cfg.data.height, cfg.use_gpu)


if __name__ == '__main__':
    main()

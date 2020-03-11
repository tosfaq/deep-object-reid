import os.path as osp
import argparse
from os import makedirs
from shutil import rmtree
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import torchreid
from torchreid.utils import check_isfile, load_pretrained_weights
from torchreid.data.datasets import init_image_dataset
from torchreid.data.transforms import build_transforms
from scripts.default_config import imagedata_kwargs, get_default_config, model_kwargs


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root


def build_dataset(mode='gallery', targets=None, height=256, width=128, transforms='random_flip',
                  norm_mean=None, norm_std=None, **kwargs):
    _, transform_test = build_transforms(
        height,
        width,
        transforms=transforms,
        norm_mean=norm_mean,
        norm_std=norm_std
    )

    main_dataset_name = targets[0]
    dataset = init_image_dataset(
        main_dataset_name,
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


def extract_probs(model, data_loader, use_gpu):
    model.eval()

    out_probs = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(data_loader):
            images = data[0]
            if use_gpu:
                images = images.cuda()

            _, logits_dict = model(images, return_logits=True)
            for name, logits in logits_dict.items():
                out_probs[name].append(F.softmax(2.5 * logits, dim=-1).data.cpu())

    for name, probs in out_probs.items():
        out_probs[name] = torch.cat(out_probs[name], 0).numpy()

    return out_probs


def extract_labels(probs):
    return np.argmax(probs, axis=-1)


def dump(dataset, labels, out_dir):
    if osp.exists(out_dir):
        rmtree(out_dir)
    makedirs(out_dir)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_dir = osp.join(out_dir, str(label))
        makedirs(label_dir)

    data_loader = build_data_loader(dataset, use_gpu=False, batch_size=1)

    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, -1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, -1)

    with torch.no_grad():
        for data_id, data in tqdm(enumerate(data_loader)):
            label = labels[data_id]

            float_image = np.transpose(data[0].data.cpu().numpy()[0], (1, 2, 0))
            image = (255.0 * (float_image * std + mean)).astype(np.uint8)

            image_path = osp.join(out_dir, str(label), 'img_{:06}.jpg'.format(data_id))
            cv2.imwrite(image_path, image[:, :, ::-1])


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--root', '-r', type=str, required=True)
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    dataset = build_dataset(mode='query', **imagedata_kwargs(cfg))
    data_loader = build_data_loader(dataset, use_gpu=cfg.use_gpu)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, dataset.num_train_pids))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    all_probs = extract_probs(model, data_loader, cfg.use_gpu)
    print('Extracted probs:')
    for name, probs in all_probs.items():
        print('   {}: {}'.format(name, probs.shape))
        print('p0: [' + ', '.join(['{:.4f}'.format(p) for p in probs[0].tolist()]) + ']')
        print('p1: [' + ', '.join(['{:.4f}'.format(p) for p in probs[1].tolist()]) + ']')
        print('p2: [' + ', '.join(['{:.4f}'.format(p) for p in probs[2].tolist()]) + ']')
        dump(dataset, extract_labels(probs), osp.join(args.out_dir, name))


if __name__ == '__main__':
    main()

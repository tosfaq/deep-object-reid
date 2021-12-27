"""
 Copyright (c) 2020-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from argparse import REMAINDER, ArgumentDefaultsHelpFormatter, ArgumentParser
from os import makedirs
from os.path import exists, join
from shutil import rmtree

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scripts.default_config import (get_default_config, imagedata_kwargs,
                                    model_kwargs, merge_from_files_with_base)
from tqdm import tqdm, trange

import torchreid
from torchreid.data.datasets import init_image_dataset
from torchreid.data.transforms import build_test_transform
from torchreid.utils import load_pretrained_weights

GRID_SPACING = 4
ANCHOR_COLOR = (0, 0, 0)
TRUE_COLOR = (0, 255, 0)
FALSE_COLOR = (0, 0, 255)


def create_dirs(dir_path):
    if exists(dir_path):
        rmtree(dir_path)

    makedirs(dir_path)


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root


def build_dataset(mode='gallery', target_name='veri', height=192, width=256, norm_mean=None, norm_std=None, **kwargs):
    transform = build_test_transform(
        height,
        width,
        norm_mean=norm_mean,
        norm_std=norm_std
    )

    dataset = init_image_dataset(
        target_name,
        transform=transform,
        mode=mode,
        verbose=False,
        **kwargs
    )

    return dataset


def build_data_loader(dataset, use_gpu=True, batch_size=20):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=use_gpu,
        drop_last=False
    )

    return data_loader


def build_query(cfg, target_name):
    data_config = imagedata_kwargs(cfg)
    dataset = build_dataset(mode='query', target_name=target_name, **data_config)
    data_loader = build_data_loader(dataset, use_gpu=cfg.use_gpu)

    return data_loader


def build_gallery(cfg, target_name):
    data_config = imagedata_kwargs(cfg)
    dataset = build_dataset(mode='gallery', target_name=target_name, **data_config)
    data_loader = build_data_loader(dataset, use_gpu=cfg.use_gpu)

    return data_loader, len(dataset)


def run_model(model, data_loader, use_gpu):
    model.eval()

    out_images = []
    out_embeddings = []
    out_ids = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            images = data[0]
            vehicle_ids = data[1]

            if use_gpu:
                images = images.cuda()

            embeddings = model(images)
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)

            out_images.append(images.float().permute(0, 2, 3, 1).data.cpu())
            out_embeddings.append(norm_embeddings.float().data.cpu())
            out_ids.append(vehicle_ids.float().data.cpu())

        std = torch.from_numpy(np.array([0.229, 0.224, 0.225], dtype=np.float32())).view(1, 1, 1, -1)
        mean = torch.from_numpy(np.array([0.485, 0.456, 0.406], dtype=np.float32())).view(1, 1, 1, -1)
        out_images = torch.cat(out_images, dim=0)
        out_images = (out_images * std + mean).clamp(0, 1)
        out_images = np.floor(out_images.numpy() * 255).astype(np.uint8)

        out_embeddings = torch.cat(out_embeddings, dim=0).numpy()
        out_ids = torch.cat(out_ids, dim=0).numpy()

    return out_images, out_embeddings, out_ids


def calculate_distances(a, b):
    return 1.0 - np.matmul(a, np.transpose(b))


def find_matches(dist_matrix, top_k=100):
    indices = np.argsort(dist_matrix, axis=1)
    out_matches = indices[:, :top_k]

    return out_matches


def visualize_matches(all_matches, images_query, images_gallery, ids_query, ids_gallery, matrix_size, out_dir):
    height, width = images_query.shape[1:3]
    cell_height, cell_width = height + 2 * GRID_SPACING, width + 2 * GRID_SPACING
    grid_height, grid_width = matrix_size * cell_height, matrix_size * cell_width

    for query_sample_id in trange(all_matches.shape[0]):
        query_id = ids_query[query_sample_id]
        query_image = images_query[query_sample_id]
        matches = all_matches[query_sample_id]
        max_num_matches = np.sum(query_id == ids_gallery)

        grid_img = np.full((grid_height, grid_width, 3), 255, dtype=np.uint8)

        num_valid = 0
        for i in range(matrix_size * matrix_size):
            if i == 0:
                image = query_image
                color = ANCHOR_COLOR
            else:
                gallery_sample_id = matches[i - 1]
                image = images_gallery[gallery_sample_id]

                gallery_id = ids_gallery[gallery_sample_id]
                color = TRUE_COLOR if query_id == gallery_id else FALSE_COLOR
                num_valid += query_id == gallery_id

            row = int(i / matrix_size)
            col = int(i % matrix_size)

            cell_top = row * cell_height
            cell_down = (row + 1) * cell_height
            cell_left = col * cell_width
            cell_right = (col + 1) * cell_width
            grid_img[cell_top:cell_down, cell_left:cell_right] = color

            image_top = cell_top + GRID_SPACING
            image_down = image_top + height
            image_left = cell_left + GRID_SPACING
            image_right = image_left + width
            grid_img[image_top:image_down, image_left:image_right] = image[:, :, ::-1]

            if num_valid >= max_num_matches:
                break

        out_path = join(out_dir, 'img_{:04}.jpg'.format(query_sample_id))
        cv2.imwrite(out_path, grid_img)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--weights', '-w', type=str, required=True)
    parser.add_argument('--root', '-r', type=str, required=True)
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    parser.add_argument('--matrix-size', '-ms', type=int, required=False, default=8)
    parser.add_argument('opts', default=None, nargs=REMAINDER)
    args = parser.parse_args()

    assert exists(args.config_file)
    assert exists(args.weights)
    assert exists(args.root)

    create_dirs(args.out_dir)

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        merge_from_files_with_base(cfg, args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    target_dataset = 'cityflow'
    data_query = build_query(cfg, target_dataset)
    data_gallery, gallery_size = build_gallery(cfg, target_dataset)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, [0]))

    print('Loading model: {}'.format(cfg.model.load_weights))
    cfg.model.load_weights = args.weights
    load_pretrained_weights(model, cfg.model.load_weights)
    model = model.cuda() if cfg.use_gpu else model

    print('Extracting query embeddings ...')
    images_query, embeddings_query, ids_query = run_model(model, data_query, cfg.use_gpu)

    print('Extracting gallery embeddings ...')
    images_gallery, embeddings_gallery, ids_gallery = run_model(model, data_gallery, cfg.use_gpu)

    print('Calculating distance matrices ...')
    distance_matrix_qg = calculate_distances(embeddings_query, embeddings_gallery)

    print('Finding matches ...')
    top_k = args.matrix_size ** 2 - 1
    matches = find_matches(distance_matrix_qg, top_k=top_k)

    print('Dumping visualizations ...')
    visualize_matches(matches,
                      images_query, images_gallery,
                      ids_query, ids_gallery,
                      args.matrix_size, args.out_dir)


if __name__ == '__main__':
    main()

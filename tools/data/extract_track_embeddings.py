import os.path as osp
import argparse
from os import makedirs
from shutil import rmtree

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchreid
from torchreid.utils import check_isfile, load_pretrained_weights
from torchreid.data.datasets import init_image_dataset
from torchreid.data.transforms import build_transforms
from scripts.default_config import imagedata_kwargs, get_default_config, model_kwargs


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root


def build_dataset(root='', targets=None, height=256, width=128, transforms='random_flip', norm_mean=None,
                  norm_std=None, split_id=0, combineall=False, cuhk03_labeled=False, cuhk03_classic_split=False,
                  market1501_500k=False, **kwargs):
    _, transform_test = build_transforms(
        height,
        width,
        transforms=transforms,
        norm_mean=norm_mean,
        norm_std=norm_std
    )

    assert len(targets) == 1
    name = targets[0]

    dataset = init_image_dataset(
        name,
        transform=transform_test,
        mode='gallery',
        combineall=combineall,
        verbose=False,
        root=root,
        split_id=split_id,
        cuhk03_labeled=cuhk03_labeled,
        cuhk03_classic_split=cuhk03_classic_split,
        market1501_500k=market1501_500k
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


def extract_features(model, data_loader, use_gpu):
    model.eval()

    f = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            images = data[0]
            if use_gpu:
                images = images.cuda()

            model_out = model(images)
            f.append(model_out.data.cpu())

    f = torch.cat(f, 0).numpy()

    return f


def load_tracks(annot_path):
    tracks = []
    glob_id = 0
    for line in open(annot_path):
        file_names = line.replace('\n', '').split(' ')
        file_names = [n for n in file_names if len(n) > 0]
        assert len(file_names) > 1

        indexed_names = []
        for file_name in file_names:
            indexed_names.append((file_name, glob_id))
            glob_id += 1

        tracks.append(indexed_names)

    return tracks


def estimate_track_centers(tracks, all_embeddings):
    assert len([r for t in tracks for r in t]) == all_embeddings.shape[0]

    out_centers = []
    for track in tracks:
        track_ids = [record[1] for record in track]
        track_embeddings = all_embeddings[track_ids]

        center = np.mean(track_embeddings, axis=0, keepdims=True)

        norm_center = center / np.sqrt(np.sum(np.square(center), axis=1, keepdims=True))
        norm_track_embeddings = track_embeddings / np.sqrt(np.sum(np.square(track_embeddings), axis=1, keepdims=True))

        similarities = np.matmul(norm_track_embeddings, np.transpose(norm_center)).reshape(-1)
        threshold = np.percentile(similarities, 75.0)

        filtered_embeddings = norm_track_embeddings[similarities > threshold]
        filtered_center = np.mean(filtered_embeddings, axis=0, keepdims=True)
        norm_filtered_center = filtered_center / np.sqrt(np.sum(np.square(filtered_center), axis=1, keepdims=True))

        out_centers.append(norm_filtered_center)

    return np.concatenate(out_centers, axis=0)


def print_centers_stat(similarities):
    indices = np.triu_indices(similarities.shape[0], 1)
    values = similarities[indices]
    print('Centers stat:')
    print('   min: {:.3f}, max: {:.3f}'.format(np.min(values),
                                               np.max(values)))
    print('   p@5: {:.3f}, p@50: {:.3f}, p@95: {:.3f}'.format(np.percentile(values, 5.0),
                                                              np.percentile(values, 50.0),
                                                              np.percentile(values, 95.0)))


def get_connections_graph(similarities, threshold):
    adjacency_matrix = np.triu(similarities > threshold, 1)
    graph = nx.from_numpy_matrix(adjacency_matrix)

    return graph


def show_graph(graph):
    nx.draw_networkx(graph, arrows=False, with_labels=True)
    plt.show()


def find_connected_components(graph):
    return [list(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]


def print_connected_components_stat(components):
    print('Connected components stat:')
    print('   num components: {}'.format(len(components)))

    lengths = [len(c) for c in components]
    print('   sizes min: {}, max: {}'.format(np.min(lengths),
                                             np.max(lengths)))
    print('   sizes p@5: {}, p@50: {}, p@95: {}'.format(np.percentile(lengths, 5.0),
                                                        np.percentile(lengths, 50.0),
                                                        np.percentile(lengths, 95.0)))


def dump_classes(components, dataset, out_dir):
    if osp.exists(out_dir):
        rmtree(out_dir)
    makedirs(out_dir)

    data_loader = build_data_loader(dataset, use_gpu=False, batch_size=1)

    sufficient_components = [c for c in components if len(c) > 1]
    print('Num sufficient components: {}'.format(len(sufficient_components)))

    sample_ids = set(sample_id for c in sufficient_components for sample_id in c)

    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, -1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, -1)

    out_images = {}
    with torch.no_grad():
        for data_id, data in tqdm(enumerate(data_loader)):
            if data_id not in sample_ids:
                continue

            float_image = np.transpose(data[0].data.cpu().numpy()[0], (1, 2, 0))
            image = 255.0 * (float_image * std + mean)

            out_images[data_id] = image.astype(np.uint8)

    for comp_id, component in enumerate(sufficient_components):
        comp_dir = osp.join(out_dir, 'c{:03}'.format(comp_id))
        makedirs(comp_dir)

        for sample_id in component:
            image = out_images[sample_id]

            image_path = osp.join(comp_dir, 'img_{:06}.jpg'.format(sample_id))
            cv2.imwrite(image_path, image)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--root', '-r', type=str, required=True)
    parser.add_argument('--tracks-annot', '-t', type=str, required=True)
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert osp.exists(args.tracks_annot)

    tracks = load_tracks(args.tracks_annot)

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    dataset = build_dataset(**imagedata_kwargs(cfg))
    data_loader = build_data_loader(dataset, use_gpu=cfg.use_gpu)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, 2))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    features = extract_features(model, data_loader, cfg.use_gpu)
    print('Extracted features: {}'.format(features.shape))

    track_centers = estimate_track_centers(tracks, features)
    print('Centers: {}'.format(track_centers.shape))

    similarity_matrix = np.matmul(track_centers, np.transpose(track_centers))
    print_centers_stat(similarity_matrix)

    connections_graph = get_connections_graph(similarity_matrix, 0.95)
    connected_components = find_connected_components(connections_graph)
    print_connected_components_stat(connected_components)

    dump_classes(connected_components, dataset, args.out_dir)

    # show_graph(connections_graph)


if __name__ == '__main__':
    main()

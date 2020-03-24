from os.path import exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, REMAINDER

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import torchreid
from torchreid.utils import check_isfile, load_pretrained_weights, re_ranking
from torchreid.data.datasets import init_image_dataset
from torchreid.data.transforms import build_transforms
from scripts.default_config import imagedata_kwargs, get_default_config, model_kwargs


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


def build_query(cfg):
    data_config = imagedata_kwargs(cfg)
    dataset = build_dataset(mode='query', **data_config)
    data_loader = build_data_loader(dataset, use_gpu=cfg.use_gpu)

    return data_loader, dataset.num_train_pids


def build_gallery(cfg):
    data_config = imagedata_kwargs(cfg)
    dataset = build_dataset(mode='gallery', **data_config)
    data_loader = build_data_loader(dataset, use_gpu=cfg.use_gpu)

    return data_loader, len(dataset)


def extract_features(model, data_loader, use_gpu, enable_flipping=True):
    model.eval()

    out_embeddings = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            images = data[0]
            if use_gpu:
                images = images.cuda()

            embeddings = model(images)

            if enable_flipping:
                flipped_images = torch.flip(images, dims=[3])
                flipped_embeddings = model(flipped_images)
                embeddings = 0.5 * (embeddings + flipped_embeddings)

            norm_embeddings = F.normalize(embeddings, dim=-1)

            out_embeddings.append(norm_embeddings.data.cpu())

    out_embeddings = torch.cat(out_embeddings, 0).numpy()

    return out_embeddings


def calculate_distances(a, b):
    return 1.0 - np.matmul(a, np.transpose(b))


def cluster_samples(distance_matrix, min_num_components, data_loader):
    assert len(distance_matrix.shape) == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]

    num_samples = distance_matrix.shape[0]

    distances = [(distance_matrix[i, j], i, j) for i in range(num_samples) for j in range(i + 1, num_samples)]
    distances = [tup for tup in distances if tup[0] < 0.1]
    distances.sort(key=lambda tup: tup[0])

    G = nx.Graph()
    G.add_nodes_from(list(range(num_samples)))

    for d, i, j in distances:
        G.add_edge(i, j)

        connected_components = find_connected_components(G)
        print('d: {:.4f} num: {}'.format(d, len(connected_components)))
        if len(connected_components) <= min_num_components:
            break

    connected_components = find_connected_components(G)
    print_connected_components_stat(connected_components)
    exit()

    # large_components = [c for c in connected_components if len(c) > 1]
    # print('Num large components: {}'.format(len(large_components)))
    #
    # import cv2
    # from os import makedirs
    # from shutil import rmtree
    # from os.path import exists, join
    # out_dir = '/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_queue_v70'
    # if exists(out_dir):
    #     rmtree(out_dir)
    # makedirs(out_dir)
    #
    # image_ids_map = dict()
    # for comp_id, comp in enumerate(connected_components):
    #     if len(comp) <= 1:
    #         continue
    #
    #     for i in comp:
    #         image_ids_map[i] = comp_id
    #
    #     comp_dir = join(out_dir, str(comp_id))
    #     makedirs(comp_dir)
    #
    # image_ids = set(image_ids_map.keys())
    # print('Num images: {}'.format(len(image_ids)))
    #
    # std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1)
    # mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1)
    # with torch.no_grad():
    #     last_data_id = 0
    #     for data in tqdm(data_loader):
    #         float_images = np.transpose(data[0].data.cpu().numpy(), (0, 2, 3, 1))
    #         images = (255.0 * (float_images * std + mean)).astype(np.uint8)
    #
    #         for image_id, image in enumerate(images):
    #             data_id = last_data_id + image_id
    #             if data_id not in image_ids:
    #                 continue
    #
    #             comp_id = image_ids_map[data_id]
    #             comp_dir = join(out_dir, str(comp_id))
    #
    #             image_path = join(comp_dir, 'img_{:06}.jpg'.format(data_id))
    #             cv2.imwrite(image_path, image[:, :, ::-1])
    #
    #         last_data_id += images.shape[0]
    #
    # exit()


def get_connections_graph(distances, threshold):
    similarities = 1.0 - distances
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


def load_tracks(file_path, gallery_size):
    tracks = []
    for line in open(file_path):
        str_values = [s for s in line.replace('\n', '').split(' ') if len(s) > 0]
        ids = [int(s) - 1 for s in str_values]
        assert len(ids) > 0

        for sample_id in ids:
            assert 0 <= sample_id < gallery_size

        tracks.append(ids)

    track_ids = [sample_id for track in tracks for sample_id in track]
    assert len(track_ids) == len(set(track_ids))

    rest_ids = set(range(gallery_size)) - set(track_ids)
    print('Num gallery images without track info: {} / {}'.format(len(rest_ids), gallery_size))
    for rest_id in rest_ids:
        tracks.append([rest_id])

    assert sum([len(track) for track in tracks]) == gallery_size

    return tracks


def calculate_track_distances(distance_matrix, tracks):
    track_distances = []
    for track_ids in tracks:
        distances = distance_matrix[:, track_ids]
        group_distance = np.percentile(distances, 10, axis=1)
        track_distances.append(group_distance.reshape([-1, 1]))
    track_distances = np.concatenate(tuple(track_distances), axis=1)

    return track_distances


def find_matches(distance_matrix, tracks, top_k=100, enable_track_info=True):
    if not enable_track_info:
        return np.argsort(distance_matrix, axis=-1)[:, :top_k]

    track_distances = []
    for track_ids in tracks:
        distances = distance_matrix[:, track_ids]
        group_distance = np.percentile(distances, 10, axis=1)
        track_distances.append(group_distance.reshape([-1, 1]))
    track_distances = np.concatenate(tuple(track_distances), axis=1)

    track_indices = np.argsort(track_distances, axis=1)
    # track_values = np.sort(track_distances, axis=1)

    # num_duplicates = []
    # for g_id in range(track_indices.shape[1]):
    #     indices = track_indices[:, g_id]
    #     unique_values, unique_counts = np.unique(indices, return_counts=True)
    #     print('{:<4}:'.format(g_id), [c for c in unique_counts if c > 1])
    #     num_duplicates.append(track_indices.shape[0] - len(unique_values))
    # print(num_duplicates)
    # exit()

    candidates = set(range(track_distances.shape[1]))

    out_matches = []
    for q_id in range(distance_matrix.shape[0]):
        # top_values = track_values[q_id, : 40]
        # top_values_str = '[' + ' '.join(['{:.3f}'.format(v) for v in top_values]) + ']\n'
        # print(top_values_str)

        ids = []
        for track_id in track_indices[q_id]:
            ids.extend(tracks[int(track_id)])

        out_matches.append(ids)

    return np.array(out_matches)[:, :top_k]


def dump_matches(matches, out_file):
    shifted_matches = matches + 1

    with open(out_file, 'w') as out_stream:
        for row in shifted_matches:
            line = ' '.join(map(str, row.reshape([-1]).tolist()))
            out_stream.write(line + '\n')


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--root', '-r', type=str, required=True)
    parser.add_argument('--tracks-file', '-t', type=str, required=True)
    parser.add_argument('--out-file', '-o', type=str, required=True)
    parser.add_argument('opts', default=None, nargs=REMAINDER)
    args = parser.parse_args()

    assert exists(args.config_file)
    assert exists(args.root)
    assert exists(args.tracks_file)

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    data_query, num_pids = build_query(cfg)
    data_gallery, gallery_size = build_gallery(cfg)

    gallery_tracks = load_tracks(args.tracks_file, gallery_size)
    print('Loaded tracks: {}'.format(len(gallery_tracks)))

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_pids))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = model.cuda()

    embeddings_query = extract_features(model, data_query, cfg.use_gpu, enable_flipping=True)
    print('Extracted query: {}'.format(embeddings_query.shape))

    embeddings_gallery = extract_features(model, data_gallery, cfg.use_gpu, enable_flipping=True)
    print('Extracted gallery: {}'.format(embeddings_gallery.shape))

    print('Calculating distance matrices ...')
    distance_matrix_qg = calculate_distances(embeddings_query, embeddings_gallery)
    distance_matrix_qq = calculate_distances(embeddings_query, embeddings_query)
    distance_matrix_gg = calculate_distances(embeddings_gallery, embeddings_gallery)

    # cluster_samples(distance_matrix_qq, 333, data_query)

    print('Applying re-ranking ...')
    distance_matrix_qg = re_ranking(distance_matrix_qg, distance_matrix_qq, distance_matrix_gg,
                                    k1=50, k2=15, lambda_value=0.1)
    print('Distance matrix after re-ranking: {}'.format(distance_matrix_qg.shape))

    # calculate_track_distances(distance_matrix_qg)

    matches = find_matches(distance_matrix_qg, gallery_tracks, top_k=100, enable_track_info=True)
    print('Matches: {}'.format(matches.shape))

    dump_matches(matches, args.out_file)
    print('Submission file has been stored at: {}'.format(args.out_file))


if __name__ == '__main__':
    main()

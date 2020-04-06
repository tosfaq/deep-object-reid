from os.path import exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, REMAINDER
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchreid
from torchreid.utils import check_isfile, load_pretrained_weights, re_ranking
from torchreid.data.datasets import init_image_dataset
from torchreid.data.transforms import build_submission_transforms
from scripts.default_config import imagedata_kwargs, get_default_config, model_kwargs


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root


def build_dataset(mode='gallery', targets=None, height=192, width=256, norm_mean=None, norm_std=None, **kwargs):
    transform = build_submission_transforms(
        height,
        width,
        norm_mean=norm_mean,
        norm_std=norm_std
    )

    main_target_name = targets[0]
    dataset = init_image_dataset(
        main_target_name,
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


def extract_features(model, data_loader, use_gpu):
    model.eval()

    out_embeddings = []
    with torch.no_grad():

        for data in tqdm(data_loader):
            images = data[0]

            b, n, c, h, w = images.size()
            images = images.view(b * n, c, h, w)
            if use_gpu:
                images = images.cuda()

            all_embeddings = model(images).view(b, n, -1)

            left_embeddings = all_embeddings[:, :int(n / 2)]
            right_embeddings = all_embeddings[:, int(n / 2):]
            mean_embeddings = 0.5 * (left_embeddings + right_embeddings)

            final_embeddings = mean_embeddings.view(b, -1)
            norm_embeddings = F.normalize(final_embeddings, dim=1)

            out_embeddings.append(norm_embeddings.data.cpu())

    out_embeddings = torch.cat(out_embeddings, 0).numpy()

    return out_embeddings


def calculate_distances(a, b):
    return 1.0 - np.matmul(a, np.transpose(b))


def merge_query_samples(distance_matrix, max_distance, data_loader=None):
    assert len(distance_matrix.shape) == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]

    num_samples = distance_matrix.shape[0]

    distances = [(distance_matrix[i, j], i, j) for i in range(num_samples) for j in range(i + 1, num_samples)]
    distances = [tup for tup in distances if tup[0] < max_distance]
    distances.sort(key=lambda tup: tup[0])

    G = nx.Graph()
    G.add_nodes_from(list(range(num_samples)))

    for _, i, j in distances:
        G.add_edge(i, j)
    print('Added {} edges'.format(len(distances)))

    connected_components = find_connected_components(G)
    print_connected_components_stat(connected_components, 'Query')

    large_components = [c for c in connected_components if len(c) > 1]
    print('Num large components: {}'.format(len(large_components)))

    image_ids = set(i for comp in large_components for i in comp)
    print('Num clustered images: {} / {}'.format(len(image_ids), num_samples))

    if data_loader is None:
        return connected_components

    import cv2
    from os import makedirs
    from shutil import rmtree
    from os.path import exists, join
    out_dir = '/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_v142/query'
    if exists(out_dir):
        rmtree(out_dir)
    makedirs(out_dir)

    image_ids_map = dict()
    for comp_id, comp in enumerate(connected_components):
        if len(comp) <= 1:
            continue

        for i in comp:
            image_ids_map[i] = comp_id

        comp_dir = join(out_dir, str(comp_id))
        makedirs(comp_dir)

    image_ids = set(image_ids_map.keys())

    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1)
    with torch.no_grad():
        last_data_id = 0
        for data in tqdm(data_loader):
            float_images = np.transpose(data[0].data[:, 0].cpu().numpy(), (0, 2, 3, 1))
            images = (255.0 * (float_images * std + mean)).astype(np.uint8)

            for image_id, image in enumerate(images):
                data_id = last_data_id + image_id
                if data_id not in image_ids:
                    continue

                comp_id = image_ids_map[data_id]
                comp_dir = join(out_dir, str(comp_id))

                image_path = join(comp_dir, 'img_{:06}.jpg'.format(data_id))
                cv2.imwrite(image_path, image[:, :, ::-1])

            last_data_id += images.shape[0]

    return connected_components


def merge_gallery_tracklets(distance_matrix, max_distance, src_tracklets, data_loader=None):
    assert len(distance_matrix.shape) == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]

    num_samples = distance_matrix.shape[0]

    G = nx.Graph()
    G.add_nodes_from(list(range(num_samples)))

    free_pairs = np.ones([num_samples, num_samples], dtype=np.bool)
    for tracklet in src_tracklets:
        tracklet_ids = list(sorted(tracklet))
        tracklet_size = len(tracklet_ids)
        for i in range(tracklet_size):
            anchor_id = tracklet_ids[i]
            for j in range(i + 1, tracklet_size):
                ref_id = tracklet_ids[j]

                G.add_edge(anchor_id, ref_id)
                free_pairs[anchor_id, ref_id] = False

    init_connected_components = find_connected_components(G)
    print_connected_components_stat(init_connected_components, 'Gallery (before)')

    distances = [(distance_matrix[i, j], i, j)
                 for i in range(num_samples)
                 for j in range(i + 1, num_samples)
                 if free_pairs[i, j]]
    distances = [tup for tup in distances if tup[0] < max_distance]
    distances.sort(key=lambda tup: tup[0])

    for _, i, j in distances:
        G.add_edge(i, j)
    print('Added {} edges'.format(len(distances)))

    connected_components = find_connected_components(G)
    print_connected_components_stat(connected_components, 'Gallery (after)')

    large_components = [c for c in connected_components if len(c) > 1]
    print('Num large components: {}'.format(len(large_components)))

    image_ids = set(i for comp in large_components for i in comp)
    print('Num clustered images: {} / {}'.format(len(image_ids), num_samples))

    if data_loader is None:
        return connected_components

    import cv2
    from os import makedirs
    from shutil import rmtree
    from os.path import exists, join
    out_dir = '/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_v142/gallery'
    if exists(out_dir):
        rmtree(out_dir)
    makedirs(out_dir)

    comp_map = defaultdict(list)
    for comp in init_connected_components:
        comp_map[len(comp)].append(set(comp))

    image_ids_map = dict()
    for comp_id, comp in enumerate(connected_components):
        if len(comp) <= 1:
            continue

        new_comp = True
        if len(comp) in comp_map:
            candidates = comp_map[len(comp)]
            for candidate in candidates:
                if set(comp) == candidate:
                    new_comp = False
                    break

        if not new_comp:
            continue

        for i in comp:
            image_ids_map[i] = comp_id

        comp_dir = join(out_dir, str(comp_id))
        makedirs(comp_dir)

    image_ids = set(image_ids_map.keys())

    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1)
    with torch.no_grad():
        last_data_id = 0
        for data in tqdm(data_loader):
            float_images = np.transpose(data[0].data[:, 0].cpu().numpy(), (0, 2, 3, 1))
            images = (255.0 * (float_images * std + mean)).astype(np.uint8)

            for image_id, image in enumerate(images):
                data_id = last_data_id + image_id
                if data_id not in image_ids:
                    continue

                comp_id = image_ids_map[data_id]
                comp_dir = join(out_dir, str(comp_id))

                image_path = join(comp_dir, 'img_{:06}.jpg'.format(data_id))
                cv2.imwrite(image_path, image[:, :, ::-1])

            last_data_id += images.shape[0]

    return connected_components


def find_connected_components(graph):
    return [list(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]


def print_connected_components_stat(components, header=''):
    print('{} connected components stat:'.format(header))
    print('   num components: {}'.format(len(components)))

    lengths = [len(c) for c in components]
    print('   sizes min: {}, max: {}'.format(np.min(lengths),
                                             np.max(lengths)))
    print('   sizes p@5: {}, p@50: {}, p@95: {}'.format(np.percentile(lengths, 5.0),
                                                        np.percentile(lengths, 50.0),
                                                        np.percentile(lengths, 95.0)))


def load_tracklets(file_path, gallery_size):
    tracklets = []
    for line in open(file_path):
        str_values = [s for s in line.replace('\n', '').split(' ') if len(s) > 0]
        ids = [int(s) - 1 for s in str_values]
        assert len(ids) > 0

        for sample_id in ids:
            assert 0 <= sample_id < gallery_size

        tracklets.append(ids)

    track_ids = [sample_id for track in tracklets for sample_id in track]
    assert len(track_ids) == len(set(track_ids))

    rest_ids = set(range(gallery_size)) - set(track_ids)
    print('Num gallery images without track info: {} / {}'.format(len(rest_ids), gallery_size))
    for rest_id in rest_ids:
        tracklets.append([rest_id])

    assert sum([len(track) for track in tracklets]) == gallery_size

    return tracklets


def gg_add_track_info(dist_matrix, tracks):
    g_track_dist = []
    for track_ids in tracks:
        distances = dist_matrix[:, track_ids]
        group_distance = np.percentile(distances, 10, axis=1)
        g_track_dist.append(group_distance.reshape([-1, 1]))
    g_track_dist = np.concatenate(tuple(g_track_dist), axis=1)

    track_track_dist = []
    for track_ids in tracks:
        distances = g_track_dist[track_ids, :]
        group_distance = np.percentile(distances, 20, axis=0)
        track_track_dist.append(group_distance.reshape([1, -1]))
    track_track_dist = np.concatenate(tuple(track_track_dist), axis=0)

    np.fill_diagonal(track_track_dist, 0.0)

    return track_track_dist


def qg_add_track_info(dist_matrix, tracks):
    track_distances = []
    for track_ids in tracks:
        distances = dist_matrix[:, track_ids]
        group_distance = np.percentile(distances, 10, axis=1)
        track_distances.append(group_distance.reshape([-1, 1]))
    qg_dist_matrix = np.concatenate(tuple(track_distances), axis=1)

    return qg_dist_matrix


def find_matches_image_to_track(dist_matrix, query_tracklets, gallery_tracklets, top_k=100):
    track_distances = []
    for gallery_tracklet_ids in gallery_tracklets:
        distances = dist_matrix[:, gallery_tracklet_ids]
        group_distance = np.percentile(distances, 10, axis=1)
        track_distances.append(group_distance.reshape([-1, 1]))
    track_distances = np.concatenate(tuple(track_distances), axis=1)
    track_indices = np.argsort(track_distances, axis=1)

    out_matches = []
    for q_id in range(dist_matrix.shape[0]):
        ids = []
        for track_id in track_indices[q_id]:
            ids.extend(gallery_tracklets[int(track_id)])

        out_matches.append(ids)

    return np.array(out_matches)[:, :top_k]


def find_matches_track_to_track(dist_matrix, query_tracklets, gallery_tracklets, top_k=100):
    track_distances = []
    for query_tracklet_ids in query_tracklets:
        row_distances = dist_matrix[query_tracklet_ids]

        for gallery_tracklet_ids in gallery_tracklets:
            set_distances = row_distances[:, gallery_tracklet_ids]
            dist = np.percentile(set_distances, 10)
            track_distances.append(dist)
    track_distances = np.array(track_distances, dtype=np.float32)
    track_distances = track_distances.reshape([len(query_tracklets), len(gallery_tracklets)])

    track_indices = np.argsort(track_distances, axis=1)

    out_matches = np.empty([dist_matrix.shape[0], top_k], dtype=np.int32)
    for q_tracklet_id in range(len(query_tracklets)):
        ids = []
        for track_id in track_indices[q_tracklet_id]:
            ids.extend(gallery_tracklets[int(track_id)])

        out_matches[query_tracklets[q_tracklet_id]] = ids[:top_k]

    return out_matches


def find_matches_image_to_track2(dist_matrix, tracks, top_k=100):
    track_indices = np.argsort(dist_matrix, axis=1)

    out_matches = []
    for q_id in range(dist_matrix.shape[0]):
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

    gallery_tracklets = load_tracklets(args.tracks_file, gallery_size)
    print('Loaded tracklets: {}'.format(len(gallery_tracklets)))

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_pids))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = model.cuda()

    embeddings_query = extract_features(model, data_query, cfg.use_gpu)
    print('Extracted query: {}'.format(embeddings_query.shape))

    embeddings_gallery = extract_features(model, data_gallery, cfg.use_gpu)
    print('Extracted gallery: {}'.format(embeddings_gallery.shape))

    print('Calculating distance matrices ...')
    distance_matrix_qg = calculate_distances(embeddings_query, embeddings_gallery)
    distance_matrix_qq = calculate_distances(embeddings_query, embeddings_query)
    distance_matrix_gg = calculate_distances(embeddings_gallery, embeddings_gallery)

    print('Merging query samples')
    query_tracklets = merge_query_samples(distance_matrix_qq, max_distance=0.15,
                                          data_loader=data_query)
    print('Merging gallery samples')
    gallery_tracklets = merge_gallery_tracklets(distance_matrix_gg, max_distance=0.05, src_tracklets=gallery_tracklets,
                                                data_loader=data_gallery)

    enable_track2track = False
    if enable_track2track:
        print('Calculating query to track distance matrix ...')
        distance_matrix_q_track_g = qg_add_track_info(distance_matrix_qg, gallery_tracklets)

        print('Calculating track to track distance matrix ...')
        distance_matrix_track_gg = gg_add_track_info(distance_matrix_gg, gallery_tracklets)

        print('Applying re-ranking ...')
        distance_matrix_qg = re_ranking(distance_matrix_q_track_g, distance_matrix_qq, distance_matrix_track_gg,
                                        k1=50, k2=15, lambda_value=0.1)
        print('Distance matrix after re-ranking: {}'.format(distance_matrix_qg.shape))

        matches = find_matches_image_to_track2(distance_matrix_qg, gallery_tracklets, top_k=100)
        print('Matches: {}'.format(matches.shape))
    else:
        print('Applying re-ranking ...')
        distance_matrix_qg = re_ranking(distance_matrix_qg, distance_matrix_qq, distance_matrix_gg,
                                        k1=50, k2=15, lambda_value=0.1)
        print('Distance matrix after re-ranking: {}'.format(distance_matrix_qg.shape))

        matches = find_matches_track_to_track(distance_matrix_qg, query_tracklets, gallery_tracklets, top_k=100)
        print('Matches: {}'.format(matches.shape))

    dump_matches(matches, args.out_file)
    print('Submission file has been stored at: {}'.format(args.out_file))


if __name__ == '__main__':
    main()

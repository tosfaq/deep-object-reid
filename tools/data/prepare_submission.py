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

    distances_queue = [tup for tup in distances if tup[0] < max_distance]
    distances_queue.sort(key=lambda tup: tup[0])

    G = nx.Graph()
    G.add_nodes_from(list(range(num_samples)))

    num_comp_last = len(find_connected_components(G))
    num_added_bridges = 0
    for d, i, j in distances_queue:
        G.add_edge(i, j)

        num_comp_new = len(find_connected_components(G))
        if num_comp_new == num_comp_last:
            G.remove_edge(i, j)
        else:
            num_comp_last = num_comp_new
            num_added_bridges += 1
    print('Added {} bridge edges'.format(num_added_bridges))

    # init_connected_components = find_connected_components(G)
    # print_connected_components_stat(init_connected_components, 'Query (init)')
    #
    # comp_map = defaultdict(list)
    # for comp in init_connected_components:
    #     comp_map[len(comp)].append(set(comp))
    #
    # distances_queue = [tup for tup in distances if max_distance <= tup[0] < 0.19]
    # distances_queue.sort(key=lambda tup: tup[0])
    #
    # comp_last = find_connected_components(G)
    # num_added_bridges = 0
    # for d, i, j in distances_queue:
    #     i_size = [len(c) for c in comp_last if i in set(c)][0]
    #     j_size = [len(c) for c in comp_last if j in set(c)][0]
    #
    #     if i_size + j_size >= 6:
    #         continue
    #
    #     G.add_edge(i, j)
    #
    #     comp_new = find_connected_components(G)
    #     if len(comp_new) == len(comp_last):
    #         G.remove_edge(i, j)
    #     else:
    #         comp_last = comp_new
    #         num_added_bridges += 1
    # print('Added {} bridge edges'.format(num_added_bridges))

    connected_components = find_connected_components(G)
    print_connected_components_stat(connected_components, 'Query (last)')

    large_components = [c for c in connected_components if len(c) > 1]
    print('Num large components: {}'.format(len(large_components)))

    image_ids = set(i for comp in large_components for i in comp)
    print('Num clustered images: {} / {}'.format(len(image_ids), num_samples))

    # if data_loader is None:
    #     return connected_components
    #
    # import cv2
    # from os import makedirs
    # from shutil import rmtree
    # from os.path import exists, join
    # out_dir = '/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_all_m/query'
    # if exists(out_dir):
    #     rmtree(out_dir)
    # makedirs(out_dir)
    #
    # image_ids_map = dict()
    # for comp_id, comp in enumerate(connected_components):
    #     if len(comp) <= 1:
    #         continue
    #
    #     # new_comp = True
    #     # if len(comp) in comp_map:
    #     #     candidates = comp_map[len(comp)]
    #     #     for candidate in candidates:
    #     #         if set(comp) == candidate:
    #     #             new_comp = False
    #     #             break
    #     #
    #     # if not new_comp:
    #     #     continue
    #
    #     for i in comp:
    #         image_ids_map[i] = comp_id
    #
    #     comp_dir = join(out_dir, str(comp_id))
    #     makedirs(comp_dir)
    #
    # image_ids = set(image_ids_map.keys())
    #
    # std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1)
    # mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1)
    # with torch.no_grad():
    #     last_data_id = 0
    #     for data in tqdm(data_loader):
    #         float_images = np.transpose(data[0].data[:, 0].cpu().numpy(), (0, 2, 3, 1))
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

    distances = [(distance_matrix[i, j], i, j)
                 for i in range(num_samples)
                 for j in range(i + 1, num_samples)
                 if free_pairs[i, j]]

    distances_queue = [tup for tup in distances if tup[0] < max_distance]
    distances_queue.sort(key=lambda tup: tup[0])

    init_connected_components = find_connected_components(G)
    print_connected_components_stat(init_connected_components, 'Gallery (before)')

    num_comp_last = len(find_connected_components(G))
    num_added_bridges = 0
    for d, i, j in distances_queue:
        G.add_edge(i, j)

        num_comp_new = len(find_connected_components(G))
        if num_comp_new == num_comp_last:
            G.remove_edge(i, j)
        else:
            num_comp_last = num_comp_new
            num_added_bridges += 1
    print('Added {} bridge edges'.format(num_added_bridges))

    connected_components = find_connected_components(G)
    print_connected_components_stat(connected_components, 'Gallery (after)')

    large_components = [c for c in connected_components if len(c) > 1]
    print('Num large components: {}'.format(len(large_components)))

    image_ids = set(i for comp in large_components for i in comp)
    print('Num clustered images: {} / {}'.format(len(image_ids), num_samples))

    if data_loader is None:
        return connected_components

    # import cv2
    # from os import makedirs
    # from shutil import rmtree
    # from os.path import exists, join
    # out_dir = '/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_v142-156-157/gallery'
    # if exists(out_dir):
    #     rmtree(out_dir)
    # makedirs(out_dir)
    #
    # comp_map = defaultdict(list)
    # for comp in init_connected_components:
    #     comp_map[len(comp)].append(set(comp))
    #
    # image_ids_map = dict()
    # for comp_id, comp in enumerate(connected_components):
    #     if len(comp) <= 1:
    #         continue
    #
    #     new_comp = True
    #     if len(comp) in comp_map:
    #         candidates = comp_map[len(comp)]
    #         for candidate in candidates:
    #             if set(comp) == candidate:
    #                 new_comp = False
    #                 break
    #
    #     if not new_comp:
    #         continue
    #
    #     for i in comp:
    #         image_ids_map[i] = comp_id
    #
    #     comp_dir = join(out_dir, str(comp_id))
    #     makedirs(comp_dir)
    #
    # image_ids = set(image_ids_map.keys())
    #
    # std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1)
    # mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1)
    # with torch.no_grad():
    #     last_data_id = 0
    #     for data in tqdm(data_loader):
    #         float_images = np.transpose(data[0].data[:, 0].cpu().numpy(), (0, 2, 3, 1))
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


def merge_dist_matrices(dist_matrices):
    assert len(dist_matrices) > 0

    if len(dist_matrices) == 1:
        return dist_matrices[0]

    dist_matrices = np.stack(tuple(dist_matrices), axis=0)
    dist_matrix = np.median(dist_matrices, axis=0)

    return dist_matrix


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


def add_track_info(dist_matrix, tracklets, alpha=0.3):
    if alpha <= 0.0:
        return dist_matrix

    num_tracklets = len(tracklets)
    assert num_tracklets > 0

    tracklet_dist_matrix = np.zeros_like(dist_matrix)
    for i in range(num_tracklets):
        anchor_ids = tracklets[i]
        anchor_distances = dist_matrix[anchor_ids]

        for j in range(i + 1, num_tracklets):
            ref_ids = tracklets[j]
            ref_distances = anchor_distances[:, ref_ids]

            dist = np.percentile(ref_distances, 10)

            row_indices = [ii for ii in anchor_ids for _ in ref_ids]
            col_indices = [jj for _ in anchor_ids for jj in ref_ids]
            np.add.at(tracklet_dist_matrix, (row_indices, col_indices), dist)

    update_dist_matrix = (1.0 - alpha) * dist_matrix + alpha * tracklet_dist_matrix

    return update_dist_matrix


def find_matches(dist_matrix, query_tracklets, gallery_tracklets, top_k=100, threshold=10):
    track_to_track_dist = []
    # track_to_sample_dist = []
    for query_tracklet_ids in query_tracklets:
        row_distances = dist_matrix[query_tracklet_ids]
        # track_to_sample_dist.append(np.percentile(row_distances, threshold, axis=0).reshape([1, -1]))

        for gallery_tracklet_ids in gallery_tracklets:
            set_distances = row_distances[:, gallery_tracklet_ids]
            dist = np.percentile(set_distances, threshold)
            track_to_track_dist.append(dist)
    track_to_track_dist = np.array(track_to_track_dist, dtype=np.float32)
    track_to_track_dist = track_to_track_dist.reshape([len(query_tracklets), len(gallery_tracklets)])
    # track_to_sample_dist = np.concatenate(tuple(track_to_sample_dist), axis=0)

    track_to_track_indices = np.argsort(track_to_track_dist, axis=1)

    out_matches = np.empty([dist_matrix.shape[0], top_k], dtype=np.int32)
    for q_tracklet_id in range(len(query_tracklets)):
        # q_dist = track_to_sample_dist[q_tracklet_id]

        ids = []
        for track_id in track_to_track_indices[q_tracklet_id]:
            # local_ids = np.array(gallery_tracklets[int(track_id)], dtype=np.int32)
            # local_dist_values = q_dist[local_ids]
            #
            # ordered_local_ids = local_ids[np.argsort(local_dist_values)]
            # ids.extend(ordered_local_ids.tolist())

            ids.extend(gallery_tracklets[int(track_id)])

            if len(ids) >= top_k:
                break

        out_matches[query_tracklets[q_tracklet_id]] = ids[:top_k]

    return out_matches


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
    parser.add_argument('--weights', '-w', type=str, nargs='+', required=True)
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
    mock_model = torchreid.models.build_model(**model_kwargs(cfg, num_pids))

    print('Processing models...')
    qq, qg, gg = [], [], []
    for weights_path in args.weights:
        cfg.model.load_weights = weights_path
        assert check_isfile(cfg.model.load_weights)

        print('Loading model: {}'.format(cfg.model.load_weights))
        load_pretrained_weights(mock_model, cfg.model.load_weights)
        model = mock_model.cuda() if cfg.use_gpu else mock_model

        print('Extracting query embeddings ...')
        embeddings_query = extract_features(model, data_query, cfg.use_gpu)
        print('Extracted query (NxE): {}x{}'.format(*embeddings_query.shape))

        print('Extracting gallery embeddings ...')
        embeddings_gallery = extract_features(model, data_gallery, cfg.use_gpu)
        print('Extracted gallery (NxE): {}x{}'.format(*embeddings_gallery.shape))

        print('Calculating distance matrices ...')
        qq.append(calculate_distances(embeddings_query, embeddings_query))
        qg.append(calculate_distances(embeddings_query, embeddings_gallery))
        gg.append(calculate_distances(embeddings_gallery, embeddings_gallery))

    print('Merging distance matrices ...')
    distance_matrix_qq = merge_dist_matrices(qq)
    distance_matrix_qg = merge_dist_matrices(qg)
    distance_matrix_gg = merge_dist_matrices(gg)

    np.save('/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_all2_m/qq', distance_matrix_qq)
    np.save('/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_all2_m/qg', distance_matrix_qg)
    np.save('/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_all2_m/gg', distance_matrix_gg)

    # distance_matrix_qq = np.load('/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_all2_m/qq.npy')
    # distance_matrix_qg = np.load('/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_all2_m/qg.npy')
    # distance_matrix_gg = np.load('/home/eizutov/data/ReID/Vehicle/aic20/samples_clustered_all2_m/gg.npy')

    print('Merging query samples ...')
    query_tracklets = merge_query_samples(distance_matrix_qq,
                                          max_distance=0.15,
                                          data_loader=data_query)
    print('Merging gallery samples ...')
    gallery_tracklets = merge_gallery_tracklets(distance_matrix_gg,
                                                max_distance=0.05,
                                                src_tracklets=gallery_tracklets,
                                                data_loader=data_gallery)

    print('Updating query distance matrix ...')
    distance_matrix_qq = add_track_info(distance_matrix_qq, query_tracklets, alpha=0.0)

    print('Updating gallery distance matrix ...')
    distance_matrix_gg = add_track_info(distance_matrix_gg, gallery_tracklets, alpha=0.0)

    print('Applying re-ranking ...')
    distance_matrix_qg = re_ranking(distance_matrix_qg, distance_matrix_qq, distance_matrix_gg,
                                    k1=50, k2=15, lambda_value=0.1)

    print('Finding matches ...')
    matches = find_matches(distance_matrix_qg, query_tracklets, gallery_tracklets, top_k=100)

    dump_matches(matches, args.out_file)
    print('Submission file has been stored at: {}'.format(args.out_file))


if __name__ == '__main__':
    main()

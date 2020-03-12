import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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

    return data_loader


def extract_features(model, data_loader, use_gpu):
    model.eval()

    out_embeddings = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            images = data[0]
            if use_gpu:
                images = images.cuda()

            embeddings = model(images)
            norm_embeddings = F.normalize(embeddings, dim=-1)

            out_embeddings.append(norm_embeddings.data.cpu())

    out_embeddings = torch.cat(out_embeddings, 0).numpy()

    return out_embeddings


def calculate_distances(a, b):
    return 1.0 - np.matmul(a, np.transpose(b))


def find_matches(distance_matrix, top_k):
    return np.argsort(distance_matrix, axis=-1)[:, :top_k]


def dump_matches(matches, out_file):
    shifted_matches = matches + 1

    with open(out_file, 'w') as out_stream:
        for row in shifted_matches:
            line = ' '.join(map(str, row.reshape([-1]).tolist()))
            out_stream.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--root', '-r', type=str, required=True)
    parser.add_argument('--out-file', '-o', type=str, required=True)
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

    data_query, num_pids = build_query(cfg)
    data_gallery = build_gallery(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_pids))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    embeddings_query = extract_features(model, data_query, cfg.use_gpu)
    print('Extracted query: {}'.format(embeddings_query.shape))

    embeddings_gallery = extract_features(model, data_gallery, cfg.use_gpu)
    print('Extracted gallery: {}'.format(embeddings_gallery.shape))

    distance_matrix_qg = calculate_distances(embeddings_query, embeddings_gallery)
    print('Distance matrix: {}'.format(distance_matrix_qg.shape))

    # print('Applying re-ranking ...')
    # distance_matrix_qq = calculate_distances(embeddings_query, embeddings_query)
    # distance_matrix_gg = calculate_distances(embeddings_gallery, embeddings_gallery)
    # distance_matrix_qg = re_ranking(distance_matrix_qg, distance_matrix_qq, distance_matrix_gg)
    # print('Distance matrix after re-ranking: {}'.format(distance_matrix_qg.shape))

    matches = find_matches(distance_matrix_qg, top_k=100)
    print('Matches: {}'.format(matches.shape))

    dump_matches(matches, args.out_file)
    print('Submission file has been stored at: {}'.format(args.out_file))


if __name__ == '__main__':
    main()

from os.path import exists
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, REMAINDER

import torch
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
    out_cam_ids = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            images = data[0]

            cam_id = data[2]
            out_cam_ids.append(cam_id)

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
    out_cam_ids = torch.cat(out_cam_ids, 0).numpy()

    return out_embeddings, out_cam_ids


def calculate_distances(a, b):
    return 1.0 - np.matmul(a, np.transpose(b))


def show_stat(dist_matrix, cam_ids, name):
    pass


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', '-c', type=str, required=True)
    parser.add_argument('--root', '-r', type=str, required=True)
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

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(**model_kwargs(cfg, num_pids))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = model.cuda()

    embeddings_query, cam_ids_query = extract_features(model, data_query, cfg.use_gpu, enable_flipping=True)
    print('Extracted query: {}'.format(embeddings_query.shape))

    embeddings_gallery, cam_ids_gallery = extract_features(model, data_gallery, cfg.use_gpu, enable_flipping=True)
    print('Extracted gallery: {}'.format(embeddings_gallery.shape))

    print('Calculating distance matrices ...')
    distance_matrix_qq = calculate_distances(embeddings_query, embeddings_query)
    distance_matrix_gg = calculate_distances(embeddings_gallery, embeddings_gallery)

    show_stat(distance_matrix_qq, cam_ids_query, 'Query')
    show_stat(distance_matrix_gg, cam_ids_gallery, 'Gallery')


if __name__ == '__main__':
    main()

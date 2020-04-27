import argparse
from os.path import exists, join, abspath, isfile
from os import listdir, walk, remove

import cv2
import numpy as np
from tqdm import tqdm


def parse_relative_paths(data_dir):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    relative_paths = []
    for root, sub_dirs, files in walk(data_dir):
        if len(sub_dirs) == 0 and len(files) > 0:
            relative_paths.append(root[skip_size:])

    return relative_paths


def prepare_tasks(relative_paths, in_masks):
    out_data = []
    for relative_path in relative_paths:
        in_masks_dir = join(in_masks, relative_path)
        assert exists(in_masks_dir)

        mask_names = [f.split('.')[0] for f in listdir(in_masks_dir) if isfile(join(in_masks_dir, f))]
        if len(mask_names) == 0:
            continue

        for name in mask_names:
            in_mask_path = join(in_masks_dir, '{}.png'.format(name))
            out_data.append(in_mask_path)

    return out_data


def process_tasks(tasks, min_non_zero_fraction):
    for in_mask_path in tqdm(tasks):
        mask = cv2.imread(in_mask_path, cv2.IMREAD_GRAYSCALE)
        image_height, image_width = mask.shape

        mask_area = np.count_nonzero(mask)
        non_zero_fraction = float(mask_area) / float(image_height * image_width)
        if non_zero_fraction < min_non_zero_fraction:
            print('[WARNING] Non-zero fraction is {} for image: {}'.format(non_zero_fraction, in_mask_path))

            remove(in_mask_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_masks', '-im', type=str, required=True)
    parser.add_argument('--non_zero_fraction', '-nz', type=float, required=False, default=0.25)
    args = parser.parse_args()

    assert exists(args.in_masks)

    print('Preparing tasks ...')
    relative_paths = parse_relative_paths(args.in_masks)
    tasks = prepare_tasks(relative_paths, args.in_masks)

    print('Processing tasks ...')
    process_tasks(tasks, args.non_zero_fraction)


if __name__ == '__main__':
    main()

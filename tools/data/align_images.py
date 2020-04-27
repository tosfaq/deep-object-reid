import argparse
from os.path import exists, join, abspath, isfile
from os import listdir, walk, makedirs
from shutil import rmtree

import cv2
import numpy as np
from tqdm import tqdm


def create_dirs(dir_path):
    if exists(dir_path):
        rmtree(dir_path)

    makedirs(dir_path)


def parse_relative_paths(data_dir):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    relative_paths = []
    for root, sub_dirs, files in walk(data_dir):
        if len(sub_dirs) == 0 and len(files) > 0:
            relative_paths.append(root[skip_size:])

    return relative_paths


def prepare_tasks(relative_paths, in_images, in_masks, out_images, out_masks):
    out_data = []
    for relative_path in relative_paths:
        in_images_dir = join(in_images, relative_path)
        assert exists(in_images_dir)

        in_masks_dir = join(in_masks, relative_path)
        assert exists(in_masks_dir)

        image_names = [f.split('.')[0] for f in listdir(in_images_dir) if isfile(join(in_images_dir, f))]
        mask_names = [f.split('.')[0] for f in listdir(in_masks_dir) if isfile(join(in_masks_dir, f))]
        valid_names = list(set(image_names) & set(mask_names))

        if len(valid_names) == 0:
            continue

        out_images_dir = join(out_images, relative_path)
        if not exists(out_images_dir):
            makedirs(out_images_dir)

        out_masks_dir = join(out_masks, relative_path)
        if not exists(out_masks_dir):
            makedirs(out_masks_dir)

        for name in valid_names:
            in_image_path = join(in_images_dir, '{}.jpg'.format(name))
            in_mask_path = join(in_masks_dir, '{}.png'.format(name))
            out_image_path = join(out_images_dir, '{}.jpg'.format(name))
            out_mask_path = join(out_masks_dir, '{}.png'.format(name))

            out_data.append((in_image_path, in_mask_path, out_image_path, out_mask_path))

    return out_data


def process_tasks(tasks, min_non_zero_fraction, scale):
    for in_image_path, in_mask_path, out_image_path, out_mask_path in tqdm(tasks):
        mask = cv2.imread(in_mask_path, cv2.IMREAD_GRAYSCALE)
        image_height, image_width = mask.shape

        mask_area = np.count_nonzero(mask)
        non_zero_fraction = float(mask_area) / float(image_height * image_width)
        if non_zero_fraction < min_non_zero_fraction:
            print('[WARNING] Non-zero fraction is {} for image: {}'.format(non_zero_fraction, in_image_path))
            continue

        image = cv2.imread(in_image_path, cv2.IMREAD_COLOR)

        pts = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(pts)

        crop_center_x = float(x) + 0.5 * w
        crop_center_y = float(y) + 0.5 * h

        new_crop_width_half = 0.5 * scale * float(w)
        new_crop_height_half = 0.5 * scale * float(h)

        xmin = max(0, int(crop_center_x - new_crop_width_half))
        ymin = max(0, int(crop_center_y - new_crop_height_half))
        xmax = min(image_width, int(crop_center_x + new_crop_width_half))
        ymax = min(image_height, int(crop_center_y + new_crop_height_half))

        image = image[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]

        cv2.imwrite(out_image_path, image)
        cv2.imwrite(out_mask_path, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_images', '-ii', type=str, required=True)
    parser.add_argument('--in_masks', '-im', type=str, required=True)
    parser.add_argument('--out_images', '-oi', type=str, required=True)
    parser.add_argument('--out_masks', '-om', type=str, required=True)
    parser.add_argument('--non_zero_fraction', '-nz', type=float, required=False, default=0.3)
    parser.add_argument('--crop_scale', '-cs', type=float, required=False, default=1.08)
    args = parser.parse_args()

    assert exists(args.in_images)
    assert exists(args.in_masks)

    create_dirs(args.out_images)
    create_dirs(args.out_masks)

    print('Preparing tasks ...')
    relative_paths = parse_relative_paths(args.in_images)
    tasks = prepare_tasks(relative_paths, args.in_images, args.in_masks, args.out_images, args.out_masks)

    print('Processing tasks ...')
    process_tasks(tasks, args.non_zero_fraction, args.crop_scale)


if __name__ == '__main__':
    main()

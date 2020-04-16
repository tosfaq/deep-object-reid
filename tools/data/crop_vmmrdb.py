import argparse
from os.path import exists, join, abspath, isfile
from os import listdir, walk, makedirs
from shutil import rmtree

import cv2
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


def prepare_tasks(relative_paths, images_dir, output_dir):
    out_data = []
    for relative_path in relative_paths:
        images = join(images_dir, relative_path)
        assert exists(images)

        output_path = join(output_dir, relative_path)
        makedirs(output_path)

        image_files = [f for f in listdir(images) if isfile(join(images, f))]
        for image_file in image_files:
            full_image_path = join(images, image_file)
            full_output_path = join(output_path, image_file)
            out_data.append((full_image_path, full_output_path))

    return out_data


def resize_image(image, max_size):
    src_height, src_width = image.shape[:2]
    if src_height < 10 or src_width < 10:
        return None

    ar = float(src_height) / float(src_width)
    if ar > 1.0:
        trg_height = max_size
        trg_width = int(trg_height / ar)
    else:
        trg_width = max_size
        trg_height = int(trg_width * ar)

    return cv2.resize(image, (trg_width, trg_height))


def process_tasks(tasks, max_image_size):
    for image_path, output_path in tqdm(tasks):
        src_image = cv2.imread(image_path)

        trg_image = resize_image(src_image, max_image_size)
        if trg_image is not None:
            cv2.imwrite(output_path, trg_image)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', '-i', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    parser.add_argument('--max-image-size', '-ms', type=int, required=False, default=320)
    args = parser.parse_args()

    assert exists(args.images_dir)

    create_dirs(args.output_dir)

    relative_paths = parse_relative_paths(args.images_dir)
    print('Found classes: {}'.format(len(relative_paths)))
    tasks = prepare_tasks(relative_paths, args.images_dir, args.output_dir)
    print('Prepared tasks: {}'.format(len(tasks)))

    print('Processing tasks ...')
    process_tasks(tasks, args.max_image_size)


if __name__ == '__main__':
    main()

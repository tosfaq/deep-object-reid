from os import makedirs
from os.path import join, exists
from shutil import rmtree, copyfile
from argparse import ArgumentParser


def load_image_groups(file_path, images_dir):
    out_data = []
    with open(file_path) as input_stream:
        for line in input_stream:
            parts = line.replace('\n', '').split(' ')
            ids = map(int, parts)
            image_names = ['{:06}.jpg'.format(image_id) for image_id in ids]
            image_path = [join(images_dir, image_name) for image_name in image_names]

            out_data.append(image_path)

    return out_data


def dump_files(image_groups, out_dir):
    if exists(out_dir):
        rmtree(out_dir)
    makedirs(out_dir)

    for group_id, image_group in enumerate(image_groups):
        group_dir = join(out_dir, str(group_id))
        makedirs(group_dir)

        for sample_rank, src_image_path in enumerate(image_group):
            trg_image_path = join(group_dir, '{:06}.jpg'.format(sample_rank))

            copyfile(src_image_path, trg_image_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--groups-file', '-g', type=str, required=True)
    parser.add_argument('--images-dir', '-i', type=str, required=True)
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.groups_file)
    assert exists(args.images_dir)

    image_groups = load_image_groups(args.groups_file, args.images_dir)
    dump_files(image_groups, args.out_dir)


if __name__ == '__main__':
    main()

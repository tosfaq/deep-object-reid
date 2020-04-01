from os import makedirs
from os.path import join, exists
from shutil import rmtree, copyfile
from argparse import ArgumentParser


def load_image_groups(file_path, query_images_dir, gallery_images_dir):
    out_data = []
    with open(file_path) as input_stream:
        for line_id, line in enumerate(input_stream):
            parts = line.replace('\n', '').split(' ')
            ids = map(int, parts)
            image_names = ['{:06}.jpg'.format(image_id) for image_id in ids]
            image_paths = [join(gallery_images_dir, image_name) for image_name in image_names]

            query_image_name = '{:06}.jpg'.format(line_id + 1)
            query_image_path = join(query_images_dir, query_image_name)

            out_data.append([query_image_path] + image_paths)

    return out_data


def dump_files(image_groups, out_dir):
    if exists(out_dir):
        rmtree(out_dir)
    makedirs(out_dir)

    for group_id, image_group in enumerate(image_groups):
        group_dir = join(out_dir, str(group_id))
        makedirs(group_dir)

        for sample_rank, src_image_path in enumerate(image_group):
            trg_image_path = join(group_dir, '{:03}.jpg'.format(sample_rank))

            copyfile(src_image_path, trg_image_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--submission-file', '-s', type=str, required=True)
    parser.add_argument('--gallery-images-dir', '-g', type=str, required=True)
    parser.add_argument('--query-images-dir', '-q', type=str, required=True)
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.submission_file)
    assert exists(args.gallery_images_dir)
    assert exists(args.query_images_dir)

    image_groups = load_image_groups(args.submission_file, args.query_images_dir, args.gallery_images_dir)
    dump_files(image_groups, args.out_dir)


if __name__ == '__main__':
    main()

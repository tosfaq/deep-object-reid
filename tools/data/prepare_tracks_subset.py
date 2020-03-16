from os import makedirs
from os.path import exists, join
from shutil import copyfile, rmtree
import argparse

from lxml import etree


def load_annotation(annot_path):
    tree = etree.parse(annot_path)
    root = tree.getroot()

    assert len(root) == 1
    items = root[0]

    data = []
    for item in items:
        image_name = item.attrib['imageName']
        data.append(image_name)

    return data


def load_tracks(annot_path):
    tracks = []
    for line in open(annot_path):
        file_names = [n for n in line.replace('\n', '').split(' ') if len(n) > 0]
        if len(file_names) == 0:
            continue

        tracks.append(file_names)

    return tracks


def filter_tracks(tracks, valid_names):
    out_tracks = []
    for track in tracks:
        valid_mask = [n in valid_names for n in track]
        num_valid = sum(valid_mask)
        assert num_valid == 0 or len(valid_mask)

        if num_valid == len(valid_mask):
            out_tracks.append(track)

    return out_tracks


def generate_map(file_names):
    data = dict()
    for file_name in file_names:
        image_id = int(file_name.split('.')[0])
        data[image_id] = file_name

    ordered_ids = sorted(data.keys())
    out_data = {data[img_id]: i + 1 for i, img_id in enumerate(ordered_ids)}

    return out_data


def map_tracks(tracks, names_map):
    out_tracks = []
    for track in tracks:
        out_tracks.append([names_map[n] for n in track])

    return out_tracks


def dump_tracks(tracks, out_file):
    with open(out_file, 'w') as output_stream:
        for track in tracks:
            line = ' '.join(map(str, track)) + '\n'
            output_stream.write(line)


def dump_track_images(tracks, src_dir, trg_dir):
    if exists(trg_dir):
        rmtree(trg_dir)
        makedirs(trg_dir)

    for track_id, track in enumerate(tracks):
        trg_track_dir = join(trg_dir, 'track_{:04}'.format(track_id))
        makedirs(trg_track_dir)

        for name in track:
            src_image_path = join(src_dir, name)
            trg_image_path = join(trg_track_dir, name)

            copyfile(src_image_path, trg_image_path)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotation-file', '-a', type=str, required=True)
    parser.add_argument('--tracks-file', '-t', type=str, required=True)
    parser.add_argument('--out-file', '-of', type=str, required=True)
    parser.add_argument('--images-dir', '-im', type=str, required=True)
    parser.add_argument('--out-dir', '-od', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.annotation_file)
    assert exists(args.tracks_file)

    data = load_annotation(args.annotation_file)
    print('Loaded images: {}'.format(len(data)))

    tracks = load_tracks(args.tracks_file)
    print('Loaded tracks: {}'.format(len(tracks)))

    filtered_tracks = filter_tracks(tracks, set(data))
    print('Filtered tracks: {}'.format(len(filtered_tracks)))

    names_map = generate_map(data)
    out_tracks = map_tracks(filtered_tracks, names_map)

    dump_tracks(out_tracks, args.out_file)
    print('Stored at: {}'.format(args.out_file))

    dump_track_images(filtered_tracks, args.images_dir, args.out_dir)


if __name__ == '__main__':
    main()

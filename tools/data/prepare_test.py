import os.path as osp
import argparse

from lxml import etree


def load_tracks(annot_path):
    tracks = []
    for line in open(annot_path):
        file_names = line.replace('\n', '').split(' ')
        file_names = [n for n in file_names if len(n) > 0]
        assert len(file_names) > 1

        tracks.append(file_names)

    return tracks


def to_list(tracks):
    return [record for track in tracks for record in track]


def store_annotation(records, out_path):
    root = etree.Element('TrainingImages')
    items = etree.SubElement(root, 'Items', number=str(len(records)))
    for image_name in records:
        pid_str = '0001'
        cam_id_str = 'c001'
        etree.SubElement(items, 'Item', imageName=image_name, vehicleID=pid_str, cameraID=cam_id_str)

    with open(out_path, 'wb') as output:
        output.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tracks-annot', '-t', type=str, required=True)
    parser.add_argument('--out-annot', '-o', type=str, required=True)
    args = parser.parse_args()

    assert osp.exists(args.tracks_annot)

    tracks = load_tracks(args.tracks_annot)
    records = to_list(tracks)

    store_annotation(records, args.out_annot)


if __name__ == '__main__':
    main()

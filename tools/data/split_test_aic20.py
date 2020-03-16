import os
import os.path as osp
import argparse
import random
from collections import defaultdict

from lxml import etree


def load_annotation(data, annot_path):
    tree = etree.parse(annot_path)
    root = tree.getroot()

    assert len(root) == 1
    items = root[0]

    for item in items:
        image_name = item.attrib['imageName']
        pid = int(item.attrib['vehicleID'])
        cam_id = int(item.attrib['cameraID'][1:])

        data[pid].append((image_name, pid, cam_id))

    return data


def store_annotation(records, out_path):
    root = etree.Element('TrainingImages')
    items = etree.SubElement(root, 'Items', number=str(len(records)))
    for image_name, pid, cam_id in records:
        pid_str = '{:04}'.format(pid)
        cam_id_str = 'c{:03}'.format(cam_id)
        etree.SubElement(items, 'Item', imageName=image_name, vehicleID=pid_str, cameraID=cam_id_str)

    with open(out_path, 'wb') as output:
        output.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='utf-8'))


def split_data(data, num_ids):
    all_ids = list(data.keys())
    subset_ids = random.sample(all_ids, num_ids)

    out_data = {pid: data[pid] for pid in subset_ids}

    return out_data


def get_records(data):
    return [record for pid, records in data.items() for record in records]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-annot', '-i', nargs='+', type=str, required=True)
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    parser.add_argument('--num-ids', '-n', type=int, nargs='+', required=True)
    args = parser.parse_args()

    assert osp.exists(args.input_annot)

    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    data = defaultdict(list)
    for annot_source in args.input_annot:
        data = load_annotation(data, annot_source)

    for num_ids in args.num_ids:
        assert num_ids <= len(data)

        subset_data = split_data(data, num_ids)
        records = get_records(subset_data)

        out_path = osp.join(args.out_dir, '{:03}IDs_{}'.format(num_ids, osp.basename(args.input_annot)))
        store_annotation(records, out_path)


if __name__ == '__main__':
    main()

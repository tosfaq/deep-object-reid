import os.path as osp
import random as rnd
from os import makedirs
from lxml import etree
from argparse import ArgumentParser
from shutil import copyfile, rmtree

from tqdm import tqdm


def load_annotation(annot_path):
    tree = etree.parse(annot_path)
    root = tree.getroot()

    assert len(root) == 1
    items = root[0]

    data = []
    for item in items:
        image_name = item.attrib['imageName']
        pid = int(item.attrib['vehicleID'])
        cam_id = int(item.attrib['cameraID'][1:])

        data.append((image_name, pid, cam_id))

    return data


def split_identities(data):
    out_data = dict()
    for image_path, pid, cam_id in data:
        if pid not in out_data:
            out_data[pid] = []

        out_data[pid].append((image_path, cam_id))

    return out_data


def select_data(data, identities):
    out_data = dict()
    for pid, record in data.items():
        if pid in identities:
            out_data[pid] = record

    return out_data


def to_list(data):
    out_data = list()
    for pid, records in data.items():
        for image_path, cam_id in records:
            out_data.append((image_path, pid, cam_id))

    return out_data


def split_test(data, num_query_samples):
    query_data = []
    test_data = []
    for pid, records in data.items():
        assert len(records) > num_query_samples

        rnd.shuffle(records)

        for image_path, cam_id in records[:num_query_samples]:
            query_data.append((image_path, pid, cam_id))
        for image_path, cam_id in records[num_query_samples:]:
            test_data.append((image_path, pid, cam_id))

    return query_data, test_data


def copy_data(records, src_dir, trg_dir):
    if osp.exists(trg_dir):
        rmtree(trg_dir)
    makedirs(trg_dir)

    for image_name, _, _ in tqdm(records, desc='Copying images'):
        src_path = osp.join(src_dir, image_name)
        trg_path = osp.join(trg_dir, image_name)

        copyfile(src_path, trg_path)


def store_annotation(records, out_path, shuffle=False):
    if shuffle:
        rnd.shuffle(records)

    root = etree.Element('TrainingImages')
    items = etree.SubElement(root, 'Items', number=str(len(records)))
    for image_name, pid, cam_id in records:
        pid_str = '{:04}'.format(pid)
        cam_id_str = 'c{:03}'.format(cam_id)
        etree.SubElement(items, 'Item', imageName=image_name, vehicleID=pid_str, cameraID=cam_id_str)

    with open(out_path, 'wb') as output:
        output.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='utf-8'))


def main():
    parser = ArgumentParser()
    parser.add_argument('--annot_path', '-a', type=str, required=True)
    parser.add_argument('--src_path', '-s', type=str, required=True)
    parser.add_argument('--trg_path', '-t', type=str, required=True)
    parser.add_argument('--num_train_ids', '-n', type=int, required=False, default=233)
    parser.add_argument('--num_query', '-q', type=int, required=False, default=3)
    args = parser.parse_args()

    assert osp.exists(args.annot_path)
    assert osp.exists(args.src_path)

    data = load_annotation(args.annot_path)
    print('Loaded {} items.'.format(len(data)))

    splitted_data = split_identities(data)
    print('Num unique identities: {}'.format(len(splitted_data)))

    assert len(splitted_data) > args.num_train_ids

    identities = list(splitted_data.keys())
    train_identities = rnd.sample(identities, args.num_train_ids)
    test_identities = list(set(identities) - set(train_identities))
    print('Split on {} train and {} test identities.'
          .format(len(train_identities), len(test_identities)))

    train_data = select_data(splitted_data, train_identities)
    test_data = select_data(splitted_data, test_identities)

    train_records = to_list(train_data)
    copy_data(train_records, args.src_path, osp.join(args.trg_path, 'image_train'))
    store_annotation(train_records, osp.join(args.trg_path, 'train_label.xml'), shuffle=True)

    query_records, test_records = split_test(test_data, args.num_query)

    copy_data(query_records, args.src_path, osp.join(args.trg_path, 'image_query'))
    store_annotation(query_records, osp.join(args.trg_path, 'query_label.xml'))

    copy_data(test_records, args.src_path, osp.join(args.trg_path, 'image_test'))
    store_annotation(test_records, osp.join(args.trg_path, 'test_label.xml'))


if __name__ == '__main__':
    main()

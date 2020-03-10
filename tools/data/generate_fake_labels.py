import os
import os.path as osp
import random as rnd
from lxml import etree
from argparse import ArgumentParser


def parse_images(dir_path):
    all_files = [f for f in os.listdir(dir_path) if osp.isfile(osp.join(dir_path, f))]
    image_files = [f for f in all_files if f.endswith('.jpg')]

    return image_files


def dump_fake_labels(image_names, out_path, num_pids=333, num_cams=33):
    root = etree.Element('TrainingImages')
    items = etree.SubElement(root, 'Items', number=str(len(image_names)))
    for image_name in image_names:
        pid_str = '{:04}'.format(rnd.randint(1, num_pids))
        cam_id_str = 'c{:03}'.format(rnd.randint(1, num_cams))
        etree.SubElement(items, 'Item', imageName=image_name, vehicleID=pid_str, cameraID=cam_id_str)

    with open(out_path, 'wb') as output:
        output.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='utf-8'))


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    args = parser.parse_args()

    assert osp.exists(args.input)

    image_names = parse_images(args.input)
    print('Found {} images'.format(len(image_names)))

    dump_fake_labels(image_names, args.output)
    print('Fake labels has been stored at: {}'.format(args.output))


if __name__ == '__main__':
    main()

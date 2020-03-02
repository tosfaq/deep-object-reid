import os.path as osp
from lxml import etree
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt


def load_annotation(annot_path):
    tree = etree.parse(annot_path)
    root = tree.getroot()

    assert len(root) == 1
    items = root[0]

    data = {}
    for item in items:
        image_name = item.attrib['imageName']
        data[image_name] = dict(pid=int(item.attrib['vehicleID']),
                                cam_id=int(item.attrib['cameraID'][1:]),
                                color=int(item.attrib['colorID']),
                                type=int(item.attrib['typeID']),
                                orientation=float(item.attrib['orientation']),
                                light_int=float(item.attrib['lightInt']),
                                light_dir=float(item.attrib['lightDir']),
                                cam_dis=float(item.attrib['camDis']),
                                cam_h=float(item.attrib['camHei']))

    return data


def get_values(data, record_name):
    return [item[record_name] for item in data.values()]


def print_stat(data, n_bins=50):
    print('Total num records: {}'.format(len(data)))

    cam_ids = get_values(data, 'cam_id')
    colors = get_values(data, 'color')
    types = get_values(data, 'type')
    orientations = get_values(data, 'orientation')
    light_int = get_values(data, 'light_int')
    light_dir = get_values(data, 'light_dir')
    cam_dis = get_values(data, 'cam_dis')
    cam_h = get_values(data, 'cam_h')

    fig, axs = plt.subplots(2, 4, tight_layout=True)
    axs[0, 0].hist(cam_ids, bins=n_bins)
    axs[0, 0].set_title('cam_ids')
    axs[0, 1].hist(colors, bins=n_bins)
    axs[0, 1].set_title('colors')
    axs[0, 2].hist(types, bins=n_bins)
    axs[0, 2].set_title('types')
    axs[0, 3].hist(orientations, bins=n_bins)
    axs[0, 3].set_title('orientations')
    axs[1, 0].hist(light_int, bins=n_bins)
    axs[1, 0].set_title('light_int')
    axs[1, 1].hist(light_dir, bins=n_bins)
    axs[1, 1].set_title('light_dir')
    axs[1, 2].hist(cam_dis, bins=n_bins)
    axs[1, 2].set_title('cam_dis')
    axs[1, 3].hist(cam_h, bins=n_bins)
    axs[1, 3].set_title('cam_h')
    plt.show()


def quantize_angles(data, bin_centers):
    num_classes = len(bin_centers)
    bin_centers = bin_centers + [360]

    data = np.array(data).astype(np.float32).reshape(-1, 1)
    bin_centers = np.array(bin_centers).astype(np.float32).reshape(1, -1)

    dist = np.abs(data - bin_centers)

    classes = np.argmin(dist, axis=-1)
    classes[classes == num_classes] = 0

    return classes


def main():
    parser = ArgumentParser()
    parser.add_argument('--annot', '-a', type=str, required=True)
    args = parser.parse_args()

    assert osp.exists(args.annot)

    data = load_annotation(args.annot)
    # print_stat(data)

    orientations = get_values(data, 'orientation')
    centers = [0, 30, 60, 90, 120, 210, 240, 270, 300, 330]
    orientation_classes = quantize_angles(orientations, centers)


if __name__ == '__main__':
    main()

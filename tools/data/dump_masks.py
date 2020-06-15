"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
from os.path import exists, join, abspath, isfile
from os import listdir, walk, makedirs
from shutil import rmtree

import cv2
import numpy as np
import mmcv
import pycocotools.mask as mask_utils
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector, show_result


COCO_VEHICLE_CLASSES = 2, 5, 7  # car, bus, truck
MIN_DET_CONF = 0.1
MIN_AREA_SIZE = 0.3


def create_dirs(dir_path, override=False):
    if override:
        if exists(dir_path):
            rmtree(dir_path)
        makedirs(dir_path)
    elif not exists(dir_path):
        makedirs(dir_path)


def parse_relative_paths(data_dir):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    relative_paths = []
    for root, sub_dirs, files in walk(data_dir):
        if len(sub_dirs) == 0 and len(files) > 0:
            relative_paths.append(root[skip_size:])

    return relative_paths


def prepare_tasks(relative_paths, input_dir, output_dir):
    out_data = []
    for relative_path in relative_paths:
        images = join(input_dir, relative_path)
        assert exists(images)

        output_path = join(output_dir, relative_path)
        if not exists(output_path):
            makedirs(output_path)

        image_files = [f for f in listdir(images)
                       if isfile(join(images, f)) and f.endswith('.jpg')]
        for image_file in image_files:
            full_input_path = join(images, image_file)
            full_output_path = join(output_path, '{}.png'.format(image_file.split('.')[0]))
            if not exists(full_output_path):
                out_data.append((full_input_path, full_output_path))

    return out_data


def parse_result(result, class_ids, threshold=0.4):
    bbox_result, segm_result = result

    out_data = list()
    for class_id in class_ids:
        bboxes = bbox_result[class_id]
        masks = segm_result[class_id]
        assert bboxes.shape[0] == len(masks)

        for i in range(len(masks)):
            bbox = bboxes[i]
            mask = masks[i]

            if bbox[-1] > threshold:
                out_data.append((bbox[:-1], mask))

    return out_data


def get_main_mask(objects, img_size, min_area):
    img_height, img_width = img_size[:2]

    areas = []
    for bbox, _ in objects:
        bbox_width = max(0, bbox[2] - bbox[0])
        bbox_height = max(0, bbox[3] - bbox[1])       
        areas.append(float(bbox_width * bbox_height) / float(img_height * img_width))
    areas = np.array(areas, dtype=np.float32)

    best_match = np.argmax(areas)
    best_area = areas[best_match]
    if best_area < min_area:
        return None

    return mask_utils.decode(objects[best_match][1]).astype(np.bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True, help='test config file path')
    parser.add_argument('--checkpoint', '-m', type=str, required=True, help='checkpoint file')
    parser.add_argument('--input-dir', 'i', type=str, required=True, help='input dir')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='output dir')
    args = parser.parse_args()

    assert exists(args.input_dir)
    create_dirs(args.output_dir, override=False)

    print('Loading model ...')
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    print('Preparing tasks ...')
    relative_paths = parse_relative_paths(args.input_dir)
    tasks = prepare_tasks(relative_paths, args.input_dir, args.output_dir)

    print('Dumping masks ...')
    invalid_sources = []
    for in_image_path, out_image_path in tqdm(tasks):
        img = mmcv.imread(in_image_path)
        if img is None:
            continue

        img_height, img_width = img.shape[:2]
        if img_height < 50 or img_width < 50:
            continue

        result = inference_detector(model, img)
        objects = parse_result(result, COCO_VEHICLE_CLASSES, threshold=MIN_DET_CONF)
        if len(objects) == 0:
            invalid_sources.append(in_image_path)
            continue

        vehicle_mask = get_main_mask(objects, img.shape, min_area=MIN_AREA_SIZE)
        if vehicle_mask is None:
            invalid_sources.append(in_image_path)
            continue

        cv2.imwrite(out_image_path, 255 * vehicle_mask.astype(np.uint8))

    if len(invalid_sources) > 0:
        print('Invalid sources:')
        for invalid_source in invalid_sources:
            print('   - {}'.format(invalid_source))
        print('Total invalid sources: {}'.format(len(invalid_sources)))


if __name__ == '__main__':
    main()

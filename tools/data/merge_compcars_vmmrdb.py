"""
 Copyright (c) 2020-2021 Intel Corporation

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
from os import makedirs, walk
from os.path import abspath, exists, join
from shutil import copyfile, rmtree

from scipy.io import loadmat

VMMRDB_COMPLEX_NAMES = 'alfa_romeo', 'aston_martin', 'am_general', 'can_am', 'mercedes_benz'


def create_dirs(dir_path, override=False):
    if dir_path == '':
        return

    if override:
        if exists(dir_path):
            rmtree(dir_path)
        makedirs(dir_path)
    elif not exists(dir_path):
        makedirs(dir_path)


def show_stat(data, header):
    num_classes = sum([len(y) for m in data.values() for y in m.values()])
    num_records = sum([len(r) for m in data.values() for y in m.values() for r in y.values()])

    print('{} stat: {} classes, {} records'.format(header, num_classes, num_records))


def norm_str(string):
    return string.strip().lower().replace('-', '').replace(' ', '').replace('.', '').replace('‘', '')


def load_compcars_models_map(map_file):
    data = loadmat(map_file)

    models_data = {i + 1: norm_str(m[0]) if len(m) != 0 else 'unknown'
                   for i, m in enumerate(data['model_names'].reshape(-1))}

    return models_data


def load_compcars_makes_map(map_file, id_shift=1):
    out_data = dict()
    with open(map_file) as input_stream:
        for i, line in enumerate(input_stream):
            name = line.strip()
            out_data[i + id_shift] = norm_str(name)

    return out_data


def parse_data_compcars(in_images_dir, in_masks_dir, makes_map, models_map):
    in_images_dir = abspath(in_images_dir)
    skip_size = len(in_images_dir) + 1

    out_data = dict()
    for root, sub_dirs, image_names in walk(in_images_dir):
        if len(sub_dirs) == 0 and len(image_names) > 0:
            relative_path = root[skip_size:]
            relative_path_parts = relative_path.split('/')
            assert len(relative_path_parts) == 3

            make_id, model_id, year = relative_path_parts
            if year == 'unknown':
                continue

            make = makes_map[int(make_id)]
            model = models_map[int(model_id)]
            year = int(year)

            if model.startswith(make):
                model = norm_str(model[len(make):])
            elif model.startswith('benz'):
                model = norm_str(model[len('benz'):])
            elif model.startswith('bwm'):
                model = norm_str(model[len('bwm'):])
            elif make == 'tesla' and model.startswith('model'):
                model = norm_str(model[len('model'):])
            elif make == 'renault' and model.startswith('reno'):
                model = norm_str(model[len('reno'):])
            elif make == 'jaguar' and model.startswith('gaguar'):
                model = norm_str(model[len('gaguar'):])

            if len(model) == 0 or model == 'unknown':
                continue

            if make not in out_data:
                out_data[make] = dict()
            makes_dict = out_data[make]

            if model not in makes_dict:
                makes_dict[model] = dict()
            models_dict = makes_dict[model]

            if year not in models_dict:
                models_dict[year] = []

            records = []
            for image_name in image_names:
                image_full_path = join(root, image_name)
                image_valid = exists(image_full_path)

                if in_masks_dir != '':
                    mask_name = '{}.png'.format(image_name.split('.')[0])
                    mask_full_path = join(in_masks_dir, relative_path, mask_name)
                    mask_valid = exists(mask_full_path)
                else:
                    mask_full_path = ''
                    mask_valid = True

                if image_valid and mask_valid:
                    records.append((image_full_path, mask_full_path))

            models_dict[year].extend(records)

    return out_data


def parse_data_vmmrdb(in_images_dir, in_masks_dir, complex_names_list):
    in_images_dir = abspath(in_images_dir)
    skip_size = len(in_images_dir) + 1

    out_data = dict()
    for root, sub_dirs, image_names in walk(in_images_dir):
        if len(sub_dirs) == 0 and len(image_names) > 0:
            relative_path = root[skip_size:]
            fixed_relative_path = relative_path.replace(' ', '_')
            relative_path_parts = [p for p in fixed_relative_path.split('_')]

            make_num_parts = 1
            if fixed_relative_path.startswith(complex_names_list):
                make_num_parts += 1

            assert len(relative_path_parts) >= make_num_parts + 2

            make = norm_str(''.join(relative_path_parts[:make_num_parts]))
            model = norm_str(''.join(relative_path_parts[make_num_parts:-1]))
            year = int(relative_path_parts[-1])

            if make not in out_data:
                out_data[make] = dict()
            makes_dict = out_data[make]

            if model not in makes_dict:
                makes_dict[model] = dict()
            models_dict = makes_dict[model]

            if year not in models_dict:
                models_dict[year] = []

            records = []
            for image_name in image_names:
                image_full_path = join(root, image_name)
                image_valid = exists(image_full_path)

                if in_masks_dir != '':
                    mask_name = '{}.png'.format(image_name.split('.')[0])
                    mask_full_path = join(in_masks_dir, relative_path, mask_name)
                    mask_valid = exists(mask_full_path)
                else:
                    mask_full_path = ''
                    mask_valid = True

                if image_valid and mask_valid:
                    records.append((image_full_path, mask_full_path))

            models_dict[year].extend(records)

    return out_data


def merge_data(data_a, data_b):
    out_data = data_a
    for make_b, models_b in data_b.items():
        if make_b not in out_data:
            out_data[make_b] = models_b
        else:
            models_a = out_data[make_b]
            for model_b, years_b in models_b.items():
                if model_b not in models_a:
                    models_a[model_b] = years_b
                else:
                    years_a = models_a[model_b]
                    for year_b, records_b in years_b.items():
                        if year_b not in years_a:
                            years_a[year_b] = records_b
                        else:
                            years_a[year_b].extend(records_b)

    return out_data


def filter_data(data, min_num_records):
    out_data = dict()
    for make, models in data.items():
        out_models = dict()
        for model, years in models.items():
            out_years = dict()
            for year, records in years.items():
                if len(records) >= min_num_records:
                    out_years[year] = records

            if len(out_years) > 0:
                out_models[model] = out_years

        if len(out_models) > 0:
            out_data[make] = out_models

    return out_data


def copy_data(data, out_dir, need_masks=False):
    out_images_dir = join(out_dir, 'images')
    out_masks_dir = join(out_dir, 'masks') if need_masks else ''

    create_dirs(out_images_dir, override=True)
    create_dirs(out_masks_dir, override=True)

    for make, models in data.items():
        out_images_make_dir = join(out_images_dir, make)
        out_masks_make_dir = join(out_masks_dir, make) if need_masks else ''

        create_dirs(out_images_make_dir, override=False)
        create_dirs(out_masks_make_dir, override=False)

        for model, years in models.items():
            out_images_model_dir = join(out_images_make_dir, model)
            out_masks_model_dir = join(out_masks_make_dir, model) if need_masks else ''

            create_dirs(out_images_model_dir, override=False)
            create_dirs(out_masks_model_dir, override=False)

            for year, records in years.items():
                out_images_year_dir = join(out_images_model_dir, str(year))
                out_masks_year_dir = join(out_masks_model_dir, str(year)) if need_masks else ''

                create_dirs(out_images_year_dir, override=False)
                create_dirs(out_masks_year_dir, override=False)

                for record_id, record in enumerate(records):
                    name = 'instance_{:05}'.format(record_id)
                    in_image_path, in_mask_path = record

                    out_image_path = join(out_images_year_dir, '{}.jpg'.format(name))
                    copyfile(in_image_path, out_image_path)

                    if need_masks and in_mask_path != '':
                        out_mask_path = join(out_masks_year_dir, '{}.png'.format(name))
                        copyfile(in_mask_path, out_mask_path)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--compcar-map', '-cmp', type=str, required=True)
    parser.add_argument('--compcar-makes', '-cmk', type=str, required=True)
    parser.add_argument('--compcar-images', '-ci', type=str, required=True)
    parser.add_argument('--compcar-masks', '-cm', type=str, required=False, default='')
    parser.add_argument('--vmmrdb-images', '-vi', type=str, required=True)
    parser.add_argument('--vmmrdb-masks', '-vm', type=str, required=False, default='')
    parser.add_argument('--min-num-images', '-n', type=int, required=False, default=2)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.compcar_map)
    assert exists(args.compcar_makes)
    assert exists(args.compcar_images)
    assert exists(args.vmmrdb_images)

    need_masks = args.compcar_masks != '' and args.vmmrdb_masks != ''
    if need_masks:
        assert exists(args.compcar_masks)
        assert exists(args.vmmrdb_masks)

    compcars_makes_map = load_compcars_makes_map(args.compcar_makes)
    compcars_models_map = load_compcars_models_map(args.compcar_map)
    compcars = parse_data_compcars(args.compcar_images, args.compcar_masks, compcars_makes_map, compcars_models_map)
    show_stat(compcars, 'CompCars')

    vmmrdb = parse_data_vmmrdb(args.vmmrdb_images, args.vmmrdb_masks, VMMRDB_COMPLEX_NAMES)
    show_stat(vmmrdb, 'VMMRdb')

    merged_data = merge_data(compcars, vmmrdb)
    show_stat(merged_data, 'Merged')

    filtered_data = filter_data(merged_data, min_num_records=args.min_num_images)
    show_stat(filtered_data, 'Filtered')

    copy_data(filtered_data, args.output_dir, need_masks)


if __name__ == '__main__':
    main()

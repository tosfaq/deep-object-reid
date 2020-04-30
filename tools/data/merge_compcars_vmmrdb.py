import argparse
from os.path import exists, join, abspath, isfile
from os import listdir, walk, makedirs
from shutil import rmtree

from scipy.io import loadmat


VMMRDB_DOUBLE_NAMES = 'alfa_romeo', 'aston_martin', 'am_general', 'can_am', 'mercedes_benz'


def create_dirs(dir_path):
    if exists(dir_path):
        rmtree(dir_path)

    makedirs(dir_path)


def show_stat(data, header):
    num_classes = sum([len(y) for m in data.values() for y in m.values()])
    num_records = sum([len(r) for m in data.values() for y in m.values() for r in y.values()])

    print('{} stat: {} classes, {} records'.format(header, num_classes, num_records))


def norm_str(string):
    return string.lower().replace('-', '').replace(' ', '').replace('.', '')


def load_compcars_maps(map_file):
    data = loadmat(map_file)

    makes_data = {i + 1: norm_str(m[0])
                  for i, m in enumerate(data['make_names'].reshape(-1))}
    models_data = {i + 1: norm_str(m[0]) if len(m) != 0 else 'unknown'
                   for i, m in enumerate(data['model_names'].reshape(-1))}

    return makes_data, models_data


def load_map(map_file, id_shift=1):
    out_data = dict()
    with open(map_file) as input_stream:
        for i, line in enumerate(input_stream):
            name = line.strip()
            out_data[i + id_shift] = norm_str(name)

    return out_data


def dump_map(map_dict, out_path):
    keys = list(map_dict.keys())
    keys.sort()

    with open(out_path, 'w') as output_stream:
        for key in keys:
            output_stream.write('{}\n'.format(map_dict[key]))


def parse_data_compcars(data_dir, makes_map, models_map):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    out_data = dict()
    for root, sub_dirs, files in walk(data_dir):
        if len(sub_dirs) == 0 and len(files) > 0:
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

            image_files = [join(root, f) for f in files]
            models_dict[year].extend(image_files)

    return out_data


def parse_data_vmmrdb(data_dir, double_names):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    out_data = dict()
    for root, sub_dirs, files in walk(data_dir):
        if len(sub_dirs) == 0 and len(files) > 0:
            relative_path = root[skip_size:].replace(' ', '_')
            relative_path_parts = [p for p in relative_path.split('_')]

            make_num_parts = 1
            if relative_path.startswith(double_names):
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

            image_files = [join(root, f) for f in files]
            models_dict[year].extend(image_files)

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


def filter_data(data, min_num_images):
    out_data = dict()
    for make, models in data.items():
        out_models = dict()
        for model, years in models.items():
            out_years = dict()
            for year, files in years.items():
                if len(files) >= min_num_images:
                    out_years[year] = files

            if len(out_years) > 0:
                out_models[model] = out_years

        if len(out_models) > 0:
            out_data[make] = out_models

    return out_data


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--compcar-map', '-cm', type=str, required=True)
    parser.add_argument('--compcar-dir', '-ci', type=str, required=True)
    parser.add_argument('--vmmrdb-dir', '-vi', type=str, required=True)
    parser.add_argument('--min-num-images', '-n', type=int, required=False, default=5)
    # parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.compcar_map)
    assert exists(args.compcar_dir)
    assert exists(args.vmmrdb_dir)

    # create_dirs(args.output_dir)

    compcars_makes_map, compcars_models_map = load_compcars_maps(args.compcar_map)
    compcars_makes_map = load_map('/media/datasets/ReID/Vehicle/CompCars/data/makes.txt')

    compcars = parse_data_compcars(args.compcar_dir, compcars_makes_map, compcars_models_map)
    show_stat(compcars, 'CompCars')

    vmmrdb = parse_data_vmmrdb(args.vmmrdb_dir, VMMRDB_DOUBLE_NAMES)
    show_stat(vmmrdb, 'VMMRdb')

    merged_data = merge_data(compcars, vmmrdb)
    show_stat(merged_data, 'Merged')

    filtered_data = filter_data(merged_data, min_num_images=args.min_num_images)
    show_stat(filtered_data, 'Filtered')


if __name__ == '__main__':
    main()

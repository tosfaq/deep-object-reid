import argparse
from os.path import exists, join, abspath, isfile
from os import listdir, walk, makedirs
from shutil import rmtree

from scipy.io import loadmat


def create_dirs(dir_path):
    if exists(dir_path):
        rmtree(dir_path)

    makedirs(dir_path)


def show_stat(data, header):
    num_classes = sum([len(y) for m in data.values() for y in m.values()])
    num_records = sum([len(r) for m in data.values() for y in m.values() for r in y.values()])

    print('{} stat: {} classes, {} records'.format(header, num_classes, num_records))


def load_compcars_maps(map_file):
    data = loadmat(map_file)

    makes_data = {i: m[0] for i, m in enumerate(data['make_names'].reshape(-1))}
    models_data = {i: m[0] if len(m) != 0 else i for i, m in enumerate(data['model_names'].reshape(-1))}

    return makes_data, models_data


def parse_data_compcars(data_dir, makes_map, models_map):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    out_data = dict()
    for root, sub_dirs, files in walk(data_dir):
        if len(sub_dirs) == 0 and len(files) > 0:
            relative_path = root[skip_size:]
            relative_path_parts = relative_path.split('/')
            assert len(relative_path_parts) == 3

            make, model, year = relative_path_parts

            make = makes_map[int(make)]
            model = models_map[int(model)]
            year = int(year)

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


def parse_data_vmmrdb(data_dir):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    out_data = dict()
    for root, sub_dirs, files in walk(data_dir):
        if len(sub_dirs) == 0 and len(files) > 0:
            relative_path = root[skip_size:]
            relative_path_parts = relative_path.split('_')
            assert len(relative_path_parts) >= 3

            make = relative_path_parts[0]
            model = '_'.join(relative_path_parts[1:-1])
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
            for model_b, years_b in models_b:
                if model_b not in models_a:
                    models_a[model_b] = years_b
                else:
                    years_a = models_a[model_b]
                    for year_b, records_b in years_b:
                        if years_b not in years_a:
                            years_a[year_b] = records_b
                        else:
                            years_a[year_b].extend(records_b)

    return out_data


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--compcar-map', '-cm', type=str, required=True)
    parser.add_argument('--compcar-dir', '-ci', type=str, required=True)
    parser.add_argument('--vmmrdb-dir', '-vi', type=str, required=True)
    # parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.compcar_map)
    assert exists(args.compcar_dir)
    assert exists(args.vmmrdb_dir)

    # create_dirs(args.output_dir)

    compcars_makes_map, compcars_models_map = load_compcars_maps(args.compcar_map)
    compcars = parse_data_compcars(args.compcar_dir, compcars_makes_map, compcars_models_map)
    show_stat(compcars, 'CompCars')

    vmmrdb = parse_data_vmmrdb(args.vmmrdb_dir)
    show_stat(vmmrdb, 'VMMRdb')

    merged_data = merge_data(compcars, vmmrdb)
    show_stat(merged_data, 'Merged')


if __name__ == '__main__':
    main()

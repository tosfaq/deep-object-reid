from __future__ import division, print_function, absolute_import
from os.path import join, isfile, expanduser, abspath
from os import walk, listdir

from ..dataset import ImageDataset


COLORS_MAP = {
    'red': 7,
    'black': 3,
    'dark gray': 2,
    'green': 11,
    'white': 0,
    'gray': 1,
    'blue': 10,
    'purple': 8,
    'golden': 5,
    'yellow': 4,
    'cyan': 9,
    'brown': 12
}

TYPES_MAP = {
    'minivan': 10,
    'large-sized bus': 12,
    'small-sized truck': 13,
    'HGV/large truck': 14,
    'SUV': 1,
    ' small-sized truck': 13,
    'tank car/tanker': 15,
    'minibus': 11,
    'pickup truck': 6,
    'bulk lorry/fence truck': 16,
    'sedan': 0,
    'light passenger vehicle': 17,
    'business purpose vehicle/MPV': 18
}


class VeRiWild(ImageDataset):
    """VeRi-Wild dataset.

            URL: `<https://github.com/PKU-IMRE/VERI-Wild>`

            Dataset statistics:
                - identities: 40671.
                - images: 416314.
    """

    dataset_dir = 'veri-wild'

    def __init__(self, root='', dataset_id=0, min_num_samples=2, load_masks=False, **kwargs):
        self.root = abspath(expanduser(root))
        self.dataset_dir = join(self.root, self. dataset_dir)
        self.data_dir = self.dataset_dir

        self.images_dir = join(self.data_dir, 'images')
        self.attr_file = join(self.data_dir, 'vehicle_info.txt')

        required_files = [
            self.data_dir, self.images_dir, self.attr_file
        ]
        self.check_before_run(required_files)

        attr_info = self.load_attributes(self.attr_file)

        train = self.load_annotation(
            self.images_dir,
            attr_info,
            dataset_id=dataset_id,
            min_num_samples=min_num_samples,
            load_masks=load_masks
        )
        train = self._compress_labels(train)

        query, gallery = [], []

        super(VeRiWild, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_attributes(attr_file):
        out_data = dict()
        with open(attr_file) as input_stream:
            for line_id, line in enumerate(input_stream):
                if line_id < 1:  # skip header
                    continue

                parts = line.strip().split(';')
                assert len(parts) == 6

                image_name, camera_id, _, model_name, type_name, color_name = parts
                camera_id = int(camera_id)
                color_id = COLORS_MAP[color_name] if color_name in COLORS_MAP else -1
                type_id = TYPES_MAP[type_name] if type_name in TYPES_MAP else -1

                out_data[image_name] = camera_id, color_id, type_id

        return out_data

    @staticmethod
    def load_annotation(data_dir, attr_info, dataset_id=0, min_num_samples=1, load_masks=False):
        if load_masks:
            raise NotImplementedError

        base_dirs = []
        for root, sub_dirs, files in walk(data_dir):
            if len(sub_dirs) == 0 and len(files) >= min_num_samples:
                base_dirs.append(root)

        out_data = []
        for class_id, base_dir in enumerate(base_dirs):
            image_files = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

            for image_name in image_files:
                rel_image_name = '{}/{}'.format(base_dir.split('/')[-1], image_name.split('.')[0])
                if rel_image_name not in attr_info:
                    continue

                camera_id, color_id, type_id = attr_info[rel_image_name]
                full_image_path = join(base_dir, image_name)

                out_data.append((full_image_path, class_id, camera_id, dataset_id, '', color_id, type_id))

        return out_data

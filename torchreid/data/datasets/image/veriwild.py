from __future__ import division, print_function, absolute_import
from os.path import join, isfile, expanduser, abspath
from os import walk, listdir

from ..dataset import ImageDataset


class VeRiWild(ImageDataset):
    """VeRi-Wild dataset.

            URL: `<https://github.com/PKU-IMRE/VERI-Wild>`

            Dataset statistics:
                - identities: 40671.
                - images: 416314.
    """

    dataset_dir = 'veri-wild'

    def __init__(self, root='', dataset_id=0, min_num_samples=2, **kwargs):
        self.root = abspath(expanduser(root))
        self.dataset_dir = join(self.root, self. dataset_dir)
        self.data_dir = self.dataset_dir

        self.images_dir = join(self.data_dir, 'images')

        required_files = [
            self.data_dir, self.images_dir
        ]
        self.check_before_run(required_files)

        train = self.load_annotation(
            self.images_dir,
            dataset_id=dataset_id,
            min_num_samples=min_num_samples
        )
        train = self.compress_labels(train)

        query, gallery = [], []

        super(VeRiWild, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(data_dir, dataset_id=0, min_num_samples=1):
        base_dirs = []
        for root, sub_dirs, files in walk(data_dir):
            if len(sub_dirs) == 0 and len(files) >= min_num_samples:
                base_dirs.append(root)

        out_data = []
        for class_id, base_dir in enumerate(base_dirs):
            image_files = [join(base_dir, f) for f in listdir(base_dir) if isfile(join(base_dir, f))]

            for image_path in image_files:
                out_data.append((image_path, class_id, 0, dataset_id, '', -1, -1))

        return out_data

from __future__ import division, print_function, absolute_import
from os.path import join, isfile, expanduser, abspath
from os import walk, listdir

from ..dataset import ImageDataset


class CompCarsBase(ImageDataset):
    def __init__(self, dataset_dir, root='', dataset_id=0, **kwargs):
        self.root = abspath(expanduser(root))
        self.dataset_dir = join(self.root, dataset_dir)
        self.data_dir = self.dataset_dir

        self.images_dir = join(self.data_dir, 'image')

        required_files = [
            self.data_dir, self.images_dir
        ]
        self.check_before_run(required_files)

        train = self.load_annotation(self.images_dir, dataset_id=dataset_id)
        train = self.compress_labels(train)

        query, gallery = [], []

        super(CompCarsBase, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(data_dir, dataset_id=0):
        base_dirs = []
        for root, sub_dirs, files in walk(data_dir):
            if len(sub_dirs) == 0 and len(files) > 0:
                base_dirs.append(root)

        out_data = []
        for class_id, base_dir in enumerate(base_dirs):
            image_files = [join(base_dir, f) for f in listdir(base_dir) if isfile(join(base_dir, f))]

            for image_path in image_files:
                out_data.append((image_path, class_id, 0, dataset_id, -1, -1))

        return out_data


class CompCars(CompCarsBase):
    """CompCars.

        URL: `<http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html>`_

        Dataset statistics:
            - identities: 4446.
            - images: 136726.
        """

    dataset_dir = 'compcars'

    def __init__(self, **kwargs):
        super().__init__(CompCars.dataset_dir, **kwargs)


class CompCarsSV(CompCarsBase):
    """CompCars.

        URL: `<http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html>`_

        Dataset statistics:
            - identities: .
            - images: .
        """

    dataset_dir = 'compcars_sv'

    def __init__(self, **kwargs):
        super().__init__(CompCarsSV.dataset_dir, **kwargs)

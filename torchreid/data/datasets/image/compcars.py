from __future__ import division, print_function, absolute_import
from os.path import join, isfile, expanduser, abspath
from os import walk, listdir

from ..dataset import ImageDataset


class CompCars(ImageDataset):
    """CompCars.

            URL: `<http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html>`_

            Dataset statistics (if min_num_samples parameter is set to 2):
                - identities: 4385.
                - images: 135996.
            """

    dataset_dir = 'compcars'

    def __init__(self, root='', dataset_id=0, min_num_samples=2, load_masks=False, **kwargs):
        self.root = abspath(expanduser(root))
        self.dataset_dir = join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir

        self.images_dir = join(self.data_dir, 'image')
        self.masks_dir = join(self.data_dir, 'mask')

        required_files = [
            self.data_dir, self.images_dir
        ]
        if load_masks:
            required_files.append(self.masks_dir)
        self.check_before_run(required_files)

        train = self.load_annotation(
            self.images_dir,
            self.masks_dir,
            dataset_id=dataset_id,
            min_num_samples=min_num_samples,
            load_masks=load_masks
        )
        train = self._compress_labels(train)

        query, gallery = [], []

        super(CompCars, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def load_annotation(images_dir, masks_dir=None, dataset_id=0, min_num_samples=1, load_masks=False):
        relative_path_shift = len(abspath(images_dir)) + 1

        base_dirs = []
        for root, sub_dirs, files in walk(images_dir):
            if len(sub_dirs) == 0 and len(files) >= min_num_samples:
                if root.endswith('unknown'):
                    continue

                relative_path = root[relative_path_shift:]
                base_dirs.append(relative_path)

        out_data = []
        for class_id, base_dir in enumerate(base_dirs):
            local_images_dir = join(images_dir, base_dir)
            names = [f.split('.')[0] for f in listdir(local_images_dir) if isfile(join(local_images_dir, f))]

            if load_masks:
                local_masks_dir = join(masks_dir, base_dir)
                mask_names = [f.split('.')[0] for f in listdir(local_masks_dir) if isfile(join(local_masks_dir, f))]
                names = list(set(mask_names) & set(names))

            for name in names:
                image_path = join(local_images_dir, '{}.jpg'.format(name))

                mask_path = ''
                if load_masks:
                    mask_path = join(local_masks_dir, '{}.png'.format(name))

                out_data.append((image_path, class_id, 0, dataset_id, mask_path, -1, -1))

        return out_data

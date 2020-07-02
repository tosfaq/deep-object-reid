from __future__ import division, print_function, absolute_import

import math
import random
from os.path import exists, join
from collections import deque

import cv2
import numpy as np
import torch
from torchvision.transforms import *
from torchvision.transforms import functional as F
from torchreid.utils.tools import read_image
from PIL import Image

from ..data.datasets.image.lfw import FivePointsAligner


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input_tuple):
        if random.random() > self.p:
            return input_tuple

        img, mask = input_tuple

        img = F.hflip(img)
        mask = F.hflip(mask) if mask != '' else mask

        return img, mask


class RandomCrop(object):
    def __init__(self, p=0.5, scale=0.9, **kwargs):
        self.p = p
        assert 0.0 <= self.p <= 1.0
        self.scale = scale
        assert 0.0 < self.scale < 1.0

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        img_width, img_height = img.size
        crop_width, crop_height = int(round(img_width * self.scale)), int(round(img_height * self.scale))

        x_max_range = img_width - crop_width
        y_max_range = img_height - crop_height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))

        img = img.crop((x1, y1, x1 + crop_width, y1 + crop_height))
        mask = mask.crop((x1, y1, x1 + crop_width, y1 + crop_height)) if mask != '' else mask

        return img, mask


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
    """

    def __init__(self, p=0.5, sl=0.02, sh=0.4, rl=0.5, rh=2.0, fill_color=None, norm_image=True, **kwargs):
        self.probability = p
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = rh
        self.norm_image = norm_image
        self.fill_color = fill_color
        if self.fill_color is not None and len(self.fill_color) in (1, 3):
            if len(self.fill_color) == 1:
                self.fill_color = [self.fill_color] * 3
        else:
            self.fill_color = None

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.probability:
            return input_tuple

        img, mask = input_tuple
        img_size = img.size() if self.norm_image else img.size

        for attempt in range(100):
            source_area = img_size[0] * img_size[1]
            target_area = random.uniform(self.sl, self.sh) * source_area
            aspect_ratio = random.uniform(self.rl, self.rh)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_size[1] and h < img_size[0]:
                x1 = random.randint(0, img_size[0] - h)
                y1 = random.randint(0, img_size[1] - w)

                fill_color = self.fill_color if self.fill_color is not None else [random.randint(0, 255)] * 3
                if self.norm_image:
                    fill_color = np.array(fill_color) / 255.0

                img = img if self.norm_image else np.array(img)
                img[x1:x1 + h, y1:y1 + w, 0] = fill_color[0]
                img[x1:x1 + h, y1:y1 + w, 1] = fill_color[1]
                img[x1:x1 + h, y1:y1 + w, 2] = fill_color[2]
                img = img if self.norm_image else Image.fromarray(img)

                return img, mask

        return img, mask


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.

    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.

    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5, **kwargs):
        self.p = p
        self.eig_vec = torch.Tensor([[0.4009, 0.7192, -0.5675],
                                     [-0.8140, -0.0045, -0.5808],
                                     [0.4203, -0.6948, -0.5836],
                                     ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        tensor, mask = input_tuple

        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)

        return tensor, mask


class RandomPatch(object):
    """Random patch data augmentation.

    There is a patch pool that stores randomly extracted pathces from person images.
    
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, p=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1,
                 prob_rotate=0.5, prob_flip_leftright=0.5, **kwargs):
        self.prob_happen = p

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1.0 / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h

        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, input_tuple):
        img, mask = input_tuple

        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img, mask

        if random.uniform(0, 1) > self.prob_happen:
            return img, mask

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img, mask


class RandomColorJitter(ColorJitter):
    def __init__(self, p=0.5, brightness=0.2, contrast=0.15, saturation=0, hue=0, **kwargs):
        self.p = p
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        img = transform(img)

        return img, mask


class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        num_output_channels = 1 if img.mode == 'L' else 3
        img = F.to_grayscale(img, num_output_channels=num_output_channels)

        return img, mask


class RandomPadding(object):
    """Random padding
    """

    def __init__(self, p=0.5, padding=(0, 10), **kwargs):
        self.p = p
        self.padding_limits = padding

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        rnd_padding = [random.randint(self.padding_limits[0], self.padding_limits[1]) for _ in range(4)]
        rnd_fill = random.randint(0, 255)

        img = F.pad(img, tuple(rnd_padding), fill=rnd_fill, padding_mode='constant')
        mask = F.pad(mask, tuple(rnd_padding), fill=0, padding_mode='constant') if mask != '' else mask

        return img, mask


class RandomRotate(object):
    """Random rotate
    """

    def __init__(self, p=0.5, angle=(-5, 5), **kwargs):
        self.p = p
        self.angle = angle

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        rnd_angle = random.randint(self.angle[0], self.angle[1])
        img = F.rotate(img, rnd_angle, resample=False, expand=False, center=None)
        if mask != '':
            rgb_mask = mask.convert('RGB')
            rgb_mask = F.rotate(rgb_mask, rnd_angle, resample=False, expand=False, center=None)
            mask = rgb_mask.convert('L')

        return img, mask


class RandomGrid(object):
    """Random grid
    """

    def __init__(self, p=0.5, color=-1, grid_size=(24, 64), thickness=(1, 1), angle=(0, 180), **kwargs):
        self.p = p
        self.color = color
        self.grid_size = grid_size
        self.thickness = thickness
        self.angle = angle

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        if self.color == (-1, -1, -1):  # Random color
            color = tuple([random.randint(0, 256) for _ in range(3)])
        else:
            color = self.color

        grid_size = random.randint(*self.grid_size)
        thickness = random.randint(*self.thickness)
        angle = random.randint(*self.angle)

        return self.draw_grid(img, grid_size, color, thickness, angle), mask

    @staticmethod
    def draw_grid(image, grid_size, color, thickness, angle):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        mask = np.zeros((h * 8, w * 8, 3), dtype='uint8')
        mask_h, mask_w = mask.shape[:2]
        for i in range(0, mask_h, grid_size):
            p1 = (0, i)
            p2 = (mask_w, i + grid_size)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)
        for i in range(0, mask_w, grid_size):
            p1 = (i, 0)
            p2 = (i + grid_size, mask_h)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)

        center = (mask_w // 2, mask_h // 2)

        if angle > 0:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, rot_mat, (mask_w, mask_h), flags=cv2.INTER_LINEAR)

        offset = (random.randint(-16, 16), random.randint(16, 16))
        center = (center[0] + offset[0], center[1] + offset[1])
        mask = mask[center[1] - h // 2: center[1] + h // 2, center[0] - w // 2: center[0] + w // 2, :]
        mask = cv2.resize(mask, (w, h))
        assert img.shape == mask.shape
        img = np.where(mask == 0, img, color).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


class RandomFigures(object):
    """Insert random figure or some figures from the list [line, rectangle, circle]
    with random color and thickness
    """

    def __init__(self, p=0.5, random_color=True, always_single_figure=False,
                 thicknesses=(1, 6), circle_radiuses=(5, 64), figure_prob=0.5, **kwargs):
        self.p = p
        self.random_color = random_color
        self.always_single_figure = always_single_figure
        self.figures = (cv2.line, cv2.rectangle, cv2.circle)
        self.thicknesses = thicknesses
        self.circle_radiuses = circle_radiuses
        self.figure_prob = figure_prob

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        image, mask = input_tuple

        if self.always_single_figure:
            figure = [self.figures[random.randint(0, len(self.figures) - 1)]]
        else:
            figure = []
            for i in range(len(self.figures)):
                if random.uniform(0, 1) > self.figure_prob:
                    figure.append(self.figures[i])

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = cv_image.shape[:2]
        for f in figure:
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            color = tuple([random.randint(0, 256) for _ in range(3)]) if self.random_color else (0, 0, 0)
            thickness = random.randint(*self.thicknesses)
            if f != cv2.circle:
                cv_image = f(cv_image, p1, p2, color, thickness)
            else:
                r = random.randint(*self.circle_radiuses)
                cv_image = f(cv_image, p1, r, color, thickness)

        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img), mask


class CutOutWithPrior(object):
    """Cut out around facial landmarks
    """

    def __init__(self, max_area=0.1, p=0.5, **kwargs):
        self.p = p
        self.max_area = max_area

    def __call__(self, input_tuple, *args, **kwargs):
        img, mask = input_tuple

        img = np.array(img)

        height, width = img.shape[:2]
        keypoints_ref = np.zeros((5, 2), dtype=np.float32)
        keypoints_ref[:, 0] = FivePointsAligner.ref_landmarks[:, 0] * width
        keypoints_ref[:, 1] = FivePointsAligner.ref_landmarks[:, 1] * height

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            erase_num = torch.LongTensor(1).random_(1, 4)
            erase_ratio = torch.FloatTensor(1).uniform_(self.max_area / 2, self.max_area)
            erase_h = math.sqrt(erase_ratio) / float(erase_num) * height
            erase_w = math.sqrt(erase_ratio) / float(erase_num) * width

            erased_idx = []
            for _ in range(erase_num):
                erase_pos = int(torch.LongTensor(1).random_(0, 5))
                while erase_pos in erased_idx:
                    erase_pos = int(torch.LongTensor(1).random_(0, 5))

                left_corner = (
                    int(keypoints_ref[erase_pos][0] - erase_h / 2), int(keypoints_ref[erase_pos][1] - erase_w / 2))
                right_corner = (
                    int(keypoints_ref[erase_pos][0] + erase_h / 2), int(keypoints_ref[erase_pos][1] + erase_w / 2))

                cv2.rectangle(img, tuple(left_corner), tuple(right_corner), (0, 0, 0), thickness=-1)
                erased_idx.append(erase_pos)

        img = Image.fromarray(img)
        return img, mask


class GaussianBlur(object):
    """Apply gaussian blur with random parameters
    """

    def __init__(self, p, k):
        self.p = p
        assert k % 2 == 1
        self.k = k

    def __call__(self, input_tuple, *args, **kwargs):
        img, mask = input_tuple

        img = np.array(img)
        if float(torch.FloatTensor(1).uniform_()) < self.p:
            img = cv2.blur(img, (self.k, self.k))

        img = Image.fromarray(img)
        return img, mask


class MixUp(object):
    """MixUp augmentation
    """

    def __init__(self, p=0.5, alpha=0.2, images_root_dir=None, images_list_file=None, **kwargs):
        self.p = p
        self.alpha = alpha

        self.enable = images_root_dir is not None and exists(images_root_dir) and \
                      images_list_file is not None and exists(images_list_file)

        if self.enable:
            all_image_files = []
            with open(images_list_file) as input_stream:
                for line in input_stream:
                    image_name = line.replace('\n', '')
                    if len(image_name) > 0:
                        image_path = join(images_root_dir, image_name)
                        all_image_files.append(image_path)
            self.surrogate_image_paths = all_image_files
            assert len(self.surrogate_image_paths) > 0

    def get_num_images(self):
        return len(self.surrogate_image_paths) if self.enable else 0

    def _load_surrogate_image(self, idx, trg_image_size):
        trg_image_width, trg_image_height = trg_image_size

        image = read_image(self.surrogate_image_paths[idx])

        scale = 1.125
        new_width, new_height = int(trg_image_width * scale), int(trg_image_height * scale)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        x_max_range = new_width - trg_image_width
        y_max_range = new_height - trg_image_height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))

        image = image.crop((x1, y1, x1 + trg_image_width, y1 + trg_image_height))

        if random.uniform(0, 1) > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def __call__(self, input_tuple, *args, **kwargs):
        if not self.enable or random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        alpha = np.random.beta(self.alpha, self.alpha)
        alpha = alpha if alpha < 0.5 else 1.0 - alpha

        surrogate_image_idx = random.randint(0, len(self.surrogate_image_paths) - 1)
        surrogate_image = self._load_surrogate_image(surrogate_image_idx, img.size)

        float_img = np.array(img).astype(np.float32)
        float_surrogate_image = np.array(surrogate_image).astype(np.float32)
        mixed_image = (1.0 - alpha) * float_img + alpha * float_surrogate_image

        out_image = mixed_image.clip(0.0, 255.0).astype(np.uint8)

        return Image.fromarray(out_image), mask


class RandomBackgroundSubstitution(object):
    """Random background substitution augmentation
    """

    def __init__(self, p=0.5, images_root_dir=None, images_list_file=None, **kwargs):
        self.p = p

        self.enable = images_root_dir is not None and exists(images_root_dir) and \
                      images_list_file is not None and exists(images_list_file)

        if self.enable:
            all_image_files = []
            with open(images_list_file) as input_stream:
                for line in input_stream:
                    image_name = line.replace('\n', '')
                    if len(image_name) > 0:
                        image_path = join(images_root_dir, image_name)
                        all_image_files.append(image_path)
            self.surrogate_image_paths = all_image_files
            assert len(self.surrogate_image_paths) > 0

    def get_num_images(self):
        return len(self.surrogate_image_paths) if self.enable else 0

    def _load_bg_image(self, idx, trg_image_size):
        trg_image_width, trg_image_height = trg_image_size

        image = read_image(self.surrogate_image_paths[idx])

        scale = 1.125
        new_width, new_height = int(trg_image_width * scale), int(trg_image_height * scale)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        x_max_range = new_width - trg_image_width
        y_max_range = new_height - trg_image_height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))

        image = image.crop((x1, y1, x1 + trg_image_width, y1 + trg_image_height))

        if random.uniform(0, 1) > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def __call__(self, input_tuple, *args, **kwargs):
        if not self.enable or random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        if mask == '':
            return input_tuple

        surrogate_image_idx = random.randint(0, len(self.surrogate_image_paths) - 1)
        surrogate_image = self._load_bg_image(surrogate_image_idx, img.size)

        src_img = np.array(img)
        out_img = np.array(surrogate_image)

        fg_mask = np.array(mask).astype(np.bool)
        out_img[fg_mask] = src_img[fg_mask]

        return Image.fromarray(out_img), mask


class DisableBackground(object):
    def __call__(self, input_tuple, *args, **kwargs):
        img, mask = input_tuple
        if mask == '':
            return input_tuple

        bg_mask = ~np.array(mask).astype(np.bool)

        img = np.array(img)
        img[bg_mask] = 0

        return Image.fromarray(img), mask


class PairResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input_tuple):
        image, mask = input_tuple

        image = F.resize(image, self.size, self.interpolation)
        mask = F.resize(mask, self.size, self.interpolation) if mask != '' else mask

        return image, mask


class PairNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, input_tuple):
        image, mask = input_tuple

        return F.normalize(image, self.mean, self.std, self.inplace), mask


class PairToTensor(object):
    def __call__(self, input_tuple):
        image, mask = input_tuple

        image = F.to_tensor(image)
        mask = F.to_tensor(mask) if mask != '' else mask

        return image, mask


def build_transforms(height, width, transforms=None, norm_mean=(0.485, 0.456, 0.406),
                     norm_std=(0.229, 0.224, 0.225), apply_masks_to_test=False, **kwargs):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        return None, None

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
        norm_std = [0.229, 0.224, 0.225]  # imagenet std

    print('Building train transforms ...')
    transform_tr = []
    if transforms.random_grid.enable and transforms.random_grid.before_resize:
        print('+ random_grid')
        transform_tr += [RandomGrid(**transforms.random_grid)]
    if transforms.random_figures.enable and transforms.random_figures.before_resize:
        print('+ random_figures')
        transform_tr += [RandomFigures(**transforms.random_figures)]
    if transforms.random_crop.enable:
        print('+ random crop')
        transform_tr += [RandomCrop(**transforms.random_crop)]
    if transforms.random_padding.enable:
        print('+ random_padding')
        transform_tr += [RandomPadding(**transforms.random_padding)]
    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [PairResize((height, width))]
    if transforms.random_background_substitution.enable:
        aug_module = RandomBackgroundSubstitution(**transforms.random_background_substitution)
        if aug_module.enable:
            print('+ random_background_substitution')
            transform_tr += [aug_module]
    if transforms.random_grid.enable and not transforms.random_grid.before_resize:
        print('+ random_grid')
        transform_tr += [RandomGrid(**transforms.random_grid)]
    if transforms.random_figures.enable and not transforms.random_figures.before_resize:
        print('+ random_figures')
        transform_tr += [RandomFigures(**transforms.random_figures)]
    if transforms.random_flip.enable:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip(p=transforms.random_flip.p)]
    if transforms.cut_out_with_prior.enable:
        print('+ cut out with prior')
        transform_tr += [CutOutWithPrior(p=transforms.cut_out_with_prior.p,
                                         max_area=transforms.cut_out_with_prior.max_area)]
    if transforms.random_blur.enable:
        print('+ random_blur')
        transform_tr += [GaussianBlur(p=transforms.random_blur.p,
                                         k=transforms.random_blur.k)]
    if transforms.mixup.enable:
        mixup_augmentor = MixUp(**transforms.mixup)
        print('+ mixup (with {} extra images)'.format(mixup_augmentor.get_num_images()))
        transform_tr += [mixup_augmentor]
    if transforms.random_patch.enable:
        print('+ random patch')
        transform_tr += [RandomPatch(**transforms.random_patch)]
    if transforms.color_jitter.enable:
        print('+ color jitter')
        transform_tr += [RandomColorJitter(**transforms.color_jitter)]
    if transforms.random_gray_scale.enable:
        print('+ random_gray_scale')
        transform_tr += [RandomGrayscale(p=transforms.random_gray_scale.p)]
    if transforms.random_perspective.enable:
        print('+ random_perspective')
        transform_tr += [RandomPerspective(p=transforms.random_perspective.p,
                                           distortion_scale=transforms.random_perspective.distortion_scale)]
    if transforms.random_rotate.enable:
        print('+ random_rotate')
        transform_tr += [RandomRotate(**transforms.random_rotate)]
    if transforms.random_erase.enable and not transforms.random_erase.norm_image:
        print('+ random erase')
        transform_tr += [RandomErasing(**transforms.random_erase)]
    print('+ to torch tensor of range [0, 1]')
    transform_tr += [PairToTensor()]
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [PairNormalize(mean=norm_mean, std=norm_std)]
    if transforms.random_erase.enable and transforms.random_erase.norm_image:
        print('+ random erase')
        transform_tr += [RandomErasing(**transforms.random_erase)]
    transform_tr = Compose(transform_tr)

    transform_te = build_test_transform(height, width, norm_mean, norm_std, apply_masks_to_test)

    return transform_tr, transform_te


def build_test_transform(height, width, norm_mean=(0.485, 0.456, 0.406),
                         norm_std=(0.229, 0.224, 0.225), apply_masks_to_test=False, **kwargs):
    print('Building test transforms ...')
    transform_te = []
    if apply_masks_to_test:
        print('+ background zeroing')
        transform_te.append(DisableBackground())
    print('+ resize to {}x{}'.format(height, width))
    transform_te.append(PairResize((height, width)))
    print('+ to torch tensor of range [0, 1]')
    transform_te.append(PairToTensor())
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_te.append(PairNormalize(mean=norm_mean, std=norm_std))
    transform_te = Compose(transform_te)

    return transform_te


def build_inference_transform(height, width, norm_mean=(0.485, 0.456, 0.406),
                              norm_std=(0.229, 0.224, 0.225), **kwargs):
    print('Building inference transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std),
    ])

    return transform_te

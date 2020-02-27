from __future__ import division, print_function, absolute_import

import math
import random
from collections import deque

import cv2
import numpy as np
import torch
from torchvision.transforms import *
from torchvision.transforms import functional as F
from PIL import Image


class Random2DTranslation(object):
    """Randomly translates the input image with a probability.

    Specifically, given a predefined shape (height, width), the input is first
    resized with a factor of 1.125, leading to (height*1.125, width*1.125), then
    a random crop is performed. Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, scale=1.125, interpolation=Image.BILINEAR, **kwargs):
        self.height = height
        self.width = width
        self.p = p
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * self.scale)), int(round(self.height * self.scale))
        resized_img = img.resize((new_width, new_height), self.interpolation)

        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))

        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))

        return croped_img


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

    def __init__(self, p=0.5, sl=0.02, sh=0.4, rl=0.5, rh=2.0, fill_color=None, **kwargs):
        self.probability = p
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = rh
        self.fill_color = fill_color
        if self.fill_color is not None:
            if len(self.fill_color) == 1:
                self.fill_color = [self.fill_color] * 3

            assert len(self.fill_color) == 3

    def __call__(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            source_area = img.size[0] * img.size[1]
            target_area = random.uniform(self.sl, self.sh) * source_area
            aspect_ratio = random.uniform(self.rl, self.rh)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size[1] and h < img.size[0]:
                x1 = random.randint(0, img.size[0] - h)
                y1 = random.randint(0, img.size[1] - w)

                img = np.array(img)

                fill_color = self.fill_color if self.fill_color is not None else [random.randint(0, 255)] * 3
                img[x1:x1 + h, y1:y1 + w, 0] = fill_color[0]
                img[x1:x1 + h, y1:y1 + w, 1] = fill_color[1]
                img[x1:x1 + h, y1:y1 + w, 2] = fill_color[2]

                return Image.fromarray(img)

        return img


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

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


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

    def __call__(self, img):
        W, H = img.size # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img


class RandomColorJitter(ColorJitter):
    def __init__(self, p=0.5, brightness=0.2, contrast=0.15, saturation=0, hue=0, **kwargs):
        self.p = p
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)


class RandomGrayScale(object):
    """Random grayscale
    """

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return img
        return F.to_grayscale(img, num_output_channels=3)


class RandomPadding(object):
    """Random padding
    """

    def __init__(self, p=0.5, padding=(0, 10), **kwargs):
        self.p = p
        self.padding_limits = padding

    def __call__(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return img
        rnd_padding = [random.randint(self.padding_limits[0], self.padding_limits[1]) for _ in range(4)]
        rnd_fill = random.randint(0, 255)
        return F.pad(img, tuple(rnd_padding), fill=rnd_fill, padding_mode='constant')


class RandomRotate(object):
    """Random rotate
    """

    def __init__(self, p=0.5, angle=(-5, 5), **kwargs):
        self.p = p
        self.angle = angle

    def __call__(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return img
        rnd_angle = random.randint(self.angle[0], self.angle[1])
        return F.rotate(img, rnd_angle, resample=False, expand=False, center=None)


class RandomGrid(object):
    """Random grid
    """

    def __init__(self, p=0.5, color=-1, grid_size=(24, 64), thickness=(1, 1), angle=(0, 180), **kwargs):
        self.p = p
        self.color = color
        self.grid_size = grid_size
        self.thickness = thickness
        self.angle = angle

    def __call__(self, img, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return img
        if self.color == (-1, -1, -1):  # Random color
            color = tuple([random.randint(0, 256) for _ in range(3)])
        else:
            color = self.color
        grid_size = random.randint(*self.grid_size)
        thickness = random.randint(*self.thickness)
        angle = random.randint(*self.angle)
        return self.draw_grid(img, grid_size, color, thickness, angle)

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

    def __call__(self, image):
        if random.uniform(0, 1) > self.p:
            return image
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
        return Image.fromarray(img)


def build_transforms(height, width, transforms=None, norm_mean=(0.485, 0.456, 0.406),
                     norm_std=(0.229, 0.224, 0.225), **kwargs):
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
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []
    if transforms.random_grid.enable:
        print('+ random_grid')
        transform_tr += [RandomGrid(**transforms.random_grid)]
    if transforms.random_figures.enable:
        print('+ random_figures')
        transform_tr += [RandomFigures(**transforms.random_figures)]
    if transforms.random_padding.enable:
        print('+ random_padding')
        transform_tr += [RandomPadding(**transforms.random_padding)]
    transform_tr += [Resize((height, width))]
    print('+ resize to {}x{}'.format(height, width))
    if transforms.random_flip.enable:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip(p=transforms.random_flip.p)]
    if transforms.random_crop.enable:
        print('+ random crop (enlarge to {}x{} and crop {}x{})'
              .format(int(round(height*1.125)), int(round(width*1.125)),
                      height, width, transforms.random_crop.p))
        transform_tr += [Random2DTranslation(height, width, **transforms.random_crop)]
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
    if transforms.random_erase.enable:
        print('+ random erase')
        transform_tr += [RandomErasing(**transforms.random_erase)]
    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]
    transform_tr = Compose(transform_tr)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te

import random
import cv2
import numpy as np
import torch
from ..utils.label_tools import label_vflip, label_hflip
import albumentations
import albumentations.augmentations.transforms as T


class BrailleImagePreprocessor:
    """
    Preprocess image and it's annotation
    """

    def __init__(self, params, mode):
        assert mode in {"train", "debug", "inference"}
        self.params = params
        self.albumentations = self._common_aug(mode, params)

    def preprocess_and_augment(self, img, rects=[]):
        aug_img = self.random_resize_and_stretch(
            img,
            new_width_range=self.params.augmentation.img_width_range,
            stretch_limit=self.params.augmentation.stretch_limit,
        )
        aug_res = self.albumentations(image=aug_img, bboxes=rects)
        aug_img = aug_res["image"]
        aug_bboxes = aug_res["bboxes"]
        aug_bboxes = [
            b
            for b in aug_bboxes
            if b[0] > 0
            and b[0] < 1
            and b[1] > 0
            and b[1] < 1
            and b[2] > 0
            and b[2] < 1
            and b[3] > 0
            and b[3] < 1
        ]
        if not self.params.data.get("get_points", False):
            for t in aug_res["replay"]["transforms"]:
                if t["__class_fullname__"].endswith(".VerticalFlip") and t["applied"]:
                    aug_bboxes = [self._rect_vflip(b) for b in aug_bboxes]
                if t["__class_fullname__"].endswith(".HorizontalFlip") and t["applied"]:
                    aug_bboxes = [self._rect_hflip(b) for b in aug_bboxes]
        return aug_img, aug_bboxes

    def random_resize_and_stretch(self, img, new_width_range, stretch_limit=0):
        new_width_range = T.to_tuple(new_width_range)
        stretch_limit = T.to_tuple(stretch_limit, bias=1)
        new_sz = int(random.uniform(new_width_range[0], new_width_range[1]))
        stretch = random.uniform(stretch_limit[0], stretch_limit[1])

        img_max_sz = img.shape[1]
        new_width = int(img.shape[1] * new_sz / img_max_sz)
        new_width = ((new_width + 31) // 32) * 32
        new_height = int(img.shape[0] * stretch * new_sz / img_max_sz)
        new_height = ((new_height + 31) // 32) * 32
        return self._resize(
            img, height=new_height, width=new_width, interpolation=cv2.INTER_LINEAR
        )

    def to_normalized_tensor(self, img, device="cpu"):
        """
        returns image converted to FloatTensor and normalized
        """
        assert img.ndim == 3
        ten_img = torch.from_numpy(img.transpose((2, 0, 1))).to(device).float()
        means = ten_img.view(3, -1).mean(dim=1)
        std = torch.max(
            ten_img.view(3, -1).std(dim=1),
            torch.tensor(self.params.data.get("max_std", 0) * 255).to(ten_img),
        )

        ten_img = (ten_img - means.view(-1, 1, 1)) / (3 * std.view(-1, 1, 1))

        ten_img = ten_img.mean(dim=0).expand(3, -1, -1)
        return ten_img

    @staticmethod
    def unify_shape(img):
        if len(img.shape) == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def _rect_vflip(self, b):
        """
        Flips symbol box converting label
        :param b: tuple (left, top, right, bottom, label)
        :return: converted tuple (left, top, right, bottom, label)
        """
        return b[:4] + (label_vflip(b[4]),)

    def _rect_hflip(self, b):
        """
        Flips symbol box converting label
        :param b: tuple (left, top, right, bottom, label)
        :return: converted tuple (left, top, right, bottom, label)
        """
        return b[:4] + (label_hflip(b[4]),)

    def _common_aug(self, mode, params):
        """
        :param mode: 'train', 'test', 'inference'
        :param params:
        """

        augs_list = []
        assert mode in {"train", "debug", "inference"}
        if mode == "train":
            augs_list.append(
                albumentations.PadIfNeeded(
                    min_height=params.data.net_hw[0],
                    min_width=params.data.net_hw[1],
                    border_mode=cv2.BORDER_REPLICATE,
                    always_apply=True,
                )
            )
            augs_list.append(
                albumentations.RandomCrop(
                    height=params.data.net_hw[0],
                    width=params.data.net_hw[1],
                    always_apply=True,
                )
            )
            if params.augmentation.rotate_limit:
                augs_list.append(
                    T.Rotate(
                        limit=params.augmentation.rotate_limit,
                        border_mode=cv2.BORDER_CONSTANT,
                        always_apply=True,
                    )
                )

        elif mode == "debug":
            augs_list.append(
                albumentations.CenterCrop(
                    height=params.data.net_hw[0],
                    width=params.data.net_hw[1],
                    always_apply=True,
                )
            )
        if mode != "inference":
            if params.augmentation.get("blur_limit", 4):
                augs_list.append(
                    T.Blur(blur_limit=params.augmentation.get("blur_limit", 4))
                )
            if params.augmentation.get("RandomBrightnessContrast", True):
                augs_list.append(T.RandomBrightnessContrast())
            # augs_list.append(T.MotionBlur())
            if params.augmentation.get("JpegCompression", True):
                augs_list.append(T.JpegCompression(quality_lower=30, quality_upper=100))
            # augs_list.append(T.VerticalFlip())
            if params.augmentation.get("HorizontalFlip", True):
                augs_list.append(T.HorizontalFlip())

        return albumentations.ReplayCompose(
            augs_list,
            p=1.0,
            bbox_params={"format": "albumentations", "min_visibility": 0.5},
        )

    def _resize(self, img, height, width, interpolation=cv2.INTER_LINEAR):
        num_channels = img.shape[2] if len(img.shape) == 3 else 1
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                chunk = img[:, :, index : index + 4]
                chunk = cv2.resize(
                    chunk, dsize=(width, height), interpolation=interpolation
                )
                chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = cv2.resize(img, dsize=(width, height), interpolation=interpolation)
        return img

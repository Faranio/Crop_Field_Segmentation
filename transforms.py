import torch

from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import Tuple, Dict, Optional


class Compose(object):
    """
    Compose several transformations.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Perform random horizontal flip of an image with corresponding masks changes.
    """
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> \
            Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)

            if target is not None:
                width, _ = F._get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)

        return image, target


class ToTensor(nn.Module):
    """
    Convert the image into a PyTorch tensor.
    """
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> \
            Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target


class RandomPhotometricDistort(nn.Module):
    """
    Apply random photometric distortions on an image.
    """
    def __init__(self, contrast: Tuple[float] = (0.5, 1.5), saturation: Tuple[float] = (0.5, 1.5),
                 hue: Tuple[float] = (-0.05, 0.05), brightness: Tuple[float] = (0.875, 1.125), p: float = 0.5):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> \
            Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('image should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5

        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels = F._get_image_num_channels(image)
            permutation = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.to_tensor(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        return image, target


def get_train_transform():
    """
    Training data transformations.
    """
    return Compose([RandomPhotometricDistort(), ToTensor()])


def get_test_transform():
    """
    Validation data transformations.
    """
    return Compose([ToTensor()])

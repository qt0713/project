import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, utils

AUG_TYPES = [
    "original",
    "sobel",
    "random_vflip",
    "rotate15",
    "laplacian",
    "rotate180",
    "solarize",
    "autocontrast",
    "equalize",
    "colorjitter",
    "randomaffine",
    "randomperspective",
    "randomerasing",
    "randomresizedcrop",
    "morph_dilate",
    "morph_erode",
    "rgb2hsv",
    "mixup",
    "cutmix",
    "hist_eq",
    "gaussian_blur",
    "mean_blur",
    "median_blur",
    "sharpen",
]


def _resize_and_center(img: Image.Image, img_size: int, crop_size: int) -> Image.Image:
    img = img.resize((img_size, img_size))
    left = (img_size - crop_size) // 2
    top = (img_size - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return img.crop((left, top, right, bottom))


def get_aug_transform(aug_type: str, img_size: int, crop_size: int):
    def base(img: Image.Image) -> Image.Image:
        return _resize_and_center(img, img_size, crop_size)

    if aug_type == "original":
        return lambda img: base(img)
    if aug_type == "sobel":
        def sobel(img):
            import cv2

            arr = np.array(base(img))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_val = cv2.magnitude(sobelx, sobely)
            max_val = sobel_val.max() if sobel_val.max() > 0 else 1.0
            sobel_val = np.uint8(np.clip(sobel_val / max_val * 255, 0, 255))
            return Image.fromarray(sobel_val).convert("RGB")

        return sobel
    if aug_type == "laplacian":
        def laplacian(img):
            import cv2

            arr = np.array(base(img))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            max_val = lap.max() if lap.max() > 0 else 1.0
            lap = np.uint8(np.clip((lap / max_val) * 255, 0, 255))
            return Image.fromarray(lap).convert("RGB")

        return laplacian
    if aug_type == "mean_blur":
        def mean_blur(img):
            import cv2

            arr = np.array(base(img))
            return Image.fromarray(cv2.blur(arr, (5, 5)))

        return mean_blur
    if aug_type == "median_blur":
        def median_blur(img):
            import cv2

            arr = np.array(base(img))
            return Image.fromarray(cv2.medianBlur(arr, 5))

        return median_blur
    if aug_type == "sharpen":
        def sharpen(img):
            import cv2

            arr = np.array(base(img))
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return Image.fromarray(cv2.filter2D(arr, -1, kernel))

        return sharpen
    if aug_type == "random_vflip":
        return lambda img: transforms.RandomVerticalFlip(p=1.0)(base(img)) if random.random() < 0.5 else base(img)
    if aug_type == "rotate15":
        return lambda img: transforms.functional.rotate(base(img), 15)
    if aug_type == "rotate180":
        return lambda img: transforms.functional.rotate(base(img), 180)
    if aug_type == "solarize":
        return lambda img: transforms.functional.solarize(base(img), threshold=128)
    if aug_type == "autocontrast":
        return lambda img: transforms.functional.autocontrast(base(img))
    if aug_type == "equalize":
        return lambda img: transforms.functional.equalize(base(img))
    if aug_type == "colorjitter":
        return lambda img: transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)(base(img))
    if aug_type == "randomaffine":
        return lambda img: transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)(base(img))
    if aug_type == "randomperspective":
        return lambda img: transforms.RandomPerspective(distortion_scale=0.5, p=1.0)(base(img))
    if aug_type == "randomerasing":
        def apply_erasing(img):
            tensor_img = transforms.ToTensor()(base(img))
            erased = transforms.RandomErasing(p=1.0, scale=(0.1, 0.2), ratio=(0.3, 3.3))(tensor_img)
            return transforms.ToPILImage()(erased)

        return apply_erasing
    if aug_type == "randomresizedcrop":
        return lambda img: transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0), ratio=(0.75, 1.33))(img.resize((img_size, img_size)))
    if aug_type == "morph_dilate":
        def morph_dilate(img):
            import cv2

            arr = np.array(base(img))
            kernel = np.ones((5, 5), np.uint8)
            return Image.fromarray(cv2.dilate(arr, kernel, iterations=1))

        return morph_dilate
    if aug_type == "morph_erode":
        def morph_erode(img):
            import cv2

            arr = np.array(base(img))
            kernel = np.ones((5, 5), np.uint8)
            return Image.fromarray(cv2.erode(arr, kernel, iterations=1))

        return morph_erode
    if aug_type == "rgb2hsv":
        def rgb2hsv(img):
            import cv2

            arr = np.array(base(img))
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            return Image.fromarray(hsv, mode="RGB")

        return rgb2hsv
    if aug_type == "hist_eq":
        def hist_eq(img):
            import cv2

            arr = np.array(base(img))
            img_yuv = cv2.cvtColor(arr, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))

        return hist_eq
    if aug_type == "gaussian_blur":
        def gaussian_blur(img):
            import cv2

            arr = np.array(base(img))
            return Image.fromarray(cv2.GaussianBlur(arr, (7, 7), 0))

        return gaussian_blur

    raise ValueError(f"Unknown aug_type: {aug_type}")


class RandomAugImageFolder(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        aug_types: Sequence[str],
        num_classes: int,
        img_size: int,
        crop_size: int,
    ):
        self.samples = list(samples)
        self.aug_types = list(aug_types)
        self.num_classes = num_classes
        self.img_size = img_size
        self.crop_size = crop_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def _one_hot(self, label: int) -> np.ndarray:
        y = np.zeros(self.num_classes, dtype=np.float32)
        y[label] = 1.0
        return y

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        aug_type = random.choice(self.aug_types)
        img1 = Image.open(img_path).convert("RGB")

        if aug_type == "mixup":
            lam = np.random.beta(0.4, 0.4)
            idx2 = random.randint(0, len(self.samples) - 1)
            img2_path, label2 = self.samples[idx2]
            img2 = Image.open(img2_path).convert("RGB")
            x1 = np.array(get_aug_transform("original", self.img_size, self.crop_size)(img1)).astype(np.float32)
            x2 = np.array(get_aug_transform("original", self.img_size, self.crop_size)(img2)).astype(np.float32)
            mixed = (lam * x1 + (1 - lam) * x2).astype(np.uint8)
            x = transforms.ToTensor()(Image.fromarray(mixed))
            y = np.zeros(self.num_classes, dtype=np.float32)
            y[label] = lam
            y[label2] = 1 - lam
            return x, y

        if aug_type == "cutmix":
            lam = np.random.beta(1.0, 1.0)
            idx2 = random.randint(0, len(self.samples) - 1)
            img2_path, label2 = self.samples[idx2]
            img2 = Image.open(img2_path).convert("RGB")
            x1 = np.array(get_aug_transform("original", self.img_size, self.crop_size)(img1))
            x2 = np.array(get_aug_transform("original", self.img_size, self.crop_size)(img2))
            h, w, _ = x1.shape
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            x1_min = np.clip(cx - cut_w // 2, 0, w)
            y1_min = np.clip(cy - cut_h // 2, 0, h)
            x1_max = np.clip(cx + cut_w // 2, 0, w)
            y1_max = np.clip(cy + cut_h // 2, 0, h)
            x1[y1_min:y1_max, x1_min:x1_max, :] = x2[y1_min:y1_max, x1_min:x1_max, :]
            area = (x1_max - x1_min) * (y1_max - y1_min)
            lam_area = 1 - area / (h * w)
            x = transforms.ToTensor()(Image.fromarray(x1))
            y = np.zeros(self.num_classes, dtype=np.float32)
            y[label] = lam_area
            y[label2] = 1 - lam_area
            return x, y

        img = get_aug_transform(aug_type, self.img_size, self.crop_size)(img1)
        x = transforms.ToTensor()(img)
        x = self.normalize(x)
        return x, self._one_hot(label)


def load_base_dataset(data_path: str):
    if not data_path or not os.path.isdir(data_path):
        raise RuntimeError("DATA_PATH does not exist. Set data_path to a folder containing class subfolders.")
    base_dataset = datasets.ImageFolder(root=data_path)
    return base_dataset, base_dataset.classes


def split_dataset(aug_dataset: Dataset, seed: int = 42, val_ratio: float = 0.2, test_ratio: float = 0.1):
    dataset_size = len(aug_dataset)
    test_size = int(test_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size - test_size
    return random_split(
        aug_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int, device):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_loader, test_loader


def show_batch(images, title: str = ""):
    inp = images.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    import matplotlib.pyplot as plt

    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.axis("off")


def make_grid_from_loader(train_loader, n: int = 24):
    images, _ = next(iter(train_loader))
    return utils.make_grid(images[:n], nrow=6)

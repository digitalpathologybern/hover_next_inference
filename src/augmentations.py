import torch
import numpy as np
from torchvision.transforms.transforms import ColorJitter, RandomApply, GaussianBlur


rgb_from_hed = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32
)
hed_from_rgb = np.linalg.inv(rgb_from_hed)


def torch_rgb2hed(img: torch.Tensor, hed_t: torch.Tensor, e: float):
    """
    convert rgb torch tensor to hed torch tensor (adopted from skimage)

    Parameters
    ----------
    img : torch.Tensor
        rgb image tensor (B, C, H, W) or (C, H, W)
    hed_t : torch.Tensor
        hed transform tensor (3, 3)
    e : float
        epsilon

    Returns
    -------
    torch.Tensor
        hed image tensor (B, C, H, W) or (C, H, W)
    """
    img = img.movedim(-3, -1)

    img = torch.clamp(img, min=e)
    img = torch.log(img) / torch.log(e)
    img = torch.matmul(img, hed_t)
    return img.movedim(-1, -3)


def torch_hed2rgb(img: torch.Tensor, rgb_t: torch.Tensor, e: float):
    """
    convert rgb torch tensor to hed torch tensor (adopted from skimage)

    Parameters
    ----------
    img : torch.Tensor
        hed image tensor (B, C, H, W) or (C, H, W)
    hed_t : torch.Tensor
        hed inverse transform tensor (3, 3)
    e : float
        epsilon

    Returns
    -------
    torch.Tensor
        RGB image tensor (B, C, H, W) or (C, H, W)
    """
    e = -torch.log(e)
    img = img.movedim(-3, -1)
    img = torch.matmul(-(img * e), rgb_t)
    img = torch.exp(img)
    img = torch.clamp(img, 0, 1)
    return img.movedim(-1, -3)


class Hed2Rgb(torch.nn.Module):
    """
    Pytorch module to convert hed image tensors to rgb
    """

    def __init__(self, rank):
        super().__init__()
        self.e = torch.tensor(1e-6).to(rank)
        self.rgb_t = torch.from_numpy(rgb_from_hed).to(rank)
        self.rank = rank

    def forward(self, img):
        return torch_hed2rgb(img, self.rgb_t, self.e)


class Rgb2Hed(torch.nn.Module):
    """
    Pytorch module to convert rgb image tensors to hed
    """

    def __init__(self, rank):
        super().__init__()
        self.e = torch.tensor(1e-6).to(rank)
        self.hed_t = torch.from_numpy(hed_from_rgb).to(rank)
        self.rank = rank

    def forward(self, img):
        return torch_rgb2hed(img, self.hed_t, self.e)


class HedNormalizeTorch(torch.nn.Module):
    """
    Pytorch augmentation module to apply HED stain augmentation

    Parameters
    ----------
    sigma : float
        sigma for linear scaling of HED channels
    bias : float
        bias for additive scaling of HED channels
    """

    def __init__(self, sigma, bias, rank, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.bias = bias
        self.rank = rank
        self.rgb2hed = Rgb2Hed(rank=rank)
        self.hed2rgb = Hed2Rgb(rank=rank)

    def rng(self, val, batch_size):
        return torch.empty(batch_size, 3).uniform_(-val, val).to(self.rank)

    def color_norm_hed(self, img):
        B = img.shape[0]
        sigmas = self.rng(self.sigma, B)
        biases = self.rng(self.bias, B)
        return (img * (1 + sigmas.view(*sigmas.shape, 1, 1))) + biases.view(
            *biases.shape, 1, 1
        )

    def forward(self, img):
        if img.dim() == 3:
            img = img.view(1, *img.shape)
        hed = self.rgb2hed(img)
        hed = self.color_norm_hed(hed)
        return self.hed2rgb(hed)


class GaussianNoise(torch.nn.Module):
    """
    Pytorch augmentation module to apply gaussian noise

    Parameters
    ----------
    sigma : float
        sigma for uniform distribution to sample from
    rank : str or int or torch.device
        device to put the module to
    """

    def __init__(self, sigma, rank):
        super().__init__()
        self.sigma = sigma
        self.rank = rank

    def forward(self, img):
        noise = torch.empty(img.shape).uniform_(-self.sigma, self.sigma).to(self.rank)
        return img + noise


def color_augmentations(train, sigma=0.05, bias=0.03, s=0.2, rank=0):
    """
    Color augmentation function (in theory can set to train to have more variance
    with high test time augmentations)

    Parameters
    ----------
    train : bool
        during training, the model uses more augmentation than during inference,
        set to true for more variance in colors
    sigma: float
        parameter for hed augmentation
    bias: float
        parameter for hed augmentation
    s: float
        parameter for color jitter
    rank: int or torch.device or str
        device to use for augmentation

    Returns
    -------
    torch.nn.Sequential
        sequential augmentation module
    """
    if train:
        color_jitter = ColorJitter(
            0.8 * s, 0.0 * s, 0.8 * s, 0.2 * s
        )  # brightness, contrast, saturation, hue

        data_transforms = torch.nn.Sequential(
            RandomApply([HedNormalizeTorch(sigma, bias, rank=rank)], p=0.75),
            RandomApply([color_jitter], p=0.3),
            RandomApply([GaussianNoise(0.02, rank)], p=0.3),
            RandomApply([GaussianBlur(kernel_size=15, sigma=(0.1, 0.1))], p=0.3),
        )
    else:
        data_transforms = torch.nn.Sequential(HedNormalizeTorch(sigma, bias, rank=rank))
    return data_transforms

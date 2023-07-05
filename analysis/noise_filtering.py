import torch
import torch.nn.functional as F
from skimage.filters import rank
from skimage.morphology import disk
from torchvision.transforms.functional import normalize
import os.path as osp
import analysis
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from math import exp
import inspect
import os


def median_filter(image, kernel_size):
    # Pad the image to handle border pixels
    image = image.unsqueeze(0)
    padded_image = F.pad(image.unsqueeze(0), (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                         mode='reflect')

    # Prepare the sliding window view of the image
    unfolded = padded_image.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)

    # Reshape the sliding window view to perform median calculation
    unfolded = unfolded.contiguous().view(image.size(0), image.size(1), -1, kernel_size * kernel_size)

    # Calculate the median along the sliding window
    median_values = torch.median(unfolded, dim=-1)[0]

    return median_values


def mean_filter(image, kernel_size):
    # Apply mean filtering using a sliding window
    filtered_image = F.avg_pool2d(image.unsqueeze(0), kernel_size, stride=1, padding=kernel_size // 2)

    return filtered_image.squeeze(0)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _gaussian_blur(x, window, window_size):
    B, C, H, W = x.size()
    return F.conv2d(x, window, padding=window_size // 2, groups=C)


def gaussian_blur(img, window_size):
    window = create_window(window_size, img.shape[0])
    return _gaussian_blur(img, window, window_size)


def sobel_operator(img):
    sobel_kernel = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                                 [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
    sobel_kernel = sobel_kernel.expand((2, img.shape[1], 3, 3))
    img_edges = F.conv2d(img, sobel_kernel, padding=1)
    img_edges = img_edges.abs().sum(dim=1, keepdim=True)
    return img_edges


# Bilateral filter approximation
def bilateral_filter(img, window_size):
    # Reshape the input to (batch, channel, height, width)
    img = img.unsqueeze(0)

    # Apply Gaussian Blur
    img_blurred = gaussian_blur(img, window_size)

    # Detect Edges
    img_edges = sobel_operator(img)

    # Retain Edges, Suppress Noise
    img_filtered = img_edges + img_blurred

    return img_filtered.squeeze(0)


def wiener_filter(image, kernel, noise_power):
    # Calculate the power spectrum of the kernel
    kernel_power = torch.fft.fft2(kernel, dim=(-2, -1)).pow(2).sum(dim=(-2, -1)).unsqueeze(0).unsqueeze(0)

    # Calculate the power spectrum of the noise
    noise_power_spectrum = noise_power * torch.ones_like(kernel_power)

    # Calculate the Wiener filter
    wiener_filter = torch.conj(kernel) / (kernel_power + noise_power_spectrum)
    wiener_filter = wiener_filter.abs().unsqueeze(0).unsqueeze(0)  # added .abs() here

    # Apply the Wiener filter using convolution
    filtered_image = F.conv2d(image.unsqueeze(0), wiener_filter, padding=kernel.size(-1) // 2)

    return filtered_image.squeeze(0)


def gaussian_filter(image, sigma):
    # Create a 1D Gaussian kernel
    kernel_size = int(4 * sigma + 1)
    gauss_1D = torch.tensor(
        [torch.exp(-(torch.tensor(x) - kernel_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(kernel_size)])
    gauss_1D = gauss_1D / torch.sum(gauss_1D)

    # Create a 2D Gaussian kernel from the outer product of the 1D kernel with itself
    gauss_2D = torch.outer(gauss_1D, gauss_1D)
    kernel = gauss_2D.view(1, 1, kernel_size, kernel_size)

    # Apply the Gaussian filter using convolution
    filtered_image = F.conv2d(image, kernel, padding=(kernel_size - 1) // 2)

    return filtered_image


def anisotropic_diffusion(image, niter=10, kappa=50, gamma=0.1):
    # Convert image to torch tensor
    image_tensor = torch.tensor(image, dtype=torch.float32)

    # Normalize the image to range [0, 1]
    image_tensor = (image_tensor - torch.min(image_tensor)) / (torch.max(image_tensor) - torch.min(image_tensor))
    image_tensor.requires_grad = True

    for i in range(niter):
        # Compute gradients
        loss = image_tensor.sum()
        loss.backward(create_graph=True)

        gradients = image_tensor.grad
        if gradients is None:
            continue

        # Compute the diffusion term
        diffusion = kappa * torch.norm(gradients, p=2) * torch.div(gradients, torch.norm(gradients, p=2) + 1e-8)

        # Update the image tensor
        with torch.no_grad():
            image_tensor += gamma * diffusion

        # We need to clear gradients before next iteration
        if image_tensor.grad is not None:
            image_tensor.grad.data.zero_()

    # Clip the image tensor to range [0, 1]
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Convert the image tensor back to a numpy array
    filtered_image = image_tensor.detach().cpu()

    return filtered_image

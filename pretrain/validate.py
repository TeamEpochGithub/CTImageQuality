import os
import random
import time
import json
import torch
import torch.optim as optim
import numpy as np
import albumentations as A
from glob import glob
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import os.path as osp
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn as nn
from pretrain_models.model_efficientnet_denoise import Efficient_Swin_Denoise
from pretrain_models.model_resnet_denoise import Resnet34_Swin_Denoise
from pretrain_models.resnet34_unet import UNet34_Denoise
from pretrain_models.edcnn import EDCNN
from pretrain_models.efficientnet_unet import EfficientNet_Denoise
import tifffile
import warnings

warnings.filterwarnings('ignore')

def visualize(img1, img2):
    cap1 = "Image After Denoising"
    cap2 = "Target Image"

    # Create a new figure
    fig = plt.figure(figsize=(10, 5))

    # Add first subplot for first image
    ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    ax1.imshow(img1, cmap="gray")
    ax1.set_title(cap1)
    ax1.axis('off')  # Hide axes

    # Add second subplot for second image
    ax2 = fig.add_subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    ax2.imshow(img2, cmap="gray")
    ax2.set_title(cap2)
    ax2.axis('off')  # Hide axes

    # Display the figure with two subplots
    plt.show()

def validate_AAPM(FD_file, QD_file):
    pretrain_path = osp.dirname(__file__)
    data_path = osp.join(pretrain_path, 'pretrain_data', "AAPM")
    train_FD_path = os.path.join(data_path, "train_set", 'FD_NPY', FD_file)
    train_QD_path = os.path.join(data_path, "train_set", 'QD_NPY', QD_file)

    model_path = r"C:\Users\leo\Documents\CTImageQuality\pretrain\weights\Efficientnet_B1\pretrain_weight_denoise.pkl"
    model = EfficientNet_Denoise(mode="b1")
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model = model.cuda()
    input_img, target_img = np.float32(np.load(train_FD_path)), np.float32(np.load(train_QD_path))
    input_img = torch.tensor(input_img).unsqueeze(0).unsqueeze(1).cuda()
    noise_map = model(input_img)
    clear_img = input_img-noise_map
    clear_img = clear_img.cpu().detach().numpy().squeeze()
    visualize(clear_img, target_img)

def validate_IQA():
    path = r"C:\Users\leo\Documents\CTImageQuality\LDCTIQAG2023_train\image"
    save_path = r"C:\Users\leo\Documents\CTImageQuality\pretrain\denoise_imgs_EDCNN"
    for file in os.listdir(path):
        if "tif" in file:
            path_file = osp.join(path, file)
            with tifffile.TiffFile(path_file) as tif:
                image = tif.pages[0].asarray()
                img = np.array(image)
        target_img = np.float32(img)
        model_path = r"C:\Users\leo\Documents\CTImageQuality\pretrain\weights\ED_CNN\pretrain_weight_denoise.pkl"
        model = EDCNN()
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        model = model.cuda()
        input_img = torch.tensor(target_img).unsqueeze(0).unsqueeze(1).cuda()
        clear_img = model(input_img)
        clear_img = clear_img.cpu().detach().numpy().squeeze()
        plt.imshow(clear_img, cmap='gray')
        plt.axis('off')
        plt.savefig(osp.join(save_path, file[:-3]+"png"))


    # visualize(clear_img, target_img)


# FD_file = "L067_FD_1_1.0030.npy"
# QD_file = "L067_QD_1_1.0030.npy"
#
# validate_AAPM(FD_file, QD_file)

# path = r"C:\Users\leo\Documents\CTImageQuality\LDCTIQAG2023_train\image\0006.tif"
validate_IQA()


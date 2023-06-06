from datasets import CT_Dataset
from evaluate import create_datalists

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import tifffile as tiff


def plot_label_distribution():
    imgs_list, label_list = create_datalists()
    plt.hist(label_list, bins=21, alpha=0.7, rwidth=0.85)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values')
    plt.savefig(osp.join('plots', 'label_distribution_plot.png'))
    plt.show()


if __name__ == '__main__':
    imgs_list, label_list = create_datalists()

    dataset = CT_Dataset(imgs_list, label_list)

    for img, label in dataset:
        array = img.numpy()

        label = round(label.item(), 1)

        dirname = osp.join('plots', 'imgs_labels', f"{str(label).replace('.', '-')}.tif")
        i = 0
        while osp.exists(dirname):
            dirname = osp.join('plots', 'imgs_labels', f"{str(label).replace('.', '-')}-{i}.tif")
            i += 1

        tiff.imwrite(dirname, array)

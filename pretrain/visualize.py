import numpy as np
import matplotlib.pyplot as plt

def vis(path):
    img = np.load(path)
    plt.imshow(img, cmap='gray')
    plt.show()

path = r"C:\Users\leo\Documents\CTImageQuality\pretrain\output_imgs\test_target_5.npy"
vis(path)

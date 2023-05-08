import tifffile
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

print(torch.cuda.is_available())
path = r"C:\Users\leo\Documents\CTImageQuality\LDCTIQAG2023_train\image\0333.tif"
# Open the TIF file
with tifffile.TiffFile(path) as tif:
    # Read the first page of the TIF file
    image = tif.pages[0].asarray()
    img = Image.fromarray(image)
    img = img.resize((256, 256), Image.ANTIALIAS)
    img = np.array(img)
    # Print the image shape
    print(img.shape)
    print(np.min(img))
    print(np.max(img))
    plt.imshow(img)
    plt.show()

import os
from PIL import Image
import numpy as np
import tifffile

# # Define the input and output directories
# input_dir = r'C:\Users\leo\Documents\CTImageQuality\LDCTIQAG2023_train\image'
# output_dir = r'C:\Users\leo\Documents\CTImageQuality\LDCTIQAG2023_train\imgs'
#
# # Traverse the input directory
# for root, dirs, files in os.walk(input_dir):
#     for file in files:
#         # Check if the file is a TIF file
#         if file.endswith('.tif'):
#             # Open the TIF file
#             with tifffile.TiffFile(os.path.join(root, file)) as tif:
#                 # Define the output file path
#                 image = tif.pages[0].asarray()
#                 output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.png')
#                 image = (image * 255).astype(np.uint8)
#                 img = Image.fromarray(image)
#
#                 # Save the image as a PNG file
#                 img.save(output_file)

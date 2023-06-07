import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = apply_voi_lut(dicom.pixel_array, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

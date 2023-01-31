"""
MS lesion segmentation using 3D kernel based on MONAI.
"""

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

print_config()


def match_images_and_labels(images, labels):
    """
    Assumes BIDS format.
    :param images:
    :param labels:
    :return:
    """
    images_match = []
    labels_match = []
    # Loop across images
    for image in images:
        # Fetch subject name
        subject = image.split(os.path.sep)[-1].split("_")[0]
        # Find equivalent in labels
        # TODO: check if label has 2 entries
        label = [j for i, j in enumerate(labels) if subject in j]
        if label:
            images_match.append(image)
            labels_match.append(label[0])
    return images_match, labels_match


# Setup data directory
data_dir = os.environ.get("PATH_DATA_BASEL_MP2RAGE")
# TODO: remove hard code
data_dir = "/Users/julien/data.neuro/basel-mp2rage"
print(f"Path to data: {data_dir}")

# TODO: check dataset integrity
# data: data.neuro.polymtl.ca:/datasets/basel-mp2rage
# commit: ffe427d4d1f62832e5f3567c8ce814eeff9b9764

# Set MSD Spleen dataset path
train_images = sorted(glob.glob(os.path.join(data_dir, "**", "*_UNIT1.nii.gz"), recursive=True))
train_labels = \
    sorted(glob.glob(os.path.join(data_dir, "derivatives", "**", "*_lesion-manualNeuroPoly.nii.gz"), recursive=True))
train_images_match, train_labels_match = match_images_and_labels(train_images, train_labels)
data_dicts = [{"image": image_name, "label": label_name}
              for image_name, label_name in zip(train_images_match, train_labels_match)]
# train_files, val_files = data_dicts[:-9], data_dicts[-9:]


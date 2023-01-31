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

# Split train/val
# TODO: add randomization in the split
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

# Set deterministic training for reproducibility
set_determinism(seed=0)

# Setup transforms for training and validation
# TODO: find a_min/a_max automatically based on histogram distribution.
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=4095,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=4095,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ]
)

# Check transforms in DataLoader
check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[80, :, :], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[80, :, :])
plt.show()

# Define CacheDataset and DataLoader for training and validation
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
# train_ds = Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# Create Model, Loss, Optimizer
# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")


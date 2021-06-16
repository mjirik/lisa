# %%


# %%

from __future__ import print_function
from loguru import logger
import io3d
import io3d.datasets
import sed3
import numpy as np
import matplotlib.pyplot as plt

logger.enable("io3d")
logger.disable("io3d")
import matplotlib.pyplot as plt
from pathlib import Path
import bodynavigation
import exsu

import sys
import os

import tensorflow as tf
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
from skimage.exposure import rescale_intensity
from skimage import io
# from data import load_train_data, load_test_data
from sklearn.utils import class_weight
from typing import Optional
from numbers import Number
import datetime

from unet_keras_tools import window, create_train_data, load_test_data, load_train_data, UNetTrainer, save_segmentations

# %%

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print(sys.version_info)
print(sys.executable)
print(os.getcwd())

# %%

experiment_label = "rightkidney_slides_bodynavigation_1"
organ_label = "rightkidney"
show=False

# %%
dtstr = datetime.datetime.now().strftime("%Y%m%d_%H%M")
rdir = f"report_{experiment_label}_{dtstr}"
report = exsu.Report(outputdir=rdir, show=show, additional_spreadsheet_fn="data.xlsx")
reporti=0
# %%

datap1 = io3d.datasets.read_dataset("3Dircadb1", "data3d", 1)
# datap1 = io3d.datasets.read_dataset("sliver07", "data3d", 1)
data3d = datap1["data3d"]
sed3.show_slices(data3d, shape=[2, 3], show=show)
report.savefig(f"{reporti:03d}.png")
reporti += 1
# %%

datap_mask = io3d.datasets.read_dataset("3Dircadb1", organ_label, 1)
data3d_mask = datap_mask["data3d"]
sed3.show_slices(data3d_mask, shape=[2, 3], show=show)
report.savefig(f"{reporti:03d}.png")
reporti += 1
# plt.figure()

# %% md

## windowing

# %%





data3dw = window(data3d, center=40, width=400)
# fix, axs = plt.subplots(1,2)
# axs[]

# plt.imshow(data3d[30, :, :], cmap='gray')
# plt.colorbar()
#
# plt.figure()
#
# plt.imshow(data3dw[30, :, :], cmap='gray')
# plt.colorbar()
#
# %% md

## bodynavigation

# %%

bn = bodynavigation.body_navigation.BodyNavigation(datap1["data3d"], voxelsize_mm=datap1["voxelsize_mm"])
dst = bn.dist_to_sagittal()
plt.imshow(data3d[30, :, :], cmap="gray")
plt.contour(dst[30, :, :] > 0)
report.savefig(f"{reporti:03d}.png")
reporti += 1


# %%



# %%

a = np.asarray([np.stack([data3d[0, :, :], data3d[1, :, :]], axis=2)])
a.shape


# %%



# %%

if True:
    create_train_data(
        "train",
        datasets={
#             "3Dircadb1": {"start":1, "stop":3},
            "3Dircadb1": {"start":1, "stop":16},
#             "sliver07": {"start":1, "stop":16}
        },
        experiment_label=experiment_label,
        organ_label=organ_label

    )
    create_train_data(
        "test",
        datasets={
            "3Dircadb1": {"start":16, "stop":21},
#             "sliver07": {"start":16, "stop":20}
        },
        experiment_label = experiment_label,
        organ_label = organ_label

    )

    # %%


# %%


# %%


# %% md

# CNN

# %% md

# conda install -c conda-forge keras-applications

# %%


# %%



# %%

data_oh = tf.one_hot(datap_mask['data3d'], 2)
print(data_oh.shape)
# print(data_oh)
sed3.show_slices(data_oh.numpy()[:, :, :, 0].squeeze(), shape=[2, 3], show=show)
report.savefig(f"{reporti:03d}.png")
reporti += 1



# %%


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = int(512 / 2)
img_cols = int(512 / 2)


# We divide here the number of rows and columns by two because we undersample our data (We take one pixel over two)


# %%

# wbc = weighted_binary_crossentropy(0.5, 0.5)
# u = wbc(np.array([1,1,0,1], dtype=np.float32).reshape(4,1,1,1), np.array([1,1,0,1], dtype=np.float32).reshape(4,1,1,1))


# %%



#     return imgs_train, imgs_mask_train

# %%

f"__{10:04}.png"

# %%

# weights_dct

# %%

nb_channels=2
unt = UNetTrainer(nb_channels, img_rows, img_cols, experiment_label)

history = unt.train_and_predict(epochs=1)
# history = train_and_predict(continue_training=True, epochs=10)

# %%

## Recalculate predictions for train dataset
# imgs_train, imgs_mask_train = load_test_data()

# imgs_train = preprocess(imgs_train)
# imgs_mask_train = preprocess(imgs_mask_train)

# imgs_train = imgs_train.astype('float32')
# mean = np.mean(imgs_train)  # mean for data centering
# std = np.std(imgs_train)  # std for data normalization
unt.predict_test_data(history)

# %% md

# Try one image

# %%

imgs_train, imgs_mask_train = load_test_data(experiment_label)

imgs_train = unt.preprocess(imgs_train)
imgs_mask_train = unt.preprocess(imgs_mask_train)

imgs_train = imgs_train.astype('float32')
# mean = np.mean(imgs_train)  # mean for data centering
# std = np.std(imgs_train)  # std for data normalization

# imgs_train -= mean
# imgs_train /= std
# Normalization of the train set

imgs_mask_train = imgs_mask_train.astype('float32')

print(f"Number of frames={imgs_train.shape[0]}")

# %%
# ---------------------------------------------------------

unt = UNetTrainer(nb_channels, img_rows, img_cols, experiment_label)
model = unt.get_unet()

model.load_weights(f'weights_{experiment_label}.h5')

logger.debug('-' * 30)
logger.debug('Predicting masks on test data...')
logger.debug('-' * 30)
imgs_mask_train_pred = model.predict(imgs_train, verbose=1)

# %%

import scipy.stats

logger.debug(scipy.stats.describe(imgs_mask_train_pred.flatten()))

# %%

i = 86

# %%

plt.imshow(imgs_train[i, :, :, 0], cmap='gray')
plt.colorbar()
report.savefig(f"img_{i}.png")

# %%

plt.imshow(imgs_mask_train_pred[i, :, :], cmap='gray')
plt.colorbar()
report.savefig(f"mask_pred_{i}.png")

# %%

plt.imshow(imgs_mask_train[i, :, :], cmap='gray')
plt.colorbar()
report.savefig(f"mask_true_{i}.png")

# %%

logger.debug(tf.keras.losses.binary_crossentropy(imgs_mask_train[i, :, :].flatten(), imgs_mask_train_pred[i, :, :].flatten()))

# %%

logger.debug(tf.keras.losses.binary_crossentropy(imgs_mask_train.flatten(), imgs_mask_train_pred.flatten()))

# %%

save_segmentations(imgs_train[:, :, :, 0, 0], imgs_mask_train_pred[..., 0], f"preds_test/{experiment_label}")

# %%

y_train = (imgs_mask_train > 0).astype(np.float32)
weights = class_weight.compute_class_weight('balanced', np.unique(y_train.flatten()), y_train.flatten())
# y_train.shape
# imgs_train.shape
# y_train.dtype
# print(np.unique(imgs_mask_train > 0))

# plt.imshow(imgs_mask_train[150,:,:] > 0, interpolation='None')
logger.debug(weights)

# %%



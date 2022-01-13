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

def window(
        data3d: np.ndarray,
        vmin: Optional[Number] = None,
        vmax: Optional[Number] = None,
        center: Optional[Number] = None,
        width: Optional[Number] = None,
        vmin_out: Optional[Number] = 0,
        vmax_out: Optional[Number] = 255,
        dtype=np.uint8):
    """
    Rescale input ndarray and trim the outlayers.

    :param data3d: ndarray with numbers
    :param vmin: minimal input value. Skipped if center and width is given.
    :param vmax: maximal input value. Skipped if center and width is given.
    :param center: Window center
    :param width: Window width
    :param vmin_out: Output mapping minimal value
    :param vmax_out: Output mapping maximal value
    :param dtype: Output dtype
    :return:
    """
    if width and center:
        vmin = center - (width / 2.)
        vmax = center + (width / 2.)

    #     logger.debug(f"vmin={vmin}, vmax={vmax}")
    k = float(vmax_out - vmin_out) / (vmax - vmin)
    q = vmax_out - k * vmax
    #     logger.debug(f"k={k}, q={q}")
    data3d_out = data3d * k + q

    data3d_out[data3d_out > vmax_out] = vmax_out
    data3d_out[data3d_out < vmin_out] = vmin_out

    return data3d_out.astype(dtype)

import h5py
import tensorflow as tf

class generator:
    def __init__(self, label, organ_label, is_mask=False):
        self.label = label
        self.organ_label = organ_label
        self.is_mask=is_mask

    def __call__(self):
        fnimgs = Path(f'mask_{self.label}_{self.organ_label}') if self.is_mask else Path(f'img_{self.label}')

        for indx in range(len(fnimgs.glob("*.npy"))):
            fnimg = fnimgs / f"{indx:06d}.npy"
            img = np.load(fnimg)
            yield img

        # with h5py.File(self.file, 'r') as hf:
        #     for im in hf["train_img"]:
        #         imgs_train = np.load(f'imgs_train_{experiment_label}.npy')
        #         yield im


def load_train_data(experiment_label):
    imgs_train = np.load(f'imgs_train_{experiment_label}.npy')
    masks_train = np.load(f'masks_train_{experiment_label}.npy')
    return imgs_train, masks_train


def load_test_data(experiment_label):
    imgs_test = np.load(f'imgs_test_{experiment_label}.npy')
    masks_test = np.load(f'masks_test_{experiment_label}.npy')
    return imgs_test, masks_test

def get_dataset_loaders(label, organ_label):
    imgs = tf.data.Dataset.from_generator(
        generator(label, organ_label, is_mask=False),
        tf.uint8,
        tf.TensorShape([512, 512, 3]))

    masks = tf.data.Dataset.from_generator(
        generator(label, organ_label, is_mask=True),
        tf.uint8,
        tf.TensorShape([512, 512, 3]))
    return imgs, masks

def create_train_data(label="train", datasets=None, dataset_label="", organ_label="rightkidney", skip_if_exists=True):
    # fnimgs = f'imgs_{label}_{dataset_label}.npy'
    # fnmasks =f'masks_{label}_{dataset_label}.npy'

    fnimgs = Path(f'img_{label}_{dataset_label}')
    fnmasks =Path(f'mask_{label}_{dataset_label}_{organ_label}')
    fnpattern = "{dataset}_{i:02d}_{k:05d}.npy"

    p_imgs = fnimgs
    p_masks =fnmasks

    # if p_imgs.exists() and p_imgs.is_dir() and p_masks.exists() and p_masks.is_dir() and skip_if_exists:
    #     logger.info("Files exists. Skipping creation and loading instead.")
    #     # imgs_train = np.load(fnimgs)
    #     # masks_train = np.load(fnmasks)
    if True:
        # imgs_train = []
        # masks_train = []
        if not datasets:
            datasets = {
                "3Dircadb1": {"start": 1, "stop": 2},
                #             "sliver07": {"start":0, "stop":0}
            }

        indx = 0
        for dataset in datasets:

            for i in range(
                    datasets[dataset]["start"],
                    datasets[dataset]["stop"]
            ):
                logger.debug(f"{dataset} {i}")
                fn0 = fnpattern.format(dataset=dataset, i=i, k=0)

                if not (fnmasks / fn0).exists():
                    # logger.info(f"File {fn0} exists. Skipping")
                    # continue
                    segm3dp = io3d.datasets.read_dataset(dataset, organ_label, i)
                    if segm3dp is None:
                        logger.info(f"      Organ label '{organ_label}' does not exist. Skipping.")
                        continue

                    segm3d = segm3dp["data3d"]
                    fnmasks.mkdir(parents=True, exist_ok=True)
                    for k in range(segm3dp.data3d.shape[0]):
                        np.save(fnmasks / fnpattern.format(dataset=dataset, i=i, k=k) , segm3d[k])

                if not (fnimgs / fn0).exists():
                    data3dp = io3d.datasets.read_dataset(dataset, "data3d", i)

                    data3d = window(data3dp["data3d"], center=40, width=400, vmin_out=0, vmax_out=255, dtype=np.uint8)

                    bn = bodynavigation.body_navigation.BodyNavigation(data3dp["data3d"], voxelsize_mm=data3dp["voxelsize_mm"])

                    feature_list = [
                        data3d,
                        bn.dist_to_sagittal(),
                        bn.dist_coronal(),
                        bn.dist_to_diaphragm_axial(),
                        bn.dist_to_surface(),
                    ]
                    # print(f"shapes: data3d={data3d.shape}, dst={dst.shape}")
                    # for j in range(0, data3d.shape[0]):
                    #     imgs_train.append(np.stack([data3d[j, :, :], feature_list[0][j, :, :]], axis=2))
                    #     masks_train.append(segm3d[j, :, :])

                    all_features = expand_dims_and_concat(feature_list, 3)
                    fnimgs.mkdir(parents=True, exist_ok=True)
                    for k in range(all_features.shape[0]):
                        np.save(fnimgs / fnpattern.format(dataset=dataset, i=i, k=k), all_features[k])
                        indx += 1
                    logger.debug(f"i={i}, {all_features.shape}")



        # imgs_train = np.array(imgs_train, dtype=np.int16)
        # masks_train = np.array(masks_train, dtype=np.uint8)
        # np.save(fnimgs, imgs_train)
        # np.save(fnmasks, masks_train)
        # print(f'Saving to .npy files done. imgs.shape={imgs_train.shape}, masks.shape={masks_train.shape}')
    # return imgs_train, masks_train


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# The functions return our metric and loss

# %%

# one_weight = (1-num_of_ones)/(num_of_ones + num_of_zeros)
# zero_weight = (1-num_of_zeros)/(num_of_ones + num_of_zeros)

def weighted_binary_crossentropy(zero_weight, one_weight):
    def weighted_binary_crossentropy(y_true, y_pred):
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # weighted calc
        weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def save_segmentations(imgs_test, imgs_mask_test, pred_dir='preds'):
    print(f"shapes={imgs_test.shape},{imgs_mask_test.shape}")

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for k in range(len(imgs_mask_test)):
        a = rescale_intensity(imgs_test[k][:, :], out_range=(-1, 1))
        b = (imgs_mask_test[k][:, :] > 0.5).astype('uint8')
        io.imsave(os.path.join(pred_dir, f'{k:05}_pred.png'), mark_boundaries(a, b))

# nb_channels = 2

class UNetTrainer():
    def __init__(self, nb_channels, img_rows, img_cols, experiment_label, organ_label):
        self.nb_channels = nb_channels
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.experiment_label = experiment_label
        self.organ_label = organ_label
        pass

    def get_unet(self, weights=None):
        if weights is None:
            weights = [0.05956, 3.11400]
            # {0: 0.5956388648542532, 1: 3.1140000760253925}

        inputs = Input((self.img_rows, self.img_cols, self.nb_channels))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        #     conv10 = Conv2D(2, (1, 1), activation='softmax')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        #     model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
        #     model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=[dice_coef, "accuracy"])
        model.compile(optimizer='adam', loss=weighted_binary_crossentropy(weights[0], weights[1]),
                      metrics=[dice_coef, "accuracy"])
        # model.compile(optimizer='adam',  loss=weighted_binary_crossentropy(weights[0], weights[1]), metrics=[dice_coef, "accuracy"])  # categorical crossentropy (weighted)

        return model


    # The different layers in our neural network model (including convolutions, maxpooling and upsampling)

    # %%

    def preprocess(self, imgs, is_mask=False):
        new_shape = list(imgs.shape).copy()
        new_shape[1] = self.img_rows
        new_shape[2] = self.img_cols
        #         imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, imgs.shape[3]), dtype=np.uint8)
        imgs_p = np.ndarray(new_shape, dtype=np.uint8)
        for i in range(imgs.shape[0]):
            imgs_p[i] = resize(imgs[i], new_shape[1:], preserve_range=True)
        #         imgs_p[i] = resize(imgs[i, 0 ], (img_cols, img_rows), preserve_range=True)

        # imgs_p = imgs_p[..., np.newaxis]
        if is_mask:
            imgs_p = (imgs_p > 0).astype('float32')

        else:
            imgs_p = imgs_p.astype('float32')
        return imgs_p


    # We adapt here our dataset samples dimension so that we can feed it to our network


    # %%



    # %%

    def train_and_predict(self, continue_training=False, epochs=50, step=1):
        # if True:
        print('-' * 30)
        print('Loading and preprocessing train data...')
        print('-' * 30)
        experiment_label = self.experiment_label
        # imgs_train, imgs_mask_train = load_train_data(self.experiment_label)
        imgs_train, imgs_mask_train = get_dataset_loaders("train", self.organ_label)
        imgs_train = imgs_train[::step]
        imgs_mask_train = imgs_mask_train[::step]

        logger.debug(f"imgs_train.shape={imgs_train.shape}")
        logger.debug(f"imgs_mask_train.shape={imgs_mask_train.shape}")

        imgs_train = self.preprocess(imgs_train)
        imgs_mask_train = self.preprocess(imgs_mask_train, is_mask=True)

        logger.debug(f"imgs_train.shape={imgs_train.shape}")
        logger.debug(f"imgs_mask_train.shape={imgs_mask_train.shape}")

        # TODO remove - using small part of dataset
        #     imgs_train = imgs_train[50:65]
        #     imgs_mask_train = imgs_mask_train[50:65]

        #     imgs_train = imgs_train.astype('float32')
        #     mean = np.mean(imgs_train)  # mean for data centering
        #     std = np.std(imgs_train)  # std for data normalization

        #     imgs_train -= mean
        #     imgs_train /= std
        # Normalization of the train set

        #     imgs_mask_train = (imgs_mask_train > 0).astype('float32')

        y_train = imgs_mask_train
        # Calculate the weights for each class so that we can balance the data
        cl_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(y_train.flatten()),
            y_train.flatten()
        )
        print(f"weights={cl_weights}")
        cl_weights_dct = dict(enumerate(cl_weights))

        print('-' * 30)
        print('Creating and compiling model...')
        print('-' * 30)
        model = self.get_unet(cl_weights)
        if continue_training:
            model.load_weights(f'weights_{experiment_label}.h5')
        model_checkpoint = ModelCheckpoint(f'weights_{experiment_label}.h5', monitor='val_loss', save_best_only=True)
        # Saving the weights and the loss of the best predictions we obtained

        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        log_dir = f'logs\\{experiment_label}\\'
        # Path(log_dir).mkdir(parents=True, exist_ok=True)
        model.fit_generator()
        history = model.fit(
            imgs_train, imgs_mask_train, batch_size=10, epochs=epochs, verbose=1, shuffle=True,
            validation_split=0.2,
            callbacks=[
                model_checkpoint,
                tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            ],
            #                 class_weight=weights_dct # tohle nefunguje pro 4d data
        )
        # predict_test_data(mean=None, std=None)
        self.predict_test_data(history)
        return history


    def predict_test_data(self, history):
        print('-' * 30)
        print('Loading and preprocessing test data...')
        print('-' * 30)
        # imgs_test, imgs_maskt = load_test_data(self.experiment_label)
        imgs_test, imgs_maskt = get_dataset_loaders("test", self.organ_label)
        imgs_test = self.preprocess(imgs_test)
        imgs_maskt = self.preprocess(imgs_maskt, is_mask=True)

        y_train = imgs_maskt
        # Calculate the weights for each class so that we can balance the data
        cl_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(y_train.flatten()),
            y_train.flatten()
        )
        model = self.get_unet(cl_weights)

        # TODO remove this limit
        #     imgs_test = imgs_test[50:65]
        #     imgs_maskt = imgs_maskt[50:65]

        imgs_test = imgs_test.astype('float32')
        #     imgs_test -= mean
        #     imgs_test /= std
        # Normalization of the test set

        # TODO remove this part
        # going to test on train set
        #     imgs_test = imgs_train
        #     imgs_maskt = imgs_mask_train

        print('-' * 30)
        print('Loading saved weights...')
        print('-' * 30)
        model.load_weights(f'weights_{self.experiment_label}.h5')

        print('-' * 30)
        print('Predicting masks on test data...')
        print('-' * 30)
        imgs_mask_test = model.predict(imgs_test, verbose=1)
        np.save('imgs_mask_test.npy', imgs_mask_test)
        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = f"preds/{self.experiment_label}"
        Path(pred_dir).mkdir(parents=True, exist_ok=True)
        # Saving our predictions in the directory 'preds'
        logger.debug(f"imgs_test.shape={imgs_test.shape}")
        logger.debug(f"imgs_mask_test.shape={imgs_mask_test.shape}")
        # save_segmentations(imgs_test[:, :, :, 0, 0], imgs_mask_test[:, :, :, 0], pred_dir=pred_dir)
        save_segmentations(imgs_test[:, :, :, 0], imgs_mask_test[:, :, :, 0], pred_dir=pred_dir)

        plt.plot(history.history['dice_coef'])
        plt.plot(history.history['val_dice_coef'])
        plt.title('Model dice coeff')
        plt.ylabel('Dice coeff')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        # plotting our dice coeff results in function of the number of epochs
    def load_batch():
        pass


def expand_dims_and_concat(larr:np.ndarray, axis:int):
    larr = list(map(lambda x: np.expand_dims(x,axis), larr))
    arr = np.concatenate(larr, axis=axis)
    return arr

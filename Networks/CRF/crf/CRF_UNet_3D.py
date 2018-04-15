

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation, InputLayer
from keras.optimizers import Adam, Adadelta
import tensorflow as tf
import Modules.LossFunction as lf
import Modules.Common_modules as cm
import numpy as np
from Networks.CRF.crf.crf import CRFLayer

#######################################################
# Getting 3D CRF U-net:

vo

def get_3d_unet():

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1), name='layer_no_0_input')
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_1_conv')(inputs)
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_2_conv')(conv1)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2), name='layer_no_3')(conv1)

  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_4_conv')(pool1)
  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_5_conv')(conv2)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2), name='layer_no_6')(conv2)

  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_7_conv')(pool2)
  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_8_conv')(conv3)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2), name='layer_no_9')(conv3)

  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_10_conv')(pool3)
  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_11_conv')(conv4)
  pool4 = MaxPooling3D(pool_size=(2, 2, 2), name='layer_no_12')(conv4)

  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_13_conv')(pool4)
  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_14_conv')(conv5)

  up6 = merge([UpSampling3D(size=(2, 2, 2), name='layer_no_15')(conv5), conv4], mode='concat', concat_axis=-1, name='layer_no_16')
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_17_conv')(up6)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_18_conv')(conv6)

  up7 = merge([UpSampling3D(size=(2, 2, 2), name='layer_no_19')(conv6), conv3], mode='concat', concat_axis=-1, name='layer_no_20')
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_21_conv')(up7)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_22_conv')(conv7)

  up8 = merge([UpSampling3D(size=(2, 2, 2), name='layer_no_23')(conv7), conv2], mode='concat', concat_axis=-1, name='layer_no_24')
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_25_conv')(up8)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_26_conv')(conv8)

  up9 = merge([UpSampling3D(size=(2, 2, 2), name='layer_no_27')(conv8), conv1], mode='concat', concat_axis=-1, name='layer_no_28')
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_29_conv')(up9)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_30_last')(conv9)

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid', name='layer_no_31_output')(conv9)

  model = Model(input=inputs, output=conv10)

  # weights = np.array([1.0, 1.0, 1.0])
  # loss = lf.weighted_categorical_crossentropy_loss(weights)
  # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  # model.compile(optimizer=Adam(lr=1.0e-5), loss=loss, metrics=["categorical_accuracy"])
  model.compile(optimizer=Adam(lr=1.0e-6), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  # model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

  return model


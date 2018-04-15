

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation, Deconvolution3D
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm
import numpy as np

#######################################################
# Getting 3D Residual Symmetric U-net:


def residual_3_3_1_block(x, filters):

  conv_1 = Conv3D(filters=filters, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='same')(x)
  bn1 = BatchNormalization(axis=-1)(conv_1)
  act1 = Activation('relu')(bn1)

  return act1


def residual_3_3_3_block(x, filters):
  conv_1 = Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
  bn1 = BatchNormalization(axis=-1)(conv_1)
  act1 = Activation('relu')(bn1)

  return act1


def residual_module_3d(x, filters, order):

  res_2 = residual_3_3_3_block(x, filters)

  res_3 = residual_3_3_3_block(res_2, filters)

  merge_1 = merge([res_2, res_3], mode='sum')
  # merge_1 = merge([res_2, res_3], mode='concat', concat_axis=-1)

  if order == 'prev':
    res_4 = residual_3_3_3_block(merge_1, filters)

  elif order == 'last':
    res_4 = Conv3D(filters=filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', name='layer_no_30_last')(merge_1)
    res_4 = BatchNormalization(axis=-1)(res_4)
    res_4 = Activation('relu')(res_4)

  else:
    res_4 = residual_3_3_3_block(merge_1, filters)

  return res_4


def get_3d_rsunet():

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  conv1 = residual_module_3d(inputs, 32, 'prev')
  pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

  conv2 = residual_module_3d(pool1, 64, 'prev')
  pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

  conv3 = residual_module_3d(pool2, 128, 'prev')
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

  conv4 = residual_module_3d(pool3, 256, 'prev')
  pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

  conv5 = residual_module_3d(pool4, 256, 'prev')

  up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
  conv6 = residual_module_3d(up6, 128, 'prev')

  up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
  conv7 = residual_module_3d(up7, 64, 'prev')

  up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
  conv8 = residual_module_3d(up8, 32, 'prev')

  up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
  conv9 = residual_module_3d(up9, 16, 'last')

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(conv9)

  model = Model(input=inputs, output=conv10)

  weights = np.array([1, 1, 1])
  loss = lf.weighted_categorical_crossentropy_loss(weights)
  # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  model.compile(optimizer=Adam(lr=1.0e-6), loss=loss, metrics=["categorical_accuracy"])

  return model


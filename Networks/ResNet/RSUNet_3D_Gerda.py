

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation, Deconvolution3D
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm
import numpy as np

#######################################################
# Getting 3D Residual Symmetric U-net:


def conv_block(x, F, A, Sz, Sxy, P):

  conv_1 = Conv3D(filters=F, kernel_size=(A, A, A), strides=(Sz, Sxy, Sxy), padding=P)(x)

  return conv_1

def BN_ReLU(x):

  bn1 = BatchNormalization(axis=-1)(x)
  act1 = Activation('relu')(bn1)

  return act1

def residual_module_3d(x, F, A, B, P):

  res_2 = BN_ReLU(x)
  res_2 = conv_block(res_2, F, A, 1, 1, P)

  res_3 = BN_ReLU(res_2)
  res_3 = conv_block(res_3, F, B, 1, 1, P)

  merge_1 = merge([x, res_3], mode='sum')
  # merge_1 = merge([res_2, res_3], mode='concat', concat_axis=-1)

  return merge_1


def plain_module_3d(x, F, A, B, P):

  res_2 = BN_ReLU(x)
  res_2 = conv_block(res_2, F, A, 1, 1, P)

  res_3 = BN_ReLU(res_2)
  res_3 = conv_block(res_3, F, B, 1, 1, P)

  return res_3


def concat_block(up1,input, F, z, xy, P):

  up1 = Conv3D(filters=F, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding=P)(up1)

  up1 = UpSampling3D(size=(z, xy, xy))(up1)

  conv1 = Conv3D(filters=F, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding=P)(input)

  merge1 = merge([up1, conv1], mode='concat', concat_axis=-1)

  return merge1


def get_3d_rsunet_Gerda(opti):

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
  layer2 = BN_ReLU(layer1)
  layer3 = conv_block(layer2, 32, 2, 2, 2, 'same')

  block1 = residual_module_3d(layer3, 32, 3, 3, 'same')

  layer4 = BN_ReLU(block1)
  layer5 = conv_block(layer4, 64, 2, 2, 2, 'same')

  block2 = residual_module_3d(layer5, 64, 3, 1, 'same')

  block3 = residual_module_3d(block2, 64, 3, 1, 'same')

  layer6 = BN_ReLU(block3)
  layer7 = conv_block(layer6, 128, 2, 2, 2, 'same')

  block4 = residual_module_3d(layer7, 128, 3, 1, 'same')

  block5 = residual_module_3d(block4, 128, 3, 1, 'same')

  layer8 = BN_ReLU(block5)

  concat1 = concat_block(layer8, layer6, 32, 2, 2, 'same')

  block6 = residual_module_3d(concat1, 64, 3 ,1, 'same')

  block7 = residual_module_3d(block6, 64, 3, 1, 'same')

  layer9 = BN_ReLU(block7)

  concat2 = concat_block(layer9, layer4, 16, 2, 2, 'same')

  block8 = residual_module_3d(concat2, 32, 3, 1, 'same')

  layer10 = BN_ReLU(block8)

  concat3 = concat_block(layer10, layer2, 8, 2, 2, 'same')

  layer11 = conv_block(concat3, 16, 3, 1, 1, 'same')

  layer12 = BN_ReLU(layer11)

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(layer12)

  model = Model(input=inputs, output=conv10)

  weights = np.array([0.1, 10, 10])
  loss = lf.weighted_categorical_crossentropy_loss(weights)
  # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  model.compile(optimizer=opti, loss=loss, metrics=["categorical_accuracy"])

  return model


def get_3d_rsunet_Gerdafeature(opti):

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  layer1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
  layer2 = BN_ReLU(layer1)
  layer3 = conv_block(layer2, 64, 2, 2, 2, 'same')

  block1 = residual_module_3d(layer3, 64, 3, 3, 'same')

  layer4 = BN_ReLU(block1)
  layer5 = conv_block(layer4, 128, 2, 2, 2, 'same')

  block2 = residual_module_3d(layer5, 128, 3, 1, 'same')

  block3 = residual_module_3d(block2, 128, 3, 1, 'same')

  layer6 = BN_ReLU(block3)
  layer7 = conv_block(layer6, 256, 2, 2, 2, 'same')

  block4 = residual_module_3d(layer7, 256, 3, 1, 'same')

  block5 = residual_module_3d(block4, 256, 3, 1, 'same')

  layer8 = BN_ReLU(block5)

  concat1 = concat_block(layer8, layer6, 64, 2, 2, 'same')

  block6 = residual_module_3d(concat1, 128, 3 ,1, 'same')

  block7 = residual_module_3d(block6, 128, 3, 1, 'same')

  layer9 = BN_ReLU(block7)

  concat2 = concat_block(layer9, layer4, 32, 2, 2, 'same')

  block8 = residual_module_3d(concat2, 64, 3, 1, 'same')

  layer10 = BN_ReLU(block8)

  concat3 = concat_block(layer10, layer2, 16, 2, 2, 'same')

  layer11 = conv_block(concat3, 32, 3, 1, 1, 'same')

  layer12 = BN_ReLU(layer11)

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(layer12)

  model = Model(input=inputs, output=conv10)

  weights = np.array([0.1, 10, 10])
  loss = lf.weighted_categorical_crossentropy_loss(weights)
  # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  model.compile(optimizer=opti, loss=loss, metrics=["categorical_accuracy"])

  return model


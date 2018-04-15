

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Lambda
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation
from keras.layers import LSTM, TimeDistributed, RepeatVector, ConvLSTM2D, Bidirectional
from keras import backend as K
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm
from Modules.lstm import LstmParam, LstmNetwork
import numpy as np
import tensorlayer as tl
import tensorflow as tf

class ToyLossLayer:
  """
  Computes square loss with first element of hidden layer array.
  """

  @classmethod
  def loss(self, pred, label):
    return (pred[0] - label) ** 2

  @classmethod
  def bottom_diff(self, pred, label):
    diff = np.zeros_like(pred)
    diff[0] = 2 * (pred[0] - label)
    return diff
#######################################################
# Getting 3D LSTM U-net:

def time_dist_softmax(x):
  assert K.ndim(x) == 5
  # e = K.exp(x - K.max(x, axis=2, keepdims=True))
  e = K.exp(x)
  s = K.sum(e, axis=2, keepdims=True)
  return e / s


def time_dist_softmax_out_shape(input_shape):
  shape = list(input_shape)
  return tuple(shape)

def time_ConvLSTM_bottleNeck_block(x, filters, row, col):
  reduced_filters = filters
  if filters >= 8:
    reduced_filters = int(round(filters / 8))
  x = TimeDistributed(
    Conv2D(nb_filter=reduced_filters, nb_row=1, nb_col=1, activation='relu', border_mode='same'))(x)
  x = Bidirectional(
    ConvLSTM2D(nb_filter=reduced_filters, nb_row=row, nb_col=col, dim_ordering='th', border_mode='same',
               return_sequences=True), merge_mode='sum')(x)
  x = TimeDistributed(Conv2D(nb_filter=filters, nb_row=1, nb_col=1, activation='relu', border_mode='same'))(x)
  return x



def time_GRU_unet_1_level():
  inputs = Input((cm.img_rows_2d, cm.img_cols_2d, 1))
  conv1 = TimeDistributed(Conv2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)
  conv1 = TimeDistributed(Conv2D(32, 3, 3, activation='relu', border_mode='same'))(conv1)
  pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

  conv2 = TimeDistributed(Conv2D(64, 3, 3, activation='relu', border_mode='same'))(pool1)
  conv2 = TimeDistributed(Conv2D(64, 3, 3, activation='relu', border_mode='same'))(conv2)
  pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

  conv3 = TimeDistributed(Conv2D(128, 3, 3, activation='relu', border_mode='same'))(pool2)
  conv3 = TimeDistributed(Conv2D(128, 3, 3, activation='relu', border_mode='same'))(conv3)
  pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)

  conv4 = TimeDistributed(Conv2D(256, 3, 3, activation='relu', border_mode='same'))(pool3)
  conv4 = TimeDistributed(Conv2D(256, 3, 3, activation='relu', border_mode='same'))(conv4)
  pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)

  conv5 = TimeDistributed(Conv2D(512, 3, 3, activation='relu', border_mode='same'))(pool4)
  conv5 = time_ConvLSTM_bottleNeck_block(conv5, 512, 3, 3)
  # conv5_1 = TimeDistributed(Conv2D(512, 3, 3, activation='relu', border_mode='same'))(conv5)

  up6 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv5), conv4], mode='concat', concat_axis=2)
  conv6 = TimeDistributed(Conv2D(256, 3, 3, activation='relu', border_mode='same'))(up6)
  conv6 = TimeDistributed(Conv2D(256, 3, 3, activation='relu', border_mode='same'))(conv6)

  up7 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv6), conv3], mode='concat', concat_axis=2)
  conv7 = TimeDistributed(Conv2D(128, 3, 3, activation='relu', border_mode='same'))(up7)
  conv7 = TimeDistributed(Conv2D(128, 3, 3, activation='relu', border_mode='same'))(conv7)

  up8 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv7), conv2], mode='concat', concat_axis=2)
  conv8 = TimeDistributed(Conv2D(64, 3, 3, activation='relu', border_mode='same'))(up8)
  conv8 = TimeDistributed(Conv2D(64, 3, 3, activation='relu', border_mode='same'))(conv8)

  up9 = merge([TimeDistributed(UpSampling2D(size=(2, 2)))(conv8), conv1], mode='concat', concat_axis=2)
  conv9 = TimeDistributed(Conv2D(32, 3, 3, activation='relu', border_mode='same'))(up9)
  conv9 = TimeDistributed(Conv2D(32, 3, 3, activation='relu', border_mode='same'))(conv9)
  conv10 = TimeDistributed(Conv2D(2, 1, 1, activation='relu'))(conv9)
  out = Lambda(time_dist_softmax, output_shape=time_dist_softmax_out_shape)(conv10)
  return out


def get_3d_lstm_unet():

  inputs = Input((cm.img_rows_2d, cm.img_cols_2d, 1))

  conv1 = TimeDistributed(Conv2D(32, 3, 3, activation='relu', border_mode='same'))(inputs)

  conv1 = time_ConvLSTM_bottleNeck_block(conv1,256,3,3)

  conv6 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv1)

  model = Model(input=inputs, output=conv6)

  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])



def get_3d_unet():

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(inputs)
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv1)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool1)
  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv2)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool2)
  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv3)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool3)
  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv4)
  pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool4)
  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv5)

  up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up6)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv6)

  up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up7)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv7)

  up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up8)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv8)

  up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up9)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv9)

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(conv9)

  model = Model(input=inputs, output=conv10)

  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model
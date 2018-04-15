
from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation, InputLayer
from keras.optimizers import Adam, Adadelta
import tensorflow as tf
import Modules.LossFunction as lf
import Modules.Common_modules as cm
import numpy as np

#######################################################
# Getting 3D CNN:
def spatial_dropout(x, keep_prob, seed=1234):
    # x is a convnet activation with shape BxWxHxF where F is the
    # number of feature maps for that layer
    # keep_prob is the proportion of feature maps we want to keep

    # get the batch size and number of feature maps
    num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]

    # get some uniform noise between keep_prob and 1 + keep_prob
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(num_feature_maps,
                                       seed=seed,
                                       dtype=x.dtype)

    # if we take the floor of this, we get a binary matrix where
    # (1-keep_prob)% of the values are 0 and the rest are 1
    binary_tensor = tf.floor(random_tensor)

    # Reshape to multiply our feature maps by this tensor correctly
    binary_tensor = tf.reshape(binary_tensor,
                               [-1, 1, 1, tf.shape(x)[3]])
    # Zero out feature maps where appropriate; scale up to compensate
    ret = tf.div(x, keep_prob) * binary_tensor
    return ret

def get_3d_unet_bn():

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(inputs)
  bn1 = BatchNormalization(axis=-1)(conv1)
  act1 = Activation('relu')(bn1)
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act1)
  bn1 = BatchNormalization(axis=-1)(conv1)
  act1 = Activation('relu')(bn1)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2))(act1)

  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(pool1)
  bn2 = BatchNormalization(axis=-1)(conv2)
  act2 = Activation('relu')(bn2)
  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act2)
  bn2 = BatchNormalization(axis=-1)(conv2)
  act2 = Activation('relu')(bn2)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2))(act2)

  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(pool2)
  bn3 = BatchNormalization(axis=-1)(conv3)
  act3 = Activation('relu')(bn3)
  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act3)
  bn3 = BatchNormalization(axis=-1)(conv3)
  act3 = Activation('relu')(bn3)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(act3)

  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(pool3)
  bn4 = BatchNormalization(axis=-1)(conv4)
  act4 = Activation('relu')(bn4)
  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act4)
  bn4 = BatchNormalization(axis=-1)(conv4)
  act4 = Activation('relu')(bn4)
  pool4 = MaxPooling3D(pool_size=(2, 2, 2))(act4)

  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(pool4)
  bn5 = BatchNormalization(axis=-1)(conv5)
  act5 = Activation('relu')(bn5)
  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act5)
  bn5 = BatchNormalization(axis=-1)(conv5)
  act5 = Activation('relu')(bn5)

  up6 = merge([UpSampling3D(size=(2, 2, 2))(act5), act4], mode='concat', concat_axis=-1)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(up6)
  bn6 = BatchNormalization(axis=-1)(conv6)
  act6 = Activation('relu')(bn6)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act6)
  bn6 = BatchNormalization(axis=-1)(conv6)
  act6 = Activation('relu')(bn6)

  up7 = merge([UpSampling3D(size=(2, 2, 2))(act6), act3], mode='concat', concat_axis=-1)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(up7)
  bn7 = BatchNormalization(axis=-1)(conv7)
  act7 = Activation('relu')(bn7)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act7)
  bn7 = BatchNormalization(axis=-1)(conv7)
  act7 = Activation('relu')(bn7)

  up8 = merge([UpSampling3D(size=(2, 2, 2))(act7), act2], mode='concat', concat_axis=-1)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(up8)
  bn8 = BatchNormalization(axis=-1)(conv8)
  act8 = Activation('relu')(bn8)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act8)
  bn8 = BatchNormalization(axis=-1)(conv8)
  act8 = Activation('relu')(bn8)

  up9 = merge([UpSampling3D(size=(2, 2, 2))(act8), act1], mode='concat', concat_axis=-1)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(up9)
  bn9 = BatchNormalization(axis=-1)(conv9)
  act9 = Activation('relu')(bn9)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act9)
  bn9 = BatchNormalization(axis=-1)(conv9)
  act9 = Activation('relu')(bn9)

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(act9)

  model = Model(input=inputs, output=conv10)

  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model


def get_3d_cnn():

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

  up6 = UpSampling3D(size=(2, 2, 2), name='layer_no_15')(conv5)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_17_conv')(up6)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_18_conv')(conv6)

  up7 = UpSampling3D(size=(2, 2, 2), name='layer_no_19')(conv6)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_21_conv')(up7)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_22_conv')(conv7)

  up8 = UpSampling3D(size=(2, 2, 2), name='layer_no_23')(conv7)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_25_conv')(up8)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_26_conv')(conv8)

  up9 = UpSampling3D(size=(2, 2, 2), name='layer_no_27')(conv8)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_29_conv')(up9)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same', name='layer_no_30_last')(conv9)

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid', name='layer_no_31_output')(conv9)

  model = Model(input=inputs, output=conv10)

  # weights = np.array([1.0, 1.0, 1.0])
  # loss = lf.weighted_categorical_crossentropy_loss(weights)
  # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  # model.compile(optimizer=Adam(lr=1.0e-5), loss=loss, metrics=["categorical_accuracy"])
  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  # model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

  return model


def get_3d_unet_4class(opti):

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

  weights = np.array([1, 100, 100])
  loss = lf.weighted_categorical_crossentropy_loss(weights)
  # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  model.compile(optimizer=opti, loss=loss, metrics=["categorical_accuracy"])
  # model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  # model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

  return model

def get_3d_wnet(opti):

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  conv1 = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(inputs)
  conv1 = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv1)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

  conv1p = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(inputs)
  conv1p = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv1p)
  pool1p = MaxPooling3D(pool_size=(2, 2, 2))(conv1p)

  conv2 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool1)
  conv2 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv2)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

  conv2p = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool1p)
  conv2p = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv2p)
  pool2p = MaxPooling3D(pool_size=(2, 2, 2))(conv2p)

  conv3 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool2)
  conv3 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv3)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

  conv3p = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool2p)
  conv3p = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv3p)
  pool3p = MaxPooling3D(pool_size=(2, 2, 2))(conv3p)

  conv4 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool3)
  conv4 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv4)
  pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

  conv4p = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool3p)
  conv4p = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv4p)
  pool4p = MaxPooling3D(pool_size=(2, 2, 2))(conv4p)

  conv5 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool4)
  conv5 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv5)

  conv5p = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool4p)
  conv5p = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv5p)

  up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4, UpSampling3D(size=(2, 2, 2))(conv5p), conv4p], mode='concat', concat_axis=-1)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up6)
  conv6 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv6)

  conv6p = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up6)
  conv6p = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv6p)

  up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3, UpSampling3D(size=(2, 2, 2))(conv6p), conv3p], mode='concat', concat_axis=-1)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up7)
  conv7 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv7)

  conv7p = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up7)
  conv7p = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv7p)

  up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2, UpSampling3D(size=(2, 2, 2))(conv7p), conv2p], mode='concat', concat_axis=-1)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up8)
  conv8 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv8)

  conv8p = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up8)
  conv8p = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv8p)

  up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1, UpSampling3D(size=(2, 2, 2))(conv8p), conv1p], mode='concat', concat_axis=-1)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up9)
  conv9 = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv9)

  conv9p = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up9)
  conv9p = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv9p)

  conv10a = merge([conv9, conv9p],  mode='concat', concat_axis=-1)
  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(conv10a)

  model = Model(input=inputs, output=conv10)

  weights = np.array([1, 100, 100])
  loss = lf.weighted_categorical_crossentropy_loss(weights)
  # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  model.compile(optimizer=opti, loss=loss, metrics=["categorical_accuracy"])
  # model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model
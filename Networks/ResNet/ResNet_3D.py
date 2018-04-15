

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, AveragePooling3D, Activation
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm

#######################################################
# Getting 3D ResNet:


def get_3d_resnet_34():

  k = 64

  # Input
  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  # Convolution 1
  conv1 = Conv3D(filters=k, kernel_size=(7, 7, 7), strides=(1, 2, 2), padding='same')(inputs)
  bn1 = BatchNormalization(axis=-1)(conv1)
  act1 = Activation('relu')(bn1)

  # Max pooling 1
  pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(act1)

  # Residual block 1-1
  conv2_1 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool1)
  bn2 = BatchNormalization(axis=-1)(conv2_1)
  act2 = Activation('relu')(bn2)

  conv2_1 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)
  bn3 = BatchNormalization(axis=-1)(conv2_1)
  act3 = Activation('relu')(bn3)

  # Residual block 1-2
  merge1 = merge([act3, pool1], mode='sum')
  conv2_2 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge1)
  bn4 = BatchNormalization(axis=-1)(conv2_2)
  act4 = Activation('relu')(bn4)

  conv2_2 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)
  bn5 = BatchNormalization(axis=-1)(conv2_2)
  act5 = Activation('relu')(bn5)

  # Residual block 1-3
  merge2 = merge([act5, act3], mode='sum')
  conv2_3 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge2)
  bn6 = BatchNormalization(axis=-1)(conv2_3)
  act6 = Activation('relu')(bn6)

  conv2_3 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act6)
  bn7 = BatchNormalization(axis=-1)(conv2_3)
  act7 = Activation('relu')(bn7)

  conv2_3_ws = Conv3D(filters=(2 * k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act7)
  bn7_ws = BatchNormalization(axis=-1)(conv2_3_ws)
  act7_ws = Activation('relu')(bn7_ws)

  # Residual block 2-1
  merge3 = merge([act7, act5], mode='sum')
  conv3_1 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge3)
  bn8 = BatchNormalization(axis=-1)(conv3_1)
  act8 = Activation('relu')(bn8)

  conv3_1 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act8)
  bn9 = BatchNormalization(axis=-1)(conv3_1)
  act9 = Activation('relu')(bn9)

  # Residual block 2-2
  merge4 = merge([act9, act7_ws], mode='sum')
  conv3_2 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge4)
  bn10 = BatchNormalization(axis=-1)(conv3_2)
  act10 = Activation('relu')(bn10)

  conv3_2 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act10)
  bn11 = BatchNormalization(axis=-1)(conv3_2)
  act11 = Activation('relu')(bn11)

  # Residual block 2-3
  merge5 = merge([act11, act9], mode='sum')
  conv3_3 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge5)
  bn12 = BatchNormalization(axis=-1)(conv3_3)
  act12 = Activation('relu')(bn12)

  conv3_3 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act12)
  bn13 = BatchNormalization(axis=-1)(conv3_3)
  act13 = Activation('relu')(bn13)

  # Residual block 2-4
  merge6 = merge([act13, act11], mode='sum')
  conv3_4 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge6)
  bn14 = BatchNormalization(axis=-1)(conv3_4)
  act14 = Activation('relu')(bn14)

  conv3_4 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act14)
  bn15 = BatchNormalization(axis=-1)(conv3_4)
  act15 = Activation('relu')(bn15)

  conv3_4_ws = Conv3D(filters=(4 * k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act15)
  bn15_ws = BatchNormalization(axis=-1)(conv3_4_ws)
  act15_ws = Activation('relu')(bn15_ws)

  # Residual block 3-1
  merge7 = merge([act15, act13], mode='sum')

  conv4_1 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge7)
  bn16 = BatchNormalization(axis=-1)(conv4_1)
  act16 = Activation('relu')(bn16)

  conv4_1 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act16)
  bn17 = BatchNormalization(axis=-1)(conv4_1)
  act17 = Activation('relu')(bn17)

  # Residual block 3-2
  merge8 = merge([act17, act15_ws], mode='sum')

  conv4_2 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge8)
  bn18 = BatchNormalization(axis=-1)(conv4_2)
  act18 = Activation('relu')(bn18)

  conv4_2 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act18)
  bn19 = BatchNormalization(axis=-1)(conv4_2)
  act19 = Activation('relu')(bn19)

  # Residual block 3-3
  merge9 = merge([act19, act17], mode='sum')

  conv4_3 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge9)
  bn20 = BatchNormalization(axis=-1)(conv4_3)
  act20 = Activation('relu')(bn20)

  conv4_3 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act20)
  bn21 = BatchNormalization(axis=-1)(conv4_3)
  act21 = Activation('relu')(bn21)

  # Residual block 3-4
  merge10 = merge([act21, act19], mode='sum')

  conv4_4 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge10)
  bn22 = BatchNormalization(axis=-1)(conv4_4)
  act22 = Activation('relu')(bn22)

  conv4_4 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act22)
  bn23 = BatchNormalization(axis=-1)(conv4_4)
  act23 = Activation('relu')(bn23)

  # Residual block 3-5
  merge11 = merge([act23, act21], mode='sum')

  conv4_5 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge11)
  bn24 = BatchNormalization(axis=-1)(conv4_5)
  act24 = Activation('relu')(bn24)

  conv4_5 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act24)
  bn25 = BatchNormalization(axis=-1)(conv4_5)
  act25 = Activation('relu')(bn25)

  # Residual block 3-6
  merge12 = merge([act25, act23], mode='sum')

  conv4_6 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge12)
  bn26 = BatchNormalization(axis=-1)(conv4_6)
  act26 = Activation('relu')(bn26)

  conv4_6 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act26)
  bn27 = BatchNormalization(axis=-1)(conv4_6)
  act27 = Activation('relu')(bn27)

  conv4_6_ws = Conv3D(filters=(8 * k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act27)
  bn27_ws = BatchNormalization(axis=-1)(conv4_6_ws)
  act27_ws = Activation('relu')(bn27_ws)

  # Residual block 4-1
  merge13 = merge([act27, act25], mode='sum')

  conv5_1 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge13)
  bn28 = BatchNormalization(axis=-1)(conv5_1)
  act28 = Activation('relu')(bn28)

  conv5_1 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act28)
  bn29 = BatchNormalization(axis=-1)(conv5_1)
  act29 = Activation('relu')(bn29)

  # Residual block 4-2
  merge14 = merge([act29, act27_ws], mode='sum')

  conv5_2 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge14)
  bn30 = BatchNormalization(axis=-1)(conv5_2)
  act30 = Activation('relu')(bn30)

  conv5_2 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act30)
  bn31 = BatchNormalization(axis=-1)(conv5_2)
  act31 = Activation('relu')(bn31)

  # Residual block 4-3
  merge15 = merge([act31, act29], mode='sum')

  conv5_3 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge15)
  bn32 = BatchNormalization(axis=-1)(conv5_3)
  act32 = Activation('relu')(bn32)

  conv5_3 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act32)
  bn33 = BatchNormalization(axis=-1)(conv5_3)
  act33 = Activation('relu')(bn33)

  # Average pooling + FC + sigmoid
  merge16 = merge([act33, act31], mode='sum')
  pool2 = AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(merge16)
  conv6 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(pool2)

  # Complie
  model = Model(inputs=inputs, outputs=conv6)
  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model


def get_3d_resnet_34_no_bn():

  k = 64

  # Input
  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  # Convolution 1
  conv1 = Conv3D(filters=k, kernel_size=(7, 7, 7), strides=(1, 2, 2), padding='same')(inputs)

  # Max pooling 1
  pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(conv1)

  # Residual block 1-1

  act2 = Activation('relu')(pool1)
  conv2_1 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)

  act3 = Activation('relu')(conv2_1)
  conv2_1 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act3)

  # Residual block 1-2

  act4 = Activation('relu')(conv2_1 + pool1)
  conv2_2 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)

  act5 = Activation('relu')(conv2_2)
  conv2_2 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act5)

  # Residual block 1-3

  act6 = Activation('relu')(conv2_2 + conv2_1)
  conv2_3 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act6)

  act7 = Activation('relu')(conv2_3)
  conv2_3 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7)

  act7_ws = Activation('relu')(conv2_3)
  conv2_3_ws = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7_ws)

  # Residual block 2-1

  act8 = Activation('relu')(conv2_3 + conv2_2)
  conv3_1 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 2, 2))(act8)

  act9 = Activation('relu')(conv3_1)
  conv3_1 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act9)

  # Residual block 2-2

  act10 = Activation('relu')(conv3_1 + conv2_3_ws)
  conv3_2 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act10)

  act11 = Activation('relu')(conv3_2)
  conv3_2 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act11)

  # Residual block 2-3

  act12 = Activation('relu')(conv3_2 + conv3_1)
  conv3_3 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act12)

  act13 = Activation('relu')(conv3_3)
  conv3_3 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act13)

  # Residual block 2-4

  act14 = Activation('relu')(conv3_3 + conv3_2)
  conv3_4 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act14)

  act15 = Activation('relu')(conv3_4)
  conv3_4 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act15)

  act15_ws = Activation('relu')(conv3_4)
  conv3_4_ws = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act15_ws)

  # Residual block 3-1

  act16 = Activation('relu')(conv3_4 + conv3_3)
  conv4_1 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 2, 2))(act16)

  act17 = Activation('relu')(conv4_1)
  conv4_1 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act17)

  # Residual block 3-2

  act18 = Activation('relu')(conv4_1 + conv3_4_ws)
  conv4_2 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act18)

  act19 = Activation('relu')(conv4_2)
  conv4_2 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act19)

  # Residual block 3-3

  act20 = Activation('relu')(conv4_2 + conv4_1)
  conv4_3 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act20)

  act21 = Activation('relu')(conv4_3)
  conv4_3 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act21)

  # Residual block 3-4

  act22 = Activation('relu')(conv4_3 + conv4_2)
  conv4_4 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act22)

  act23 = Activation('relu')(conv4_4)
  conv4_4 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act23)

  # Residual block 3-5

  act24 = Activation('relu')(conv4_4 + conv4_3)
  conv4_5 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act24)

  act25 = Activation('relu')(conv4_5)
  conv4_5 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act25)

  # Residual block 3-6

  act26 = Activation('relu')(conv4_5 + conv4_4)
  conv4_6 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act26)

  act27 = Activation('relu')(conv4_6)
  conv4_6 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act27)

  act27_ws = Activation('relu')(conv4_6)
  conv4_6_ws = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act27_ws)

  # Residual block 4-1

  act28 = Activation('relu')(conv4_6 + conv4_5)
  conv5_1 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 2, 2))(act28)

  act29 = Activation('relu')(conv5_1)
  conv5_1 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act29)

  # Residual block 4-2

  act30 = Activation('relu')(conv5_1 + conv4_6_ws)
  conv5_2 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act30)

  act31 = Activation('relu')(conv5_2)
  conv5_2 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act31)

  # Residual block 4-3

  act32 = Activation('relu')(conv5_2 + conv5_1)
  conv5_3 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act32)

  act33 = Activation('relu')(conv5_3)
  conv5_3 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act33)

  # Average pooling + FC + sigmoid
  pool2 = AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(conv5_3 + conv5_2)

  act34 = Activation('relu')(pool2)
  conv6 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(act34)

  # Complie
  model = Model(input=inputs, output=conv6)
  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model

def get_small3d_resnet():

  k = 64

  # Input
  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  # Convolution 1
  conv1 = Conv3D(filters=k, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu', padding='same')(inputs)

  # Max pooling 1
  pool1 = MaxPooling3D(pool_size=(1, 1, 1), strides=(1, 1, 1))(conv1)

  # Residual block 1-1
  conv2_1 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pool1)
  bn2 = BatchNormalization(axis=-1)(conv2_1)
  act2 = Activation('relu')(bn2)
  conv2_1 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act2)
  bn3 = BatchNormalization(axis=-1)(conv2_1)
  act3 = Activation('relu')(bn3)

  # Residual block 1-2
  merge1 = merge([act3, pool1], mode='sum')
  conv2_2 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(merge1)
  bn4 = BatchNormalization(axis=-1)(merge1)
  act4 = Activation('relu')(bn4)
  conv2_2 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act4)
  bn5 = BatchNormalization(axis=-1)(conv2_2)
  act5 = Activation('relu')(bn5)

  conv6 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid', padding='same')(act5)

  # Complie
  model = Model(input=inputs, output=conv6)
  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model
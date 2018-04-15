

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, AveragePooling3D, Activation
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm

#######################################################
# Getting 3D PlainNet:


def get_3d_plainnet_34():

  k = 64

  # Input
  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  # Convolution 1
  bn1 = BatchNormalization(axis=-1)(inputs)
  act1 = Activation('relu')(bn1)
  conv1 = Conv3D(filters=k, kernel_size=(7, 7, 7), strides=(1, 2, 2), activation='relu', padding='same')(act1)

  # Max pooling 1
  pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(conv1)

  # Residual block 1-1
  bn2 = BatchNormalization(axis=-1)(pool1)
  act2 = Activation('relu')(bn2)
  conv2_1 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act2)
  bn3 = BatchNormalization(axis=-1)(conv2_1)
  act3 = Activation('relu')(bn3)
  conv2_1 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act3)

  # Residual block 1-2
  bn4 = BatchNormalization(axis=-1)(conv2_1)
  act4 = Activation('relu')(bn4)
  conv2_2 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act4)
  bn5 = BatchNormalization(axis=-1)(conv2_2)
  act5 = Activation('relu')(bn5)
  conv2_2 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act5)

  # Residual block 1-3
  bn6 = BatchNormalization(axis=-1)(conv2_2)
  act6 = Activation('relu')(bn6)
  conv2_3 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act6)
  bn7 = BatchNormalization(axis=-1)(conv2_3)
  act7 = Activation('relu')(bn7)
  conv2_3 = Conv3D(filters=(k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act7)

  # Residual block 2-1
  bn8 = BatchNormalization(axis=-1)(conv2_3)
  act8 = Activation('relu')(bn8)
  conv3_1 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 2, 2), activation='relu')(act8)
  bn9 = BatchNormalization(axis=-1)(conv3_1)
  act9 = Activation('relu')(bn9)
  conv3_1 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act9)

  # Residual block 2-2
  bn10 = BatchNormalization(axis=-1)(conv3_1)
  act10 = Activation('relu')(bn10)
  conv3_2 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act10)
  bn11 = BatchNormalization(axis=-1)(conv3_2)
  act11 = Activation('relu')(bn11)
  conv3_2 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act11)

  # Residual block 2-3
  bn12 = BatchNormalization(axis=-1)(conv3_2)
  act12 = Activation('relu')(bn12)
  conv3_3 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act12)
  bn13 = BatchNormalization(axis=-1)(conv3_3)
  act13 = Activation('relu')(bn13)
  conv3_3 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act13)

  # Residual block 2-4
  bn14 = BatchNormalization(axis=-1)(conv3_3)
  act14 = Activation('relu')(bn14)
  conv3_4 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act14)
  bn15 = BatchNormalization(axis=-1)(conv3_4)
  act15 = Activation('relu')(bn15)
  conv3_4 = Conv3D(filters=(2 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act15)

  # Residual block 3-1
  bn16 = BatchNormalization(axis=-1)(conv3_4)
  act16 = Activation('relu')(bn16)
  conv4_1 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 2, 2), activation='relu')(act16)
  bn17 = BatchNormalization(axis=-1)(conv4_1)
  act17 = Activation('relu')(bn17)
  conv4_1 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act17)

  # Residual block 3-2
  bn18 = BatchNormalization(axis=-1)(conv4_1)
  act18 = Activation('relu')(bn18)
  conv4_2 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act18)
  bn19 = BatchNormalization(axis=-1)(conv4_2)
  act19 = Activation('relu')(bn19)
  conv4_2 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act19)

  # Residual block 3-3
  bn20 = BatchNormalization(axis=-1)(conv4_2)
  act20 = Activation('relu')(bn20)
  conv4_3 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act20)
  bn21 = BatchNormalization(axis=-1)(conv4_3)
  act21 = Activation('relu')(bn21)
  conv4_3 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act21)

  # Residual block 3-4
  bn22 = BatchNormalization(axis=-1)(conv4_3)
  act22 = Activation('relu')(bn22)
  conv4_4 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act22)
  bn23 = BatchNormalization(axis=-1)(conv4_4)
  act23 = Activation('relu')(bn23)
  conv4_4 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act23)

  # Residual block 3-5
  bn24 = BatchNormalization(axis=-1)(conv4_4)
  act24 = Activation('relu')(bn24)
  conv4_5 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act24)
  bn25 = BatchNormalization(axis=-1)(conv4_5)
  act25 = Activation('relu')(bn25)
  conv4_5 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act25)

  # Residual block 3-6
  bn26 = BatchNormalization(axis=-1)(conv4_5)
  act26 = Activation('relu')(bn26)
  conv4_6 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act26)
  bn27 = BatchNormalization(axis=-1)(conv4_6)
  act27 = Activation('relu')(bn27)
  conv4_6 = Conv3D(filters=(4 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act27)

  # Residual block 4-1
  bn28 = BatchNormalization(axis=-1)(conv4_6)
  act28 = Activation('relu')(bn28)
  conv5_1 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 2, 2), activation='relu')(act28)
  bn29 = BatchNormalization(axis=-1)(conv5_1)
  act29 = Activation('relu')(bn29)
  conv5_1 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act29)

  # Residual block 4-2
  bn30 = BatchNormalization(axis=-1)(conv5_1)
  act30 = Activation('relu')(bn30)
  conv5_2 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act30)
  bn31 = BatchNormalization(axis=-1)(conv5_2)
  act31 = Activation('relu')(bn31)
  conv5_2 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act31)

  # Residual block 4-3
  bn32 = BatchNormalization(axis=-1)(conv5_2)
  act32 = Activation('relu')(bn32)
  conv5_3 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act32)
  bn33 = BatchNormalization(axis=-1)(conv5_3)
  act33 = Activation('relu')(bn33)
  conv5_3 = Conv3D(filters=(8 * k), kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(act33)

  # Average pooling + FC + sigmoid
  pool2 = AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 2, 2))(conv5_3)
  bn34 = BatchNormalization(axis=-1)(pool2)
  act34 = Activation('relu')(bn34)
  conv6 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(act34)

  # Complie
  model = Model(input=inputs, output=conv6)
  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model

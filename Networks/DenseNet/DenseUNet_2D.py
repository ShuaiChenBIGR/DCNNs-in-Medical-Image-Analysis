

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling3D, BatchNormalization
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm

################
# Getting 2D DenseUNet:
def get_2D_DenseUNet():
  
  k = 48
  c = 0.5
  
  # Input
  inputs = Input((cm.img_rows_2d, cm.img_cols_2d, 1))

  # Convolution 1
  channel1 = (3*k)
  bn1 = BatchNormalization(axis=-1)(inputs)
  conv1 = Conv2D(filters=channel1, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same')(bn1)

  # Pooling
  pool0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

  # Dense Block 1
  bn2 = BatchNormalization(axis=-1)(pool0)
  dense_1_1_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn2)
  bn3 = BatchNormalization(axis=-1)(dense_1_1_conv1)
  dense_1_1_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn3)

  bn4 = BatchNormalization(axis=-1)(dense_1_1_conv2)
  dense_1_2_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn4)
  bn5 = BatchNormalization(axis=-1)(dense_1_2_conv1)
  dense_1_2_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn5)

  shortcut1_1 = merge([dense_1_1_conv2, dense_1_2_conv2], mode='concat', concat_axis=-1)
  bn6 = BatchNormalization(axis=-1)(shortcut1_1)
  dense_1_3_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn6)
  bn7 = BatchNormalization(axis=-1)(dense_1_3_conv1)
  dense_1_3_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn7)

  shortcut1_2 = merge([dense_1_1_conv2, dense_1_2_conv2, dense_1_3_conv2], mode='concat', concat_axis=-1)
  bn6_1 = BatchNormalization(axis=-1)(shortcut1_2)
  dense_1_4_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn6_1)
  bn7_1 = BatchNormalization(axis=-1)(dense_1_4_conv1)
  dense_1_4_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn7_1)

  shortcut1_3 = merge([dense_1_1_conv2, dense_1_2_conv2, dense_1_3_conv2, dense_1_4_conv2], mode='concat', concat_axis=-1)
  bn6_2 = BatchNormalization(axis=-1)(shortcut1_3)
  dense_1_5_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn6_2)
  bn7_2 = BatchNormalization(axis=-1)(dense_1_5_conv1)
  dense_1_5_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn7_2)
  
  shortcut1_4 = merge([dense_1_1_conv2, dense_1_2_conv2, dense_1_3_conv2, dense_1_4_conv2, dense_1_5_conv2], mode='concat', concat_axis=-1)
  bn6_3 = BatchNormalization(axis=-1)(shortcut1_4)
  dense_1_6_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn6_3)
  bn7_3 = BatchNormalization(axis=-1)(dense_1_6_conv1)
  dense_1_6_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn7_3)

  # Transition layer 1
  c_channel1 = int((3 * k + 3 * k) * c)
  bn8 = BatchNormalization(axis=-1)(dense_1_6_conv2)
  transition1 = Conv2D(filters=c_channel1, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn8)
  pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(transition1)

  # Dense Block 2
  bn9 = BatchNormalization(axis=-1)(pool1)
  dense_2_1_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn9)
  bn10 = BatchNormalization(axis=-1)(dense_2_1_conv1)
  dense_2_1_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn10)

  bn11 = BatchNormalization(axis=-1)(dense_2_1_conv2)
  dense_2_2_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn11)
  bn12 = BatchNormalization(axis=-1)(dense_2_2_conv1)
  dense_2_2_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn12)

  shortcut2_1 = merge([dense_2_1_conv2, dense_2_2_conv2], mode='concat', concat_axis=-1)
  bn13 = BatchNormalization(axis=-1)(shortcut2_1)
  dense_2_3_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn13)
  bn14 = BatchNormalization(axis=-1)(dense_2_3_conv1)
  dense_2_3_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn14)

  shortcut2_2 = merge([dense_2_1_conv2, dense_2_2_conv2, dense_2_3_conv2], mode='concat', concat_axis=-1)
  bn15 = BatchNormalization(axis=-1)(shortcut2_2)
  dense_2_4_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn15)
  bn16 = BatchNormalization(axis=-1)(dense_2_4_conv1)
  dense_2_4_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn16)

  # Transition layer 2
  c_channel2 = int((c_channel1 + 4 * k) * c)
  bn17 = BatchNormalization(axis=-1)(dense_2_4_conv2)
  transition2 = Conv2D(filters=c_channel2, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn17)
  pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(transition2)

  # Dense Block 3
  bn18 = BatchNormalization(axis=-1)(pool2)
  dense_3_1_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn18)
  bn19 = BatchNormalization(axis=-1)(dense_3_1_conv1)
  dense_3_1_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn19)

  bn20 = BatchNormalization(axis=-1)(dense_3_1_conv2)
  dense_3_2_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn20)
  bn21 = BatchNormalization(axis=-1)(dense_3_2_conv1)
  dense_3_2_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn21)

  shortcut3_1 = merge([dense_3_2_conv2, dense_3_1_conv2], mode='concat', concat_axis=-1)
  bn22 = BatchNormalization(axis=-1)(shortcut3_1)
  dense_3_3_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn22)
  bn23 = BatchNormalization(axis=-1)(dense_3_3_conv1)
  dense_3_3_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn23)

  shortcut3_2 = merge([dense_3_3_conv2, dense_3_1_conv2, dense_3_2_conv2], mode='concat', concat_axis=-1)
  bn24 = BatchNormalization(axis=-1)(shortcut3_2)
  dense_3_4_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn24)
  bn25 = BatchNormalization(axis=-1)(dense_3_4_conv1)
  dense_3_4_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn25)

  shortcut3_3 = merge([dense_3_4_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2], mode='concat', concat_axis=-1)
  bn26 = BatchNormalization(axis=-1)(shortcut3_3)
  dense_3_5_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn26)
  bn27 = BatchNormalization(axis=-1)(dense_3_5_conv1)
  dense_3_5_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn27)

  shortcut3_4 = merge([dense_3_5_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2], mode='concat', concat_axis=-1)
  bn28 = BatchNormalization(axis=-1)(shortcut3_4)
  dense_3_6_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn28)
  bn29 = BatchNormalization(axis=-1)(dense_3_6_conv1)
  dense_3_6_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn29)

  shortcut3_5 = merge([dense_3_6_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2], mode='concat', concat_axis=-1)
  bn30 = BatchNormalization(axis=-1)(shortcut3_5)
  dense_3_7_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn30)
  bn31 = BatchNormalization(axis=-1)(dense_3_7_conv1)
  dense_3_7_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn31)

  shortcut3_6 = merge([dense_3_7_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2], mode='concat', concat_axis=-1)
  bn32 = BatchNormalization(axis=-1)(shortcut3_6)
  dense_3_8_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn32)
  bn33 = BatchNormalization(axis=-1)(dense_3_8_conv1)
  dense_3_8_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn33)

  shortcut3_7 = merge([dense_3_8_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2, dense_3_7_conv2], mode='concat', concat_axis=-1)
  bn34 = BatchNormalization(axis=-1)(shortcut3_7)
  dense_3_9_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn34)
  bn35 = BatchNormalization(axis=-1)(dense_3_9_conv1)
  dense_3_9_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn35)

  shortcut3_8 = merge([dense_3_9_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2, dense_3_7_conv2, dense_3_8_conv2], mode='concat', concat_axis=-1)
  bn36 = BatchNormalization(axis=-1)(shortcut3_8)
  dense_3_10_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn36)
  bn37 = BatchNormalization(axis=-1)(dense_3_10_conv1)
  dense_3_10_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn37)

  shortcut3_9 = merge([dense_3_10_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2, dense_3_7_conv2, dense_3_8_conv2, dense_3_9_conv2], mode='concat', concat_axis=-1)
  bn38 = BatchNormalization(axis=-1)(shortcut3_9)
  dense_3_11_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn38)
  bn39 = BatchNormalization(axis=-1)(dense_3_11_conv1)
  dense_3_11_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn39)

  shortcut3_10 = merge([dense_3_11_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2, dense_3_7_conv2, dense_3_8_conv2, dense_3_9_conv2, dense_3_10_conv2], mode='concat', concat_axis=-1)
  bn40 = BatchNormalization(axis=-1)(shortcut3_10)
  dense_3_12_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn40)
  bn41 = BatchNormalization(axis=-1)(dense_3_12_conv1)
  dense_3_12_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn41)

  # Transition layer 3
  c_channel3 = int((c_channel2 + 12 * k) * c)
  bn42 = BatchNormalization(axis=-1)(dense_3_12_conv2)
  transition3 = Conv2D(filters=c_channel3, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn42)
  pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(transition3)

  # Dense Block 4
  bn43 = BatchNormalization(axis=-1)(pool3)
  dense_4_1_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn43)
  bn44 = BatchNormalization(axis=-1)(dense_4_1_conv1)
  dense_4_1_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn44)

  bn45 = BatchNormalization(axis=-1)(dense_4_1_conv2)
  dense_4_2_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn45)
  bn46 = BatchNormalization(axis=-1)(dense_4_2_conv1)
  dense_4_2_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn46)

  shortcut4_1 = merge([dense_4_2_conv2, dense_4_1_conv2], mode='concat', concat_axis=-1)
  bn47 = BatchNormalization(axis=-1)(shortcut4_1)
  dense_4_3_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn47)
  bn48 = BatchNormalization(axis=-1)(dense_4_3_conv1)
  dense_4_3_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn48)

  shortcut4_2 = merge([dense_4_3_conv2, dense_4_1_conv2, dense_4_2_conv2], mode='concat', concat_axis=-1)
  bn49 = BatchNormalization(axis=-1)(shortcut4_2)
  dense_4_4_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn49)
  bn50 = BatchNormalization(axis=-1)(dense_4_4_conv1)
  dense_4_4_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn50)

  shortcut4_3 = merge([dense_4_4_conv2, dense_4_1_conv2, dense_4_2_conv2, dense_4_3_conv2], mode='concat', concat_axis=-1)
  bn51 = BatchNormalization(axis=-1)(shortcut4_3)
  dense_4_5_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn51)
  bn52 = BatchNormalization(axis=-1)(dense_4_5_conv1)
  dense_4_5_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn52)

  shortcut4_4 = merge([dense_4_5_conv2, dense_4_1_conv2, dense_4_2_conv2, dense_4_3_conv2, dense_4_4_conv2], mode='concat', concat_axis=-1)
  bn53 = BatchNormalization(axis=-1)(shortcut4_4)
  dense_4_6_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn53)
  bn54 = BatchNormalization(axis=-1)(dense_4_6_conv1)
  dense_4_6_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn54)

  shortcut4_5 = merge([dense_4_6_conv2, dense_4_1_conv2, dense_4_2_conv2, dense_4_3_conv2, dense_4_4_conv2, dense_4_5_conv2], mode='concat', concat_axis=-1)
  bn55 = BatchNormalization(axis=-1)(shortcut4_5)
  dense_4_7_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn55)
  bn56 = BatchNormalization(axis=-1)(dense_4_7_conv1)
  dense_4_7_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn56)

  shortcut4_6 = merge([dense_4_7_conv2, dense_4_1_conv2, dense_4_2_conv2, dense_4_3_conv2, dense_4_4_conv2, dense_4_5_conv2, dense_4_6_conv2], mode='concat', concat_axis=-1)
  bn57 = BatchNormalization(axis=-1)(shortcut4_6)
  dense_4_8_conv1 = Conv2D(filters=(4*k), kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(bn57)
  bn58 = BatchNormalization(axis=-1)(dense_4_8_conv1)
  dense_4_8_conv2 = Conv2D(filters=k, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn58)

  # Upsampling layer 1
  up_channel1 = 504
  up1 = merge([UpSampling3D(size=(2, 2))(dense_4_8_conv2), dense_3_12_conv2], mode='concat', concat_axis=-1)
  bn59 = BatchNormalization(axis=-1)(up1)
  up1 = Conv2D(filters=up_channel1, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn59)

  # Upsampling layer 2
  up_channel2 = 224
  up2 = merge([UpSampling3D(size=(2, 2))(up1), dense_2_4_conv2], mode='concat', concat_axis=-1)
  bn60 = BatchNormalization(axis=-1)(up2)
  up2 = Conv2D(filters=up_channel2, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn60)

  # Upsampling layer 3
  up_channel3 = 192
  up3 = merge([UpSampling3D(size=(2, 2))(up2), dense_1_3_conv2], mode='concat', concat_axis=-1)
  bn61 = BatchNormalization(axis=-1)(up3)
  up3 = Conv2D(filters=up_channel3, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn61)

  # Upsampling layer 4
  up_channel4 = 96
  up4 = merge([UpSampling3D(size=(2, 2, 2))(up3), conv1], mode='concat', concat_axis=-1)
  bn62 = BatchNormalization(axis=-1)(up4)
  up4 = Conv2D(filters=up_channel4, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn62)

  # Upsampling layer 5
  up_channel5 = 64
  up5 = UpSampling3D(size=(2, 2, 2))(up4)
  bn63 = BatchNormalization(axis=-1)(up5)
  up5 = Conv2D(filters=up_channel5, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(bn63)

  # Convolution 2
  conv2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(up5)

  model = Model(input=inputs, output=conv2)
  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model
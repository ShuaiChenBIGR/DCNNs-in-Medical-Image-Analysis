

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm

################
# Getting 3D DenseUNet:


def get_3d_denseunet():
  
  k = 32
  c = 0.5
  
  # Input
  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  # Convolution 1
  channel1 = (3*k)
  conv1 = Conv3D(filters=channel1, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(inputs)

  # Pooling
  pool0 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(conv1)

  # Dense Block 1
  bn2 = BatchNormalization(axis=-1)(pool0)
  act2 = Activation('relu')(bn2)
  dense_1_1_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act2)
  bn3 = BatchNormalization(axis=-1)(dense_1_1_conv1)
  act3 = Activation('relu')(bn3)
  dense_1_1_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act3)

  bn4 = BatchNormalization(axis=-1)(dense_1_1_conv2)
  act4 = Activation('relu')(bn4)
  dense_1_2_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act4)
  bn5 = BatchNormalization(axis=-1)(dense_1_2_conv1)
  act5 = Activation('relu')(bn5)
  dense_1_2_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act5)

  shortcut1_1 = merge([dense_1_1_conv2, dense_1_2_conv2], mode='concat', concat_axis=-1)
  bn6 = BatchNormalization(axis=-1)(shortcut1_1)
  act6 = Activation('relu')(bn6)
  dense_1_3_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act6)
  bn7 = BatchNormalization(axis=-1)(dense_1_3_conv1)
  act7 = Activation('relu')(bn7)
  dense_1_3_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act7)

  # Transition layer 1
  c_channel1 = int((3 * k + 3 * k) * c)
  bn8 = BatchNormalization(axis=-1)(dense_1_3_conv2)
  act8 = Activation('relu')(bn8)
  transition1 = Conv3D(filters=c_channel1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act8)
  pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(transition1)

  # Dense Block 2
  bn9 = BatchNormalization(axis=-1)(pool1)
  act9 = Activation('relu')(bn9)
  dense_2_1_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act9)
  bn10 = BatchNormalization(axis=-1)(dense_2_1_conv1)
  act10 = Activation('relu')(bn10)
  dense_2_1_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act10)

  bn11 = BatchNormalization(axis=-1)(dense_2_1_conv2)
  act11 = Activation('relu')(bn11)
  dense_2_2_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act11)
  bn12 = BatchNormalization(axis=-1)(dense_2_2_conv1)
  act12 = Activation('relu')(bn12)
  dense_2_2_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act12)

  shortcut2_1 = merge([dense_2_1_conv2, dense_2_2_conv2], mode='concat', concat_axis=-1)
  bn13 = BatchNormalization(axis=-1)(shortcut2_1)
  act13 = Activation('relu')(bn13)
  dense_2_3_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act13)
  bn14 = BatchNormalization(axis=-1)(dense_2_3_conv1)
  act14 = Activation('relu')(bn14)
  dense_2_3_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act14)

  shortcut2_2 = merge([dense_2_1_conv2, dense_2_2_conv2, dense_2_3_conv2], mode='concat', concat_axis=-1)
  bn15 = BatchNormalization(axis=-1)(shortcut2_2)
  act15 = Activation('relu')(bn15)
  dense_2_4_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act15)
  bn16 = BatchNormalization(axis=-1)(dense_2_4_conv1)
  act16 = Activation('relu')(bn16)
  dense_2_4_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act16)

  # Transition layer 2
  c_channel2 = int((c_channel1 + 4 * k) * c)
  bn17 = BatchNormalization(axis=-1)(dense_2_4_conv2)
  act17 = Activation('relu')(bn17)
  transition2 = Conv3D(filters=c_channel2, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act17)
  pool2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(transition2)

  # Dense Block 3
  bn18 = BatchNormalization(axis=-1)(pool2)
  act18 = Activation('relu')(bn18)
  dense_3_1_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act18)
  bn19 = BatchNormalization(axis=-1)(dense_3_1_conv1)
  act19 = Activation('relu')(bn19)
  dense_3_1_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act19)

  bn20 = BatchNormalization(axis=-1)(dense_3_1_conv2)
  act20 = Activation('relu')(bn20)
  dense_3_2_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act20)
  bn21 = BatchNormalization(axis=-1)(dense_3_2_conv1)
  act21 = Activation('relu')(bn21)
  dense_3_2_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act21)

  shortcut3_1 = merge([dense_3_2_conv2, dense_3_1_conv2], mode='concat', concat_axis=-1)
  bn22 = BatchNormalization(axis=-1)(shortcut3_1)
  act22 = Activation('relu')(bn22)
  dense_3_3_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act22)
  bn23 = BatchNormalization(axis=-1)(dense_3_3_conv1)
  act23 = Activation('relu')(bn23)
  dense_3_3_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act23)

  shortcut3_2 = merge([dense_3_3_conv2, dense_3_1_conv2, dense_3_2_conv2], mode='concat', concat_axis=-1)
  bn24 = BatchNormalization(axis=-1)(shortcut3_2)
  act24 = Activation('relu')(bn24)
  dense_3_4_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act24)
  bn25 = BatchNormalization(axis=-1)(dense_3_4_conv1)
  act25 = Activation('relu')(bn25)
  dense_3_4_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act25)

  shortcut3_3 = merge([dense_3_4_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2], mode='concat', concat_axis=-1)
  bn26 = BatchNormalization(axis=-1)(shortcut3_3)
  act26 = Activation('relu')(bn26)
  dense_3_5_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act26)
  bn27 = BatchNormalization(axis=-1)(dense_3_5_conv1)
  act27 = Activation('relu')(bn27)
  dense_3_5_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act27)

  shortcut3_4 = merge([dense_3_5_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2], mode='concat', concat_axis=-1)
  bn28 = BatchNormalization(axis=-1)(shortcut3_4)
  act28 = Activation('relu')(bn28)
  dense_3_6_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act28)
  bn29 = BatchNormalization(axis=-1)(dense_3_6_conv1)
  act29 = Activation('relu')(bn29)
  dense_3_6_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act29)

  shortcut3_5 = merge([dense_3_6_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2], mode='concat', concat_axis=-1)
  bn30 = BatchNormalization(axis=-1)(shortcut3_5)
  act30 = Activation('relu')(bn30)
  dense_3_7_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act30)
  bn31 = BatchNormalization(axis=-1)(dense_3_7_conv1)
  act31 = Activation('relu')(bn31)
  dense_3_7_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act31)

  shortcut3_6 = merge([dense_3_7_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2], mode='concat', concat_axis=-1)
  bn32 = BatchNormalization(axis=-1)(shortcut3_6)
  act32 = Activation('relu')(bn32)
  dense_3_8_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act32)
  bn33 = BatchNormalization(axis=-1)(dense_3_8_conv1)
  act33 = Activation('relu')(bn33)
  dense_3_8_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act33)

  shortcut3_7 = merge([dense_3_8_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2, dense_3_7_conv2], mode='concat', concat_axis=-1)
  bn34 = BatchNormalization(axis=-1)(shortcut3_7)
  act34 = Activation('relu')(bn34)
  dense_3_9_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act34)
  bn35 = BatchNormalization(axis=-1)(dense_3_9_conv1)
  act35 = Activation('relu')(bn35)
  dense_3_9_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act35)

  shortcut3_8 = merge([dense_3_9_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2, dense_3_7_conv2, dense_3_8_conv2], mode='concat', concat_axis=-1)
  bn36 = BatchNormalization(axis=-1)(shortcut3_8)
  act36 = Activation('relu')(bn36)
  dense_3_10_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act36)
  bn37 = BatchNormalization(axis=-1)(dense_3_10_conv1)
  act37 = Activation('relu')(bn37)
  dense_3_10_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act37)

  shortcut3_9 = merge([dense_3_10_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2, dense_3_7_conv2, dense_3_8_conv2, dense_3_9_conv2], mode='concat', concat_axis=-1)
  bn38 = BatchNormalization(axis=-1)(shortcut3_9)
  act38 = Activation('relu')(bn38)
  dense_3_11_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act38)
  bn39 = BatchNormalization(axis=-1)(dense_3_11_conv1)
  act39 = Activation('relu')(bn39)
  dense_3_11_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act39)

  shortcut3_10 = merge([dense_3_11_conv2, dense_3_1_conv2, dense_3_2_conv2, dense_3_3_conv2, dense_3_4_conv2, dense_3_5_conv2, dense_3_6_conv2, dense_3_7_conv2, dense_3_8_conv2, dense_3_9_conv2, dense_3_10_conv2], mode='concat', concat_axis=-1)
  bn40 = BatchNormalization(axis=-1)(shortcut3_10)
  act40 = Activation('relu')(bn40)
  dense_3_12_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act40)
  bn41 = BatchNormalization(axis=-1)(dense_3_12_conv1)
  act41 = Activation('relu')(bn41)
  dense_3_12_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act41)

  # Transition layer 3
  c_channel3 = int((c_channel2 + 12 * k) * c)
  bn42 = BatchNormalization(axis=-1)(dense_3_12_conv2)
  act42 = Activation('relu')(bn42)
  transition3 = Conv3D(filters=c_channel3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act42)
  pool3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(transition3)

  # Dense Block 4
  bn43 = BatchNormalization(axis=-1)(pool3)
  act43 = Activation('relu')(bn43)
  dense_4_1_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act43)
  bn44 = BatchNormalization(axis=-1)(dense_4_1_conv1)
  act44 = Activation('relu')(bn44)
  dense_4_1_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act44)

  bn45 = BatchNormalization(axis=-1)(dense_4_1_conv2)
  act45 = Activation('relu')(bn45)
  dense_4_2_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act45)
  bn46 = BatchNormalization(axis=-1)(dense_4_2_conv1)
  act46 = Activation('relu')(bn46)
  dense_4_2_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act46)

  shortcut4_1 = merge([dense_4_2_conv2, dense_4_1_conv2], mode='concat', concat_axis=-1)
  bn47 = BatchNormalization(axis=-1)(shortcut4_1)
  act47 = Activation('relu')(bn47)
  dense_4_3_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act47)
  bn48 = BatchNormalization(axis=-1)(dense_4_3_conv1)
  act48 = Activation('relu')(bn48)
  dense_4_3_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act48)

  shortcut4_2 = merge([dense_4_3_conv2, dense_4_1_conv2, dense_4_2_conv2], mode='concat', concat_axis=-1)
  bn49 = BatchNormalization(axis=-1)(shortcut4_2)
  act49 = Activation('relu')(bn49)
  dense_4_4_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act49)
  bn50 = BatchNormalization(axis=-1)(dense_4_4_conv1)
  act50 = Activation('relu')(bn50)
  dense_4_4_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act50)

  shortcut4_3 = merge([dense_4_4_conv2, dense_4_1_conv2, dense_4_2_conv2, dense_4_3_conv2], mode='concat', concat_axis=-1)
  bn51 = BatchNormalization(axis=-1)(shortcut4_3)
  act51 = Activation('relu')(bn51)
  dense_4_5_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act51)
  bn52 = BatchNormalization(axis=-1)(dense_4_5_conv1)
  act52 = Activation('relu')(bn52)
  dense_4_5_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act52)

  shortcut4_4 = merge([dense_4_5_conv2, dense_4_1_conv2, dense_4_2_conv2, dense_4_3_conv2, dense_4_4_conv2], mode='concat', concat_axis=-1)
  bn53 = BatchNormalization(axis=-1)(shortcut4_4)
  act53 = Activation('relu')(bn53)
  dense_4_6_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act53)
  bn54 = BatchNormalization(axis=-1)(dense_4_6_conv1)
  act54 = Activation('relu')(bn54)
  dense_4_6_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act54)

  shortcut4_5 = merge([dense_4_6_conv2, dense_4_1_conv2, dense_4_2_conv2, dense_4_3_conv2, dense_4_4_conv2, dense_4_5_conv2], mode='concat', concat_axis=-1)
  bn55 = BatchNormalization(axis=-1)(shortcut4_5)
  act55 = Activation('relu')(bn55)
  dense_4_7_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act55)
  bn56 = BatchNormalization(axis=-1)(dense_4_7_conv1)
  act56 = Activation('relu')(bn56)
  dense_4_7_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act56)

  shortcut4_6 = merge([dense_4_7_conv2, dense_4_1_conv2, dense_4_2_conv2, dense_4_3_conv2, dense_4_4_conv2, dense_4_5_conv2, dense_4_6_conv2], mode='concat', concat_axis=-1)
  bn57 = BatchNormalization(axis=-1)(shortcut4_6)
  act57 = Activation('relu')(bn57)
  dense_4_8_conv1 = Conv3D(filters=(4*k), kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(act57)
  bn58 = BatchNormalization(axis=-1)(dense_4_8_conv1)
  act58 = Activation('relu')(bn58)
  dense_4_8_conv2 = Conv3D(filters=k, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act58)

  # Upsampling layer 1
  up_channel1 = 504
  up1 = merge([UpSampling3D(size=(1, 2, 2))(dense_4_8_conv2), dense_3_12_conv2], mode='concat', concat_axis=-1)
  bn59 = BatchNormalization(axis=-1)(up1)
  act59 = Activation('relu')(bn59)
  up1 = Conv3D(filters=up_channel1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act59)

  # Upsampling layer 2
  up_channel2 = 224
  up2 = merge([UpSampling3D(size=(1, 2, 2))(up1), dense_2_4_conv2], mode='concat', concat_axis=-1)
  bn60 = BatchNormalization(axis=-1)(up2)
  act60 = Activation('relu')(bn60)
  up2 = Conv3D(filters=up_channel2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act60)

  # Upsampling layer 3
  up_channel3 = 192
  up3 = merge([UpSampling3D(size=(1, 2, 2))(up2), dense_1_3_conv2], mode='concat', concat_axis=-1)
  bn61 = BatchNormalization(axis=-1)(up3)
  act61 = Activation('relu')(bn61)
  up3 = Conv3D(filters=up_channel3, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act61)

  # Upsampling layer 4
  up_channel4 = 96
  up4 = merge([UpSampling3D(size=(2, 2, 2))(up3), conv1], mode='concat', concat_axis=-1)
  bn62 = BatchNormalization(axis=-1)(up4)
  act62 = Activation('relu')(bn62)
  up4 = Conv3D(filters=up_channel4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act62)

  # Upsampling layer 5
  up_channel5 = 64
  up5 = UpSampling3D(size=(2, 2, 2))(up4)
  bn63 = BatchNormalization(axis=-1)(up5)
  act63 = Activation('relu')(bn63)
  up5 = Conv3D(filters=up_channel5, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(act63)

  # Convolution 2
  conv2 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(up5)

  model = Model(input=inputs, output=conv2)
  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model

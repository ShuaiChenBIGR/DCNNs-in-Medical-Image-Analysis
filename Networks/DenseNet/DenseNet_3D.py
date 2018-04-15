

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm

################
# Getting 3D DenseNet:


def get_3d_densenet():

  k = 32
  # Input
  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))

  # Convolution 1
  BN1 = BatchNormalization(axis=-1)(inputs)
  conv1 = Conv3D(filters=(2*k), kernel_size=(7, 7, 7), strides=(2, 2, 2), activation='relu', padding='same')(BN1)









  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(up1)

  model = Model(input=inputs, output=conv10)
  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model
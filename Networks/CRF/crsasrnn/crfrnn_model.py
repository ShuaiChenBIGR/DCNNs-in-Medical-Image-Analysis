"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from keras.models import Model
from keras.optimizers import Adam, Adadelta
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add
from keras.layers import Conv3D, MaxPooling3D, merge, UpSampling3D, ZeroPadding3D, \
    Conv3DTranspose, Cropping3D, BatchNormalization
from Modules.Networks.CRF.crsasrnn.crfrnn_layer import CrfRnnLayer_3d, CrfRnnLayer_2d
import Modules.Common_modules as cm
import Modules.LossFunction as lf
import numpy as np
import tensorflow as tf


def get_2d_crfrnn_model_def():
    """ Returns Keras CRN-RNN model definition.

    Currently, only 500 x 500 images are supported. However, one can get this to
    work with different image sizes by adjusting the parameters of the Cropping2D layers
    below.
    """

    channels, height, weight = 3, 500, 500

    # Input
    input_shape = (height, weight, 3)
    img_input = Input(shape=input_shape)

    # Add plenty of zero padding
    x = ZeroPadding2D(padding=(100, 100))(img_input)

    # VGG-16 convolution block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # VGG-16 convolution block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', padding='same')(x)

    # VGG-16 convolution block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', padding='same')(x)
    pool3 = x

    # VGG-16 convolution block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', padding='same')(x)
    pool4 = x

    # VGG-16 convolution block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5', padding='same')(x)

    # Fully-connected layers converted to convolution layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(21, (1, 1), padding='valid', name='score-fr')(x)

    # Deconvolution
    score2 = Conv2DTranspose(21, (4, 4), strides=2, name='score2')(x)

    # Skip connections from pool4
    score_pool4 = Conv2D(21, (1, 1), name='score-pool4')(pool4)
    score_pool4c = Cropping2D((5, 5))(score_pool4)
    score_fused = Add()([score2, score_pool4c])
    score4 = Conv2DTranspose(21, (4, 4), strides=2, name='score4', use_bias=False)(score_fused)

    # Skip connections from pool3
    score_pool3 = Conv2D(21, (1, 1), name='score-pool3')(pool3)
    score_pool3c = Cropping2D((9, 9))(score_pool3)

    # Fuse things together
    score_final = Add()([score4, score_pool3c])

    # Final up-sampling and cropping
    upsample = Conv2DTranspose(21, (16, 16), strides=8, name='upsample', use_bias=False)(score_final)
    upscore = Cropping2D(((31, 37), (31, 37)))(upsample)

    output = CrfRnnLayer_2d(image_dims=(height, weight),
                         num_classes=21,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([upscore, img_input])

    # Build the model
    model = Model(img_input, output, name='crfrnn_net')

    return model


def get_3d_crfrnn_model_def():
    channels, slices, height, weight = 1, cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d

    # Input
    inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1), name='layer_no_0_input')

    conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_1_conv')(inputs)
    conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_2_conv')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), name='layer_no_3')(conv1)

    conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_4_conv')(pool1)
    conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_5_conv')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), name='layer_no_6')(conv2)

    conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_7_conv')(pool2)
    conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_8_conv')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), name='layer_no_9')(conv3)

    conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_10_conv')(pool3)
    conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_11_conv')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), name='layer_no_12')(conv4)

    conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_13_conv')(pool4)
    conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_14_conv')(conv5)

    up6 = merge([UpSampling3D(size=(2, 2, 2), name='layer_no_15')(conv5), conv4], mode='concat', concat_axis=-1,
                name='layer_no_16')
    conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_17_conv')(up6)
    conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_18_conv')(conv6)

    up7 = merge([UpSampling3D(size=(2, 2, 2), name='layer_no_19')(conv6), conv3], mode='concat', concat_axis=-1,
                name='layer_no_20')
    conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_21_conv')(up7)
    conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_22_conv')(conv7)

    up8 = merge([UpSampling3D(size=(2, 2, 2), name='layer_no_23')(conv7), conv2], mode='concat', concat_axis=-1,
                name='layer_no_24')
    conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_25_conv')(up8)
    conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_26_conv')(conv8)

    up9 = merge([UpSampling3D(size=(2, 2, 2), name='layer_no_27')(conv8), conv1], mode='concat', concat_axis=-1,
                name='layer_no_28')
    conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_29_conv')(up9)
    conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same',
                   name='layer_no_30_last')(conv9)

    conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid',
                    name='layer_no_31_output')(conv9)

    output = CrfRnnLayer_3d(image_dims=(slices, height, weight),
                            num_classes=3,
                            theta_alpha=160.,
                            theta_beta=3.,
                            theta_gamma=3.,
                            num_iterations=1,
                            name='crfrnn')([conv10, inputs])

    weights = np.array([1.0, 100.0, 100.0])
    loss = lf.weighted_categorical_crossentropy_loss(weights)

    # Build the model
    model = Model(inputs, conv10, name='crfrnn_UNet')

    # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    model.compile(optimizer=Adam(lr=1.0e-5), loss=loss, metrics=["categorical_accuracy"])


    return model

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, BatchNormalization
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm
from Modules.Networks.CRF.crsasrnn.crfrnn_layer import CrfRnnLayer_2d


########################################################
# Getting 2D U-net:

def get_2d_unet():
    inputs = Input((cm.img_rows_2d, cm.img_cols_2d, 1))
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=["accuracy"])
    # model.compile(optimizer=opti, loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])
    return model


def get_2d_unet_crf():
    inputs = Input((cm.img_rows_2d, cm.img_cols_2d, 1))
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    output = CrfRnnLayer_2d(image_dims=(512, 512),
                            num_classes=2,
                            theta_alpha=160.,
                            theta_beta=3.,
                            theta_gamma=3.,
                            num_iterations=10,
                            name='crfrnn')([conv10, inputs])

    model = Model(input=inputs, output=output)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=["accuracy"])
    # model.compile(optimizer=opti, loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])
    return model


def get_shallow_unet(sgd):
    inputs = Input((cm.img_rows, cm.img_cols, 1))

    conv3 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(inputs)
    conv3 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
    conv7 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv7)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv7)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=sgd, loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

    return model


def get_dropout_unet():
    inputs = Input((cm.cm.img_rows, cm.cm.img_cols, 1))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    drop1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(drop1)
    drop1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    drop2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(drop2)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    drop3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(drop3)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    drop4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(drop4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    drop5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(drop5)
    drop5 = Dropout(0.2)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(drop5), drop4], mode='concat', concat_axis=-1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    drop6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(drop6)
    drop6 = Dropout(0.2)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(drop6), drop3], mode='concat', concat_axis=-1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    drop7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(drop7)
    drop7 = Dropout(0.2)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(drop7), drop2], mode='concat', concat_axis=-1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    drop8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(drop8)
    drop8 = Dropout(0.2)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(drop8), drop1], mode='concat', concat_axis=-1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    drop9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(drop9)
    drop9 = Dropout(0.2)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(drop9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

    return model


def get_unet_less_feature():
    inputs = Input((cm.img_rows, cm.img_cols, 1))
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv4)

    up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=-1)
    conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
    conv9 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

    return model


def get_unet_more_feature():
    inputs = Input((cm.img_rows, cm.img_cols, 1))
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

    return model


def get_unet_dilated_conv_7():
    inputs = Input((cm.img_rows, cm.img_cols, 1))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dilation_rate=(1, 1))(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dilation_rate=(1, 1))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dilation_rate=(2, 2))(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dilation_rate=(2, 2))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dilation_rate=(4, 4))(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dilation_rate=(4, 4))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dilation_rate=(8, 8))(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', dilation_rate=(8, 8))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', dilation_rate=(16, 16))(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', dilation_rate=(16, 16))(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', dilation_rate=(32, 32))(pool5)
    conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', dilation_rate=(32, 32))(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Convolution2D(2048, 3, 3, activation='relu', border_mode='same', dilation_rate=(64, 64))(pool6)
    conv7 = Convolution2D(2048, 3, 3, activation='relu', border_mode='same', dilation_rate=(64, 64))(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv6], mode='concat', concat_axis=-1)
    conv8 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv5], mode='concat', concat_axis=-1)
    conv9 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv9)

    up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv4], mode='concat', concat_axis=-1)
    conv10 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up10)
    conv10 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv10)

    up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv3], mode='concat', concat_axis=-1)
    conv11 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up11)
    conv11 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv11)

    up12 = merge([UpSampling2D(size=(2, 2))(conv11), conv2], mode='concat', concat_axis=-1)
    conv12 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up12)
    conv12 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv12)

    up13 = merge([UpSampling2D(size=(2, 2))(conv12), conv1], mode='concat', concat_axis=-1)
    conv13 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up13)
    conv13 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv13)

    conv14 = Convolution2D(1, 1, 1, activation='sigmoid', dilation_rate=(1, 1))(conv13)

    model = Model(input=inputs, output=conv14)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1.0e-5), loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

    return model


def get_unet_dilated_conv_4():
    inputs = Input((cm.img_rows, cm.img_cols, 1))
    conv1 = Convolution2D(filters=16, kernel_size=[3, 3], padding='same', activation='relu', dilation_rate=1)(inputs)
    conv1 = Convolution2D(filters=16, kernel_size=[3, 3], padding='same', activation='relu', dilation_rate=1)(conv1)

    conv2 = Convolution2D(filters=32, kernel_size=[3, 3], padding='same', activation='relu', dilation_rate=2)(conv1)
    conv2 = Convolution2D(filters=32, kernel_size=[3, 3], padding='same', activation='relu', dilation_rate=4)(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu', dilation_rate=8)(conv2)
    conv3 = Convolution2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu', dilation_rate=16)(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu', dilation_rate=16)(conv3)
    conv4 = Convolution2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu', dilation_rate=64)(conv4)
    #
    # conv5 = Convolution2D(filters=512,kernel_size=[3,3], padding='same', activation='relu', dilation_rate=1)(conv4)
    # conv5 = Convolution2D(filters=512,kernel_size=[3,3], padding='same', activation='relu', dilation_rate=1)(conv5)
    #

    conv10 = Convolution2D(1, [1, 1], activation='sigmoid', dilation_rate=1)(conv4)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer='adadelta', loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

    return model


def get_simple_unet(opti):
    inputs = Input((cm.img_rows, cm.img_cols, 1))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv1)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=opti, loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

    return model


def get_2D_Deeply_supervised_network():
    inputs = Input((cm.img_rows, cm.img_cols, 1))

    conv1 = Convolution2D(8, kernel_size=(9, 9), activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(8, kernel_size=(9, 9), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    deconv1 = Deconvolution2D(8, kernel_size=(3, 3), strides=(2, 2), padding="same")(pool1)
    prediction1 = Convolution2D(1, 1, 1, activation='sigmoid')(deconv1)

    conv2 = Convolution2D(16, kernel_size=(7, 7), activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(32, kernel_size=(7, 7), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    deconv2 = Deconvolution2D(8, kernel_size=(3, 3), strides=(2, 2), padding="same")(pool2)
    deconv2 = Deconvolution2D(8, kernel_size=(3, 3), strides=(2, 2), padding="same")(deconv2)
    prediction2 = Convolution2D(1, 1, 1, activation='sigmoid')(deconv2)

    conv3 = Convolution2D(32, kernel_size=(5, 5), activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(32, kernel_size=(1, 1), activation='relu', border_mode='same')(conv3)
    deconv3 = Deconvolution2D(8, kernel_size=(3, 3), strides=(2, 2), padding="same")(conv3)
    deconv3 = Deconvolution2D(8, kernel_size=(3, 3), strides=(2, 2), padding="same")(deconv3)
    prediction3 = Convolution2D(1, 1, 1, activation='sigmoid')(deconv3)

    model1 = Model(input=inputs, output=prediction1)
    model2 = Model(input=inputs, output=prediction2)
    model3 = Model(input=inputs, output=prediction3)

    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model1.compile(optimizer=Adam(lr=1.0e-6), loss=lf.binary_crossentropy_loss, metrics=[lf.binary_crossentropy])

    return model1

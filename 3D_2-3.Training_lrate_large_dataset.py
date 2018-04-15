#
# 3D Aorta and Pulmonary Segmentation
#
# 2. 3D training
#
# created by
# Shuai Chen
# PhD student
# Medical Informatics
#
# P.O. Box 2040, 3000 CA Rotterdam, The Netherlands, internal postal address Na-2603
# Visiting address: office 2616, Wytemaweg 80, 3015 CN Rotterdam, The Netherlands
# Email s.chen.2@erasmusmc.nl | Telephone +31 6 334 516 99
# www.erasmusmc.nl
#
# created on 09/03/2018
# Last update: 09/03/2018
########################################################################################

from __future__ import print_function

from glob import glob
import Modules.Common_modules as cm
import Modules.Callbacks as cb
import Modules.DataProcess as dp
import Networks.DenseUNet_3D as DenseUNet_3D
import Networks.RSUNet_3D as RSUNet_3D
import Networks.UNet_3D as UNet_3D
import Networks.RSUNet_3D_Gerda as RSUNet_3D_Gerda
import numpy as np
import datetime
from keras import callbacks
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import sys

# np.random.seed(0)

def train_and_predict(use_existing):

  cm.mkdir(cm.workingPath.model_path)
  cm.mkdir(cm.workingPath.best_model_path)
  cm.mkdir(cm.workingPath.visual_path)

  # learning_rate = 0.00001
  #
  # adam = Adam(lr=learning_rate)
  #
  # opti = adam
  #
  # lrate = callbacks.LearningRateScheduler(cb.step_decay)

  print('-' * 30)
  print('Loading and preprocessing train data...')
  print('-' * 30)

  # Scanning training data list:
  originFile_list = sorted(glob(cm.workingPath.trainingPatchesSet_path + 'img_*.npy'))
  mask_list = sorted(glob(cm.workingPath.trainingPatchesSet_path + 'mask_*.npy'))

  # Scanning validation data list:
  originValFile_list = sorted(glob(cm.workingPath.validationSet_path + 'valImages.npy'))
  maskVal_list = sorted(glob(cm.workingPath.validationSet_path + 'valMasks.npy'))

  x_val = np.load(originValFile_list[0])
  y_val = np.load(maskVal_list[0])

  # Calculate the total amount of training sets:
  nb_file = int(len(originFile_list))
  nb_val_file = int(len(originValFile_list))

  # Make a random list (shuffle the training data):
  random_scale = nb_file
  rand_i = np.random.choice(range(random_scale), size=random_scale, replace=False)

  # train_num, val_num, train_list, val_list = dp.train_split(nb_file, rand_i)
  train_num, train_list = dp.train_val_split(nb_file, rand_i)

  print('_' * 30)
  print('Creating and compiling model...')
  print('_' * 30)

  # Select the model you want to train:
  # model = nw.get_3D_unet()
  # model = nw.get_3D_Eunet()
  # model = DenseUNet_3D.get_3d_denseunet()
  model = UNet_3D.get_3d_unet()
  # model = RSUNet_3D.get_3d_rsunet(opti)
  # model = RSUNet_3D_Gerda.get_3d_rsunet_Gerdafeature(opti)

  # Plot the model:
  modelname = 'model.png'
  plot_model(model, show_shapes=True, to_file=cm.workingPath.model_path + modelname)
  model.summary()

  # Should we load existing weights?
  if use_existing:
    model.load_weights(cm.workingPath.model_path + './unet.hdf5')

  print('-' * 30)
  print('Fitting model...')
  print('-' * 30)

  nb_epoch = 4000

  temp_weights = model.get_weights()

  for e in range(nb_epoch):

    # Set callbacks:
    filepath = cm.workingPath.model_path + 'weights.epoch_%02d-{loss:.5f}-{val_loss:.5f}.hdf5' % (e+1)
    model_checkpoint = callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False)
    record_history = cb.RecordLossHistory()

    for i in range(train_num):

      print("epoch %04d, batch %04d" % (e+1, i+1))
      x_train, y_train = dp.BatchGenerator(i, originFile_list, mask_list, train_list)

      # gradients = cb.recordGradients_Florian(x_train, cm.workingPath.model_path, model, True)
      callbacks_list = [record_history, model_checkpoint]

      if i == (train_num-1):

        model.set_weights(temp_weights)
        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1, validation_data=(x_val, y_val),
                  callbacks=callbacks_list)
        temp_weights = model.get_weights()

      else:

        model.set_weights(temp_weights)
        model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=1)
        temp_weights = model.get_weights()

  print('training finished')


if __name__ == '__main__':
  # Choose whether to train based on the last model:
  # Show runtime:
  starttime = datetime.datetime.now()

  # train_and_predict(True)
  train_and_predict(False)

  endtime = datetime.datetime.now()
  print(endtime - starttime)


  sys.exit(0)
#
# 3D Aorta Segmentation
#
# 3-2-4. Test Multi Class patches
#
# created by
# Shuai Chen
# PhD student
# Radiology and Medical Informatics
#
# P.O. Box 2040, 3000 CA Rotterdam, The Netherlands, internal postal address Na-2603
# Visiting address: office 2616, Wytemaweg 80, 3015 CN Rotterdam, The Netherlands
# Email s.chen.2@erasmusmc.nl
# www.erasmusmc.nl
#
# created on 09/03/2018
# Last update: 09/03/2018
########################################################################################

from __future__ import print_function

import datetime
import re
import sys
from glob import glob

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import keras.backend as K
from sklearn.metrics import roc_curve, auc

import Modules.Common_modules as cm
import Modules.DataProcess as dp
import Modules.LossFunction as lf
import Modules.Visualization as vs
import Modules.Metrics as mt
import Modules.Networks.UNet.UNet_3D as UNet_3D
import Modules.Networks.CNN.CNN_3D as CNN_3D

def model_test(use_existing):

  cm.mkdir(cm.workingPath.testingResults_path)
  cm.mkdir(cm.workingPath.testingNPY_path)

  # Loading test data:

  filename = cm.filename
  modelname = cm.modellist[0]

  # Single CT:
  originFile_list = sorted(glob(cm.workingPath.originTestingSet_path + filename))
  maskAortaFile_list = sorted(glob(cm.workingPath.aortaTestingSet_path + filename))
  maskPulFile_list = sorted(glob(cm.workingPath.pulTestingSet_path + filename))

  # Zahra CTs:
  # originFile_list = sorted(glob(cm.workingPath.originTestingSet_path + "vol*.dcm"))
  # maskAortaFile_list = sorted(glob(cm.workingPath.aortaTestingSet_path + "vol*.dcm"))
  # maskPulFile_list = sorted(glob(cm.workingPath.pulTestingSet_path + "vol*.dcm"))

  # Lidia CTs:
  # originFile_list = sorted(glob(cm.workingPath.originLidiaTestingSet_path + "vol*.dcm"))[61:]
  # maskAortaFile_list = sorted(glob(cm.workingPath.originLidiaTestingSet_path + "vol*.dcm"))[61:]
  # maskPulFile_list = sorted(glob(cm.workingPath.originLidiaTestingSet_path + "vol*.dcm"))[61:]

  # Abnormal CTs:
  # originFile_list = sorted(glob(cm.workingPath.originAbnormalTestingSet_path + "vol126*.dcm"))
  # maskAortaFile_list = sorted(glob(cm.workingPath.originAbnormalTestingSet_path + "vol126*.dcm"))
  # maskPulFile_list = sorted(glob(cm.workingPath.originAbnormalTestingSet_path + "vol126*.dcm"))

  for i in range(len(originFile_list)):

    # Show runtime:
    starttime = datetime.datetime.now()

    vol_slices = []
    out_test_images = []
    out_test_masks = []

    current_file = originFile_list[i].split('/')[-1]
    current_dir = cm.workingPath.testingResults_path + str(current_file[:-17])
    cm.mkdir(current_dir)
    cm.mkdir(current_dir + '/Plots/')
    cm.mkdir(current_dir + '/Surface_Distance/Aorta/')
    cm.mkdir(current_dir + '/Surface_Distance/Pul/')
    cm.mkdir(current_dir + '/DICOM/')
    cm.mkdir(current_dir + '/mhd/')

    stdout_backup = sys.stdout
    log_file = open(current_dir + "/logs.txt", "w")
    sys.stdout = log_file

    print('-' * 30)
    print('Loading test data %04d/%04d...'%(i+1, len(originFile_list)))

    originVol, originVol_num, originVolwidth, originVolheight = dp.loadFile(originFile_list[i])
    maskAortaVol, maskAortaVol_num, maskAortaVolwidth, maskAortaVolheight = dp.loadFile(maskAortaFile_list[i])
    maskPulVol, maskPulVol_num, maskPulVolwidth, maskPulVolheight = dp.loadFile(maskPulFile_list[i])
    maskVol = maskAortaVol

    for j in range(len(maskAortaVol)):
      maskAortaVol[j] = np.where(maskAortaVol[j] != 0, 1, 0)
    for j in range(len(maskPulVol)):
      maskPulVol[j] = np.where(maskPulVol[j] != 0, 2, 0)

    maskVol = maskVol + maskPulVol

    for j in range(len(maskVol)):
      maskVol[j] = np.where(maskVol[j] > 2, 0, maskVol[j])
      # maskVol[j] = np.where(maskVol[j] != 0, 0, maskVol[j])

    # Make the Vessel class
    for j in range(len(maskVol)):
      maskVol[j] = np.where(maskVol[j] != 0, 1, 0)

    for i in range(originVol.shape[0]):
      img = originVol[i, :, :]

      out_test_images.append(img)
    for i in range(maskVol.shape[0]):
      img = maskVol[i, :, :]

      out_test_masks.append(img)

    vol_slices.append(originVol.shape[0])

    maskAortaVol = None
    maskPulVol = None
    maskVol = None
    originVol = None

    nb_class = 2
    outmasks_onehot = to_categorical(out_test_masks, num_classes=nb_class)
    final_test_images = np.ndarray([sum(vol_slices), 512, 512, 1], dtype=np.int16)
    final_test_masks = np.ndarray([sum(vol_slices), 512, 512, nb_class], dtype=np.int8)

    for i in range(len(out_test_images)):
      final_test_images[i, :, :, 0] = out_test_images[i]
      final_test_masks[i, :, :, :] = outmasks_onehot[i]

    outmasks_onehot = None
    out_test_masks = None
    out_test_images = None

    row = cm.img_rows_3d
    col = cm.img_cols_3d
    row_1 = int((512 - row) / 2)
    row_2 = int(512 - (512 - row) / 2)
    col_1 = int((512 - col) / 2)
    col_2 = int(512 - (512 - col) / 2)
    slices = cm.slices_3d
    gaps = cm.gaps_3d

    final_images_crop = final_test_images[:, row_1:row_2, col_1:col_2, :]
    final_masks_crop = final_test_masks[:, row_1:row_2, col_1:col_2, :]

    sitk.WriteImage(sitk.GetImageFromArray(np.uint16(final_test_masks[:, :, :, 1])), current_dir + '/DICOM/masksAortaGroundTruth.dcm')

    sitk.WriteImage(sitk.GetImageFromArray(np.uint16(final_test_masks[:, :, :, 1])), current_dir + '/mhd/masksAortaGroundTruth.mhd')

    # clear the masks for the final step:
    final_test_masks = np.where(final_test_masks == 0, 0, 0)

    num_patches = int((sum(vol_slices) - slices) / gaps)

    test_image = np.ndarray([1, slices, row, col, 1], dtype=np.int16)

    predicted_mask_volume = np.ndarray([sum(vol_slices), row, col, nb_class], dtype=np.float32)

    # model = DenseUNet_3D.get_3d_denseunet()
    # model = UNet_3D.get_3d_unet_bn()
    # model = RSUNet_3D.get_3d_rsunet(opti)
    # model = UNet_3D.get_3d_wnet(opti)
    # model = UNet_3D.get_3d_unet()
    # model = UNet_3D.get_3d_unet()
    model = CNN_3D.get_3d_cnn()
    # model = RSUNet_3D_Gerda.get_3d_rsunet_Gerdafeature(opti)

    using_start_end = 1
    start_slice = cm.start_slice
    end_slice = -1

    if use_existing:
      model.load_weights(modelname)

    for i in range(num_patches):
      count1 = i * gaps
      count2 = i * gaps + slices
      test_image[0] = final_images_crop[count1:count2]

      predicted_mask = model.predict(test_image)

      if i == int(num_patches*0.63):
        vs.visualize_activation_in_layer_one_plot_add_weights(model, test_image, current_dir)
      else:
        pass

      predicted_mask_volume[count1:count2] += predicted_mask[0, :, :, :, :]

    t = len(predicted_mask_volume)
    for i in range(0, slices, gaps):
      predicted_mask_volume[i:(i + gaps)] = predicted_mask_volume[i:(i + gaps)] / (i / gaps + 1)

    for i in range(0, slices, gaps):
      predicted_mask_volume[(t - i - gaps):(t - i)] = predicted_mask_volume[(t - i - gaps):(t - i)] / (i / gaps + 1)

    for i in range(slices, (len(predicted_mask_volume) - slices)):
      predicted_mask_volume[i] = predicted_mask_volume[i] / (slices / gaps)

    np.save(cm.workingPath.testingNPY_path + 'testImages.npy', final_images_crop)
    np.save(cm.workingPath.testingNPY_path + 'testMasks.npy', final_masks_crop)
    np.save(cm.workingPath.testingNPY_path + 'masksTestPredicted.npy', predicted_mask_volume)

    final_images_crop = None
    final_masks_crop = None
    predicted_mask_volume = None

    imgs_origin = np.load(cm.workingPath.testingNPY_path + 'testImages.npy').astype(np.int16)
    imgs_true = np.load(cm.workingPath.testingNPY_path + 'testMasks.npy').astype(np.int8)
    imgs_predict = np.load(cm.workingPath.testingNPY_path + 'masksTestPredicted.npy').astype(np.float32)
    imgs_predict_threshold = np.load(cm.workingPath.testingNPY_path + 'masksTestPredicted.npy').astype(np.float32)

    ########## ROC curve aorta

    actual = imgs_true[:, :, :, 1].reshape(-1)
    predictions = imgs_predict[:, :, :, 1].reshape(-1)
    # predictions = np.where(predictions < (0.7), 0, 1)

    false_positive_rate_aorta, true_positive_rate_aorta, thresholds_aorta = roc_curve(actual, predictions, pos_label=1)
    roc_auc_aorta = auc(false_positive_rate_aorta, true_positive_rate_aorta)
    plt.figure(1, figsize=(6, 6))
    plt.figure(1)
    plt.title('ROC of Aorta')
    plt.plot(false_positive_rate_aorta, true_positive_rate_aorta, 'b')
    label = 'AUC = %0.2f' % roc_auc_aorta
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.0, 1.0])
    plt.ylim([-0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.show()
    saveName = '/Plots/ROC_Aorta_curve.png'
    plt.savefig(current_dir + saveName)
    plt.close()

    false_positive_rate_aorta = None
    true_positive_rate_aorta = None

    imgs_predict_threshold = np.where(imgs_predict_threshold < 0.3, 0, 1)

    if using_start_end == 1:
      aortaMean = lf.dice_coef_np(imgs_predict_threshold[start_slice:end_slice, :, :, 1],
                                  imgs_true[start_slice:end_slice, :, :, 1])
    else:
      aortaMean = lf.dice_coef_np(imgs_predict_threshold[:, :, :, 1], imgs_true[:, :, :, 1])

    np.savetxt(current_dir + '/Plots/AortaDicemean.txt', np.array(aortaMean).reshape(1, ), fmt='%.5f')

    print('Model file:', modelname)
    print('-' * 30)
    print('Aorta Dice Coeff', aortaMean)
    print('-' * 30)

    # Draw the subplots of figures:

    color1 = 'gray'  # ***
    color2 = 'viridis'  # ******

    transparent1 = 1.0
    transparent2 = 0.5

    # Slice parameters:

    #################################### Aorta
    # Automatically:

    steps = 40
    slice = range(0, len(imgs_origin), steps)
    plt_row = 3
    plt_col = int(len(imgs_origin) / steps)

    plt.figure(3, figsize=(25, 12))
    plt.figure(3)

    for i in slice:
      if i == 0:
        plt_num = int(i / steps) + 1
      else:
        plt_num = int(i / steps)

      if plt_num <= plt_col:

        ax1 = plt.subplot(plt_row, plt_col, plt_num)
        title = 'slice=' + str(i)
        plt.title(title)
        ax1.imshow(imgs_origin[i, :, :, 0], cmap=color1, alpha=transparent1)
        ax1.imshow(imgs_true[i, :, :, 1], cmap=color2, alpha=transparent2)

        ax2 = plt.subplot(plt_row, plt_col, plt_num + plt_col)
        title = 'slice=' + str(i)
        plt.title(title)
        ax2.imshow(imgs_origin[i, :, :, 0], cmap=color1, alpha=transparent1)
        ax2.imshow(imgs_predict[i, :, :, 1], cmap=color2, alpha=transparent2)

        ax3 = plt.subplot(plt_row, plt_col, plt_num + 2 * plt_col)
        title = 'slice=' + str(i)
        plt.title(title)
        ax3.imshow(imgs_origin[i, :, :, 0], cmap=color1, alpha=transparent1)
        ax3.imshow(imgs_predict_threshold[i, :, :, 1], cmap=color2, alpha=transparent2)
      else:
        pass

    modelname = cm.modellist[0]

    imageName = re.findall(r'\d+\.?\d*', modelname)
    epoch_num = int(imageName[0]) + 1
    accuracy = float(np.loadtxt(current_dir + '/Plots/AortaDicemean.txt', float))

    # saveName = 'epoch_' + str(epoch_num) + '_dice_' +str(accuracy) + '.png'
    saveName = '/Plots/epoch_Aorta_%02d_dice_%.3f.png' % (epoch_num - 1, accuracy)

    plt.subplots_adjust(left=0.0, bottom=0.05, right=1.0, top=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(current_dir + saveName)
    plt.close()
    # plt.show()

    print('Images saved')
    # Save npy as dcm files:

    final_test_aorta_predicted_threshold = final_test_masks[:, :, :, 1]

    final_test_aorta_predicted_threshold[:, row_1:row_2, col_1:col_2] = imgs_predict_threshold[:, :, :, 1]

    new_imgs_dcm = sitk.GetImageFromArray(np.uint16(final_test_images + 4000))
    new_imgs_aorta_predict_dcm = sitk.GetImageFromArray(np.uint16(final_test_aorta_predicted_threshold))

    sitk.WriteImage(new_imgs_dcm, current_dir + '/DICOM/imagesPredicted.dcm')
    sitk.WriteImage(new_imgs_aorta_predict_dcm, current_dir + '/DICOM/masksAortaPredicted.dcm')

    sitk.WriteImage(new_imgs_dcm, current_dir + '/mhd/imagesPredicted.mhd')
    sitk.WriteImage(new_imgs_aorta_predict_dcm, current_dir + '/mhd/masksAortaPredicted.mhd')

    # mt.SegmentDist(current_dir + '/mhd/masksAortaPredicted.mhd',current_dir + '/mhd/masksAortaGroundTruth.mhd', current_dir + '/Surface_Distance/Aorta')
    # mt.SegmentDist(current_dir + '/mhd/masksPulPredicted.mhd',current_dir + '/mhd/masksPulGroundTruth.mhd', current_dir + '/Surface_Distance/Pul')

    print('DICOM saved')

    # Clear memory for the next testing sample:

    final_test_aorta_predicted_threshold = None
    final_test_pul_predicted_threshold = None
    imgs_predict_threshold = None
    new_imgs_dcm = None
    new_imgs_aorta_predict_dcm = None
    new_imgs_pul_predict_dcm = None
    final_test_images = None
    final_test_masks = None
    imgs_origin = None
    imgs_predict = None
    imgs_true = None
    predicted_mask = None
    predictions = None

    endtime = datetime.datetime.now()
    print('-' * 30)
    print('running time:', endtime - starttime)

    log_file.close()
    sys.stdout = stdout_backup

if __name__ == '__main__':
  # Choose whether to train based on the last model:
  model_test(True)


  sys.exit(0)

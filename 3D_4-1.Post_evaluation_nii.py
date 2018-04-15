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
import dicom
import nibabel as nib
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc

import Modules.Common_modules as cm
import Modules.DataProcess as dp
import Modules.LossFunction as lf
from Modules.FROC.froc import computeAndPlotFROC
import Modules.Visualization as vs
import Modules.Metrics as mt


def nifty_evaluation(vol_list):

  for i in range(len(vol_list)):

    aorta_GT_nifty_file = vol_list[i] + '/NIFTY/' + 'masksAortaGroundTruth.nii'
    pul_GT_nifty_file = vol_list[i] + '/NIFTY/' + 'masksPulGroundTruth.nii'

    aorta_Pred_nifty_file = vol_list[i] + '/NIFTY/' + 'masksAortaPredicted.nii'
    pul_Pred_nifty_file = vol_list[i] + '/NIFTY/' + 'masksPulPredicted.nii'

    # Show runtime:
    starttime = datetime.datetime.now()

    current_file = vol_list[i].split('/')[-2]
    current_dir = vol_list[i]

    stdout_backup = sys.stdout
    log_file = open(current_dir + "/logs_post.txt", "w")
    sys.stdout = log_file

    print('-' * 30)
    print('Start post-evaluating test data %04d/%04d...'%(i+1, len(vol_list)))

    originVol, originVol_num, originVolwidth, originVolheight = dp.loadFile(originFile_list[i])
    maskAortaVol, maskAortaVol_num, maskAortaVolwidth, maskAortaVolheight = dp.loadFile(maskAortaFile_list[i])
    maskPulVol, maskPulVol_num, maskPulVolwidth, maskPulVolheight = dp.loadFile(maskPulFile_list[i])
    maskVol = maskAortaVol

    ds = dicom.read_file(originFile_list[i])
    image_pixel_space = ds.ImagerPixelSpacing
    pixel_space = ds.PixelSpacing
    ds = None

    for j in range(len(maskAortaVol)):
      maskAortaVol[j] = np.where(maskAortaVol[j] != 0, 1, 0)
    for j in range(len(maskPulVol)):
      maskPulVol[j] = np.where(maskPulVol[j] != 0, 2, 0)

    maskVol = maskVol + maskPulVol

    for j in range(len(maskVol)):
      maskVol[j] = np.where(maskVol[j] > 2, 0, maskVol[j])
      # maskVol[j] = np.where(maskVol[j] != 0, 0, maskVol[j])

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

    nb_class = 3
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
    sitk.WriteImage(sitk.GetImageFromArray(np.uint16(final_test_masks[:, :, :, 2])), current_dir + '/DICOM/masksPulGroundTruth.dcm')

    dicom_temp = dicom.read_file(current_dir + '/DICOM/masksAortaGroundTruth.dcm')
    dicom_temp.ImagerPixelSpacing = image_pixel_space
    dicom_temp.PixelSpacing = pixel_space
    dicom_temp.save_as(current_dir + '/DICOM/masksAortaGroundTruth.dcm')

    dicom_temp = dicom.read_file(current_dir + '/DICOM/masksPulGroundTruth.dcm')
    dicom_temp.ImagerPixelSpacing = image_pixel_space
    dicom_temp.PixelSpacing = pixel_space
    dicom_temp.save_as(current_dir + '/DICOM/masksPulGroundTruth.dcm')

    nii_space = np.eye(4)
    nii_space[0, 0] = image_pixel_space[0]
    nii_space[1, 1] = image_pixel_space[1]

    nii_temp = nib.Nifti1Image(np.swapaxes(np.uint16(final_test_masks[:, ::-1, ::-1, 1]), 0, 2), nii_space)
    nib.save(nii_temp, current_dir + '/NIFTY/masksAortaGroundTruth.nii')

    nii_temp = nib.Nifti1Image(np.swapaxes(np.uint16(final_test_masks[:, ::-1, ::-1, 2]), 0, 2), nii_space)
    nib.save(nii_temp, current_dir + '/NIFTY/masksPulGroundTruth.nii')

    sitk.WriteImage(sitk.GetImageFromArray(np.uint16(final_test_masks[:, :, :, 1])), current_dir + '/mhd/masksAortaGroundTruth.mhd')
    sitk.WriteImage(sitk.GetImageFromArray(np.uint16(final_test_masks[:, :, :, 2])), current_dir + '/mhd/masksPulGroundTruth.mhd')

    # clear the masks for the final step:
    final_test_masks = np.where(final_test_masks == 0, 0, 0)

    num_patches = int((sum(vol_slices) - slices) / gaps)

    test_image = np.ndarray([1, slices, row, col, 1], dtype=np.int16)

    predicted_mask_volume = np.ndarray([sum(vol_slices), row, col, nb_class], dtype=np.float32)

    # model = DenseUNet_3D.get_3d_denseunet()
    # model = CRFRNN.get_3d_crfrnn_model_def()
    # model = UNet_3D.get_3d_unet_bn()
    # model = RSUNet_3D.get_3d_rsunet()
    # model = UNet_3D.get_3d_wnet1()
    model = UNet_3D.get_3d_unet()
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

      # if i == int(num_patches*0.63):
      #   vs.visualize_activation_in_layer_one_plot_add_weights(model, test_image, current_dir)
      # else:
      #   pass

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

    # ########## ROC curve aorta
    #
    # actual = imgs_true[:, :, :, 1].reshape(-1)
    # predictions = imgs_predict[:, :, :, 1].reshape(-1)
    # # predictions = np.where(predictions < (0.7), 0, 1)
    #
    # false_positive_rate_aorta, true_positive_rate_aorta, thresholds_aorta = roc_curve(actual, predictions, pos_label=1)
    # roc_auc_aorta = auc(false_positive_rate_aorta, true_positive_rate_aorta)
    # plt.figure(1, figsize=(6, 6))
    # plt.figure(1)
    # plt.title('ROC of Aorta')
    # plt.plot(false_positive_rate_aorta, true_positive_rate_aorta, 'b')
    # label = 'AUC = %0.2f' % roc_auc_aorta
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.0, 1.0])
    # plt.ylim([-0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # # plt.show()
    # saveName = '/Plots/ROC_Aorta_curve.png'
    # plt.savefig(current_dir + saveName)
    # plt.close()
    # ########## ROC curve pul
    #
    # actual = imgs_true[:, :, :, 2].reshape(-1)
    # predictions = imgs_predict[:, :, :, 2].reshape(-1)
    #
    # false_positive_rate_pul, true_positive_rate_pul, thresholds_pul = roc_curve(actual, predictions, pos_label=1)
    # roc_auc_pul = auc(false_positive_rate_pul, true_positive_rate_pul)
    # plt.figure(2, figsize=(6, 6))
    # plt.figure(2)
    # plt.title('ROC of pul')
    # plt.plot(false_positive_rate_pul, true_positive_rate_pul, 'b')
    # label = 'AUC = %0.2f' % roc_auc_pul
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.0, 1.0])
    # plt.ylim([-0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # # plt.show()
    # saveName = '/Plots/ROC_Pul_curve.png'
    # plt.savefig(current_dir + saveName)
    # plt.close()


    false_positive_rate_aorta = None
    true_positive_rate_aorta = None
    false_positive_rate_pul = None
    true_positive_rate_pul = None

    imgs_predict_threshold = np.where(imgs_predict_threshold < (0.5), 0, 1)

    if using_start_end == 1:
      aortaMean = lf.dice_coef_np(imgs_predict_threshold[start_slice:end_slice, :, :, 1],
                                  imgs_true[start_slice:end_slice, :, :, 1])
      pulMean = lf.dice_coef_np(imgs_predict_threshold[start_slice:end_slice, :, :, 2],
                                imgs_true[start_slice:end_slice, :, :, 2])
    else:
      aortaMean = lf.dice_coef_np(imgs_predict_threshold[:, :, :, 1], imgs_true[:, :, :, 1])
      pulMean = lf.dice_coef_np(imgs_predict_threshold[:, :, :, 2], imgs_true[:, :, :, 2])

    np.savetxt(current_dir + '/Plots/Aorta_Dice_mean.txt', np.array(aortaMean).reshape(1, ), fmt='%.5f')
    np.savetxt(current_dir + '/Plots/Pul_Dice_mean.txt', np.array(pulMean).reshape(1, ), fmt='%.5f')

    print('Model file:', modelname)
    print('-' * 30)
    print('Aorta Dice Coeff', aortaMean)
    print('Pul Dice Coeff', pulMean)
    print('-' * 30)

    # Draw the subplots of figures:

    color1 = 'gray'  # ***
    color2 = 'viridis'  # ******
    # color = 'plasma'  # **
    # color = 'magma'  # ***
    # color2 = 'RdPu'  # ***
    # color = 'gray'  # ***
    # color = 'gray'  # ***

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
    accuracy = float(np.loadtxt(current_dir + '/Plots/Aorta_Dice_mean.txt', float))

    # saveName = 'epoch_' + str(epoch_num) + '_dice_' +str(accuracy) + '.png'
    saveName = '/Plots/epoch_Aorta_%02d_dice_%.3f.png' % (epoch_num - 1, accuracy)

    plt.subplots_adjust(left=0.0, bottom=0.05, right=1.0, top=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(current_dir + saveName)
    plt.close()
    # plt.show()

    ################################ Pulmonary
    steps = 40
    slice = range(0, len(imgs_origin), steps)
    plt_row = 3
    plt_col = int(len(imgs_origin) / steps)

    plt.figure(4, figsize=(25, 12))
    plt.figure(4)
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
        ax1.imshow(imgs_true[i, :, :, 2], cmap=color2, alpha=transparent2)

        ax2 = plt.subplot(plt_row, plt_col, plt_num + plt_col)
        title = 'slice=' + str(i)
        plt.title(title)
        ax2.imshow(imgs_origin[i, :, :, 0], cmap=color1, alpha=transparent1)
        ax2.imshow(imgs_predict[i, :, :, 2], cmap=color2, alpha=transparent2)

        ax3 = plt.subplot(plt_row, plt_col, plt_num + 2 * plt_col)
        title = 'slice=' + str(i)
        plt.title(title)
        ax3.imshow(imgs_origin[i, :, :, 0], cmap=color1, alpha=transparent1)
        ax3.imshow(imgs_predict_threshold[i, :, :, 2], cmap=color2, alpha=transparent2)
      else:
        pass

    modelname = cm.modellist[0]

    imageName = re.findall(r'\d+\.?\d*', modelname)
    epoch_num = int(imageName[0]) + 1
    accuracy = float(np.loadtxt(current_dir + '/Plots/Pul_Dice_mean.txt', float))

    # saveName = 'epoch_' + str(epoch_num) + '_dice_' +str(accuracy) + '.png'
    saveName = '/Plots/epoch_Pul_%02d_dice_%.3f.png' % (epoch_num - 1, accuracy)

    plt.subplots_adjust(left=0.0, bottom=0.05, right=1.0, top=0.95, hspace=0.3, wspace=0.3)
    plt.savefig(current_dir + saveName)
    plt.close()
    # plt.show()

    print('Images saved')
    # Save npy as dcm files:

    final_test_aorta_predicted_threshold = final_test_masks[:, :, :, 1]
    final_test_pul_predicted_threshold = final_test_masks[:, :, :, 2]

    final_test_aorta_predicted_threshold[:, row_1:row_2, col_1:col_2] = imgs_predict_threshold[:, :, :, 1]
    final_test_pul_predicted_threshold[:, row_1:row_2, col_1:col_2] = imgs_predict_threshold[:, :, :, 2]

    new_imgs_dcm = sitk.GetImageFromArray(np.uint16(final_test_images + 4000))
    new_imgs_aorta_predict_dcm = sitk.GetImageFromArray(np.uint16(final_test_aorta_predicted_threshold))
    new_imgs_pul_predict_dcm = sitk.GetImageFromArray(np.uint16(final_test_pul_predicted_threshold))

    sitk.WriteImage(new_imgs_dcm, current_dir + '/DICOM/OriginalImages.dcm')
    sitk.WriteImage(new_imgs_aorta_predict_dcm, current_dir + '/DICOM/masksAortaPredicted.dcm')
    sitk.WriteImage(new_imgs_pul_predict_dcm, current_dir + '/DICOM/masksPulPredicted.dcm')

    dicom_temp = dicom.read_file(current_dir + '/DICOM/OriginalImages.dcm')
    dicom_temp.ImagerPixelSpacing = image_pixel_space
    dicom_temp.PixelSpacing = pixel_space
    dicom_temp.save_as(current_dir + '/DICOM/OriginalImages.dcm')

    nii_temp = nib.Nifti1Image(np.swapaxes(np.uint16(final_test_images[:, ::-1, ::-1]), 0, 2), nii_space)
    nib.save(nii_temp, current_dir + '/NIFTY/OriginalImages.nii')

    dicom_temp = dicom.read_file(current_dir + '/DICOM/masksAortaPredicted.dcm')
    dicom_temp.ImagerPixelSpacing = image_pixel_space
    dicom_temp.PixelSpacing = pixel_space
    dicom_temp.save_as(current_dir + '/DICOM/masksAortaPredicted.dcm')

    dicom_temp = dicom.read_file(current_dir + '/DICOM/masksPulPredicted.dcm')
    dicom_temp.ImagerPixelSpacing = image_pixel_space
    dicom_temp.PixelSpacing = pixel_space
    dicom_temp.save_as(current_dir + '/DICOM/masksPulPredicted.dcm')

    dicom_temp = None

    final_test_aorta_predicted = final_test_masks[:, :, :, 1]
    final_test_pul_predicted = final_test_masks[:, :, :, 2]

    final_test_aorta_predicted[:, row_1:row_2, col_1:col_2] = imgs_predict[:, :, :, 1]
    final_test_pul_predicted[:, row_1:row_2, col_1:col_2] = imgs_predict[:, :, :, 2]

    nii_temp = nib.Nifti1Image(np.swapaxes(np.uint16(final_test_aorta_predicted[:, ::-1, ::-1]), 0, 2), nii_space)
    nib.save(nii_temp, current_dir + '/NIFTY/masksAortaPredicted.nii')

    nii_temp = nib.Nifti1Image(np.swapaxes(np.uint16(final_test_pul_predicted[:, ::-1, ::-1]), 0, 2), nii_space)
    nib.save(nii_temp, current_dir + '/NIFTY/masksPulPredicted.nii')

    nii_temp = None

    sitk.WriteImage(new_imgs_dcm, current_dir + '/mhd/imagesPredicted.mhd')
    sitk.WriteImage(new_imgs_aorta_predict_dcm, current_dir + '/mhd/masksAortaPredicted.mhd')
    sitk.WriteImage(new_imgs_pul_predict_dcm, current_dir + '/mhd/masksPulPredicted.mhd')

    # mt.SegmentDist(current_dir + '/mhd/masksAortaPredicted.mhd',current_dir + '/mhd/masksAortaGroundTruth.mhd', current_dir + '/Surface_Distance/Aorta', 'Aorta')
    # mt.SegmentDist(current_dir + '/mhd/masksPulPredicted.mhd',current_dir + '/mhd/masksPulGroundTruth.mhd', current_dir + '/Surface_Distance/Pul', 'Pul')

    print('DICOM saved')

    # Clear memory for the next testing sample:

    final_test_aorta_predicted_threshold = None
    final_test_pul_predicted_threshold = None
    final_test_aorta_predicted = None
    final_test_pul_predicted = None
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

  results_path = '/hdd2/PythonCodes/Aorta_Segmentation_2D_3D_Shuai/testingSet/Results/'

  folder_name = 'vol*'

  vol_list = sorted(glob(results_path + folder_name))

  nifty_evaluation(vol_list)

  sys.exit(0)

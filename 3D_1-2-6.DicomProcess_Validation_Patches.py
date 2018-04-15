#
# 3D Aorta Segmentation Project
#
# 1. Dicom Process with large data set from hard disk
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

from glob import glob
import Modules.Common_modules as cm
import Modules.DataProcess as dp
from keras.utils import to_categorical
import numpy as np

try:
  from tqdm import tqdm  # long waits are not fun
except:
  print('TQDM does make much nicer wait bars...')
  tqdm = lambda i: i

# Import dcm file and turn it into an image array:

# Get the file list:
# originFile_list = sorted(glob(cm.workingPath.originTrainingSet_path + 'vol*.npy'))
# maskFile_list = sorted(glob(cm.workingPath.maskTrainingSet_path + 'vol*.npy'))
originFile_list = sorted(glob(cm.workingPath.originValidationSet_path + 'vol*.dcm'))
maskAortaFile_list = sorted(glob(cm.workingPath.aortaValidationSet_path + 'vol*.dcm'))
maskPulFile_list = sorted(glob(cm.workingPath.pulValidationSet_path + 'vol*.dcm'))

filename = str(originFile_list[0])[24:-4]

filelist = open(cm.workingPath.validationSet_path + "file.txt", 'w')
filelist.write(str(len(originFile_list)))
filelist.write(" datasets involved")
filelist.write("\n")
for file in originFile_list:
  filelist.write(file)
  filelist.write('\n')
filelist.close()

# Load file:

axis_process = "Axial"  ## axis you want to process
# axis_process = "Sagittal"
# axis_process = "Coronal"

# Training set:
print('-' * 30)
print('Loading files...')
print('-' * 30)

vol_slices = []
cm.mkdir(cm.workingPath.validationPatchesSet_path)

for nb_file in range(len(originFile_list)):

  out_images = []
  out_masks = []

  # Read information from dcm file:
  originVol, originVol_num, originVolwidth, originVolheight = dp.loadFile(originFile_list[nb_file])
  maskAortaVol, maskAortaVol_num, maskAortaVolwidth, maskAortaVolheight = dp.loadFile(maskAortaFile_list[nb_file])
  maskPulVol, maskPulVol_num, maskPulVolwidth, maskPulVolheight = dp.loadFile(maskPulFile_list[nb_file])
  maskVol = maskAortaVol

  # Turn the mask images to binary images:
  for j in range(len(maskAortaVol)):
    maskAortaVol[j] = np.where(maskAortaVol[j] != 0, 1, 0)
  for j in range(len(maskPulVol)):
    maskPulVol[j] = np.where(maskPulVol[j] != 0, 2, 0)

  maskVol = maskVol + maskPulVol

  for j in range(len(maskVol)):
    maskVol[j] = np.where(maskVol[j] > 2, 0, maskVol[j])

  # Make the Vessel class
  for j in range(len(maskVol)):
    maskVol[j] = np.where(maskVol[j] != 0, 1, 0)


  for j in range(originVol.shape[0]):
    img = originVol[j, :, :]
    out_images.append(img)
  for j in range(maskVol.shape[0]):
    img = maskVol[j, :, :]
    out_masks.append(img)

  vol_cur_slices = originVol.shape[0]
  nb_class = 2
  outmasks_onehot = to_categorical(out_masks, num_classes=nb_class)
  final_images = np.ndarray([vol_cur_slices, 512, 512, 1], dtype=np.int16)
  final_masks = np.ndarray([vol_cur_slices, 512, 512, nb_class], dtype=np.int8)

  for j in range(len(out_images)):
    final_images[j, :, :, 0] = out_images[j]
    final_masks[j] = outmasks_onehot[j]

  row = cm.img_rows_3d
  col = cm.img_cols_3d
  num_rowes = 1
  num_coles = 1
  row_1 = int((512 - row) / 2)
  row_2 = int(512 - (512 - row) / 2)
  col_1 = int((512 - col) / 2)
  col_2 = int(512 - (512 - col) / 2)
  slices = cm.slices_3d
  gaps = cm.gaps_3d
  thi_1 = int((512 - slices) / 2)
  thi_2 = int(512 - (512 - slices) / 2)
  final_images_crop = final_images[:, row_1:row_2, col_1:col_2, :]
  final_masks_crop = final_masks[:, row_1:row_2, col_1:col_2, :]

  num1 = 0

  final_images_crop_save = np.ndarray([num1, row, col, 1], dtype=np.int16)
  final_masks_crop_save = np.ndarray([num1, row, col, nb_class], dtype=np.int8)

  print('Images to Patches')
  print('-' * 30)

  count = 0
  for num_vol in range(0, 1):
    num_patches = int(((vol_cur_slices - slices) / gaps) + 1)
    if num_vol>0:
      count = count
    else:
      pass
    for num_patch in range(0, num_patches):
      for num_row in range(0, num_rowes):
        for num_col in range(0, num_coles):
          count1 = int(count + num_patch*gaps)
          count2 = int(count + (num_patch+1)*gaps + slices-gaps)
          final_images_crop_save = np.concatenate(
            (final_images_crop_save, final_images_crop[count1:count2,
                                         num_row * row:(num_row * row + row), num_col * col:(num_col * col + col), :]), axis=0)

          final_masks_crop_save = np.concatenate(
            (final_masks_crop_save, final_masks_crop[count1:count2,
                                        num_row * row:(num_row * row + row), num_col * col:(num_col * col + col), :]), axis=0)

  tubes = int(final_images_crop_save.shape[0]/slices)
  final_images_crop_save_last = np.ndarray([tubes, slices, row, col, 1], dtype=np.int16)
  final_masks_crop_save_last = np.ndarray([tubes, slices, row, col, nb_class], dtype=np.int8)

  for i in range(tubes):
    final_images_crop_save_last[i,:,:,:,:] = final_images_crop_save[i*slices:(i+1)*slices, :,:,:]
    final_masks_crop_save_last[i,:,:,:,:] = final_masks_crop_save[i*slices:(i+1)*slices, :,:,:]

  print('Saving Images...%04d' % nb_file)
  print('-' * 30)

  num_out = int(len(final_images_crop_save_last)/cm.nb_batch_size)
  vol_slices.append(num_out)
  pre_patch = sum(vol_slices[:-1])

  for i in range(num_out):
    out1 = np.ndarray([cm.nb_batch_size, slices, row, col, 1], dtype=np.int16)
    out2 = np.ndarray([cm.nb_batch_size, slices, row, col, nb_class], dtype=np.int8)
    for j in range(int(cm.nb_batch_size)):
      out1[j] = final_images_crop_save_last[i+j]
      out2[j] = final_masks_crop_save_last[i+j]

    np.save(cm.workingPath.validationPatchesSet_path + 'img_%04d.npy' %(i + pre_patch), out1)
    np.save(cm.workingPath.validationPatchesSet_path + 'mask_%04d.npy' %(i + pre_patch), out2)

print('Training Images Saved')
print('-' * 30)

print("Finished")

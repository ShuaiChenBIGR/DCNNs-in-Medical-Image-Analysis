#
# Aorta Segmentation Project
#
# 1. Dicom Process Concerate npy
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
# created on 12/01/2018
# Last update: 12/01/2018
########################################################################################
from glob import glob
import Modules.Common_modules as cm
import Modules.DataProcess as dp
import SimpleITK as sitk
from keras.utils import plot_model, to_categorical
import numpy as np
from PIL import Image
import dicom
import cv2

try:
  from tqdm import tqdm  # long waits are not fun
except:
  print('TQDM does make much nicer wait bars...')
  tqdm = lambda i: i


###########################################
########### Main Program Begin ############
###########################################

# Import dcm file and turn it into an image array:
# Get the file list:
originFile_list = sorted(glob(cm.workingPath.training3DSet_path + 'trainImages_*.npy'))
maskAortaFile_list = sorted(glob(cm.workingPath.training3DSet_path + 'trainAortaMasks_*.npy'))
maskPulFile_list = sorted(glob(cm.workingPath.training3DSet_path + 'trainPulMasks_*.npy'))

# Load file:

# out_test_images = []
# out_test_masks = []
axis_process = "Axial"  ## axis you want to process
# axis_process = "Sagittal"
# axis_process = "Coronal"

# Training set:
print('-' * 30)
print('Loading files...')
print('-' * 30)

vol_slices = []

if axis_process == "Axial":
  for nb_file in range(len(originFile_list)):
    out_images = []
    out_masks = []
    # Read information from dcm file:
    # originVolInfo = loadFileInformation(originFile_list[i])
    # maskVolInfo = loadFileInformation(maskFile_list[i])
    originVol = np.load(originFile_list[nb_file])
    maskAortaVol = np.load(maskAortaFile_list[nb_file])
    maskPulVol = np.load(maskPulFile_list[nb_file])
    maskVol = maskAortaVol
    # originVol = np.squeeze(originVol, axis=-1)
    # maskVol = np.squeeze(maskVol, axis=-1)

    # Turn the mask images to binary images:

    # Make the Aorta class
    for j in range(len(maskAortaVol)):
      maskAortaVol[j] = np.where(maskAortaVol[j] != 0, 1, 0)

    # Make the Pulmonary class
    for j in range(len(maskPulVol)):
      maskPulVol[j] = np.where(maskPulVol[j] != 0, 2, 0)

    maskVol = maskVol + maskPulVol

    for j in range(len(maskVol)):
      maskVol[j] = np.where(maskVol[j] > 2, 0, maskVol[j])

    # Make the Vessel class
    for j in range(len(maskVol)):
      maskVol[j] = np.where(maskVol[j] != 0, 1, 0)

    for i in range(originVol.shape[0]):
      img = originVol[i, :, :]
      # new_img = resize(img, [512, 512])
      out_images.append(img)
    for i in range(maskVol.shape[0]):
      # new_mask = resize(img, [512, 512])
      img = maskVol[i, :, :]
      # img = to_categorical(img, num_classes=3)
      out_masks.append(img)

    vol_slices = originVol.shape[0]

    nb_class = 2
    outmasks_onehot = to_categorical(out_masks, num_classes=nb_class)
    final_images = np.ndarray([vol_slices, 512, 512, 1], dtype=np.int16)
    final_masks = np.ndarray([vol_slices, 512, 512, nb_class], dtype=np.int8)

    for i in range(len(out_images)):
      final_images[i, :, :, 0] = out_images[i]
      final_masks[i] = outmasks_onehot[i]

    print('Saving Images...')
    print('-' * 30)

    np.save(cm.workingPath.training3DSet_path + '/vesselclass/Images_%04d.npy' %nb_file, final_images)
    np.save(cm.workingPath.training3DSet_path + '/vesselclass/Masks_%04d.npy' %nb_file, final_masks)

else:
  pass

print('Training Images Saved')
print('-' * 30)
print('Finished')

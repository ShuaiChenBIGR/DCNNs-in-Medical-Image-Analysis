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

# originFile_list = sorted(glob(cm.workingPath.home_path + 'trainImages3D16.npy'))
# maskFile_list = sorted(glob(cm.workingPath.home_path + 'trainMasks3D16.npy'))

originFile_list = sorted(glob(cm.workingPath.originTrainingSet_path + 'vol*.dcm'))
maskAortaFile_list = sorted(glob(cm.workingPath.aortaTrainingSet_path + 'vol*.dcm'))
maskPulFile_list = sorted(glob(cm.workingPath.pulTrainingSet_path + 'vol*.dcm'))

axis_process = "Axial"  ## axis you want to process
# axis_process = "Sagittal"
# axis_process = "Coronal"
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
# Training set:
print('-' * 30)
print('Loading files...')
print('-' * 30)

if axis_process == "Axial":
  for i in range(len(originFile_list)):

    originVol, originVol_num, originVolwidth, originVolheight = dp.loadFile(originFile_list[i])
    maskAortaVol, maskAortaVol_num, maskAortaVolwidth, maskAortaVolheight = dp.loadFile(maskAortaFile_list[i])
    maskPulVol, maskPulVol_num, maskPulVolwidth, maskPulVolheight = dp.loadFile(maskPulFile_list[i])

    np.save(cm.workingPath.training3DSet_path + 'trainImages_%04d.npy' %i, originVol[:, row_1:row_2, col_1:col_2])
    np.save(cm.workingPath.training3DSet_path + 'trainAortaMasks_%04d.npy' %i, maskAortaVol[:, row_1:row_2, col_1:col_2])
    np.save(cm.workingPath.training3DSet_path + 'trainPulMasks_%04d.npy' %i, maskPulVol[:, row_1:row_2, col_1:col_2])

else:
  pass

print('Training Images Saved')
print('-' * 30)
print('Finished')

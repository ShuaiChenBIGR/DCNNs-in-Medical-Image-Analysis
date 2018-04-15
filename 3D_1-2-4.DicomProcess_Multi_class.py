########################################################################################
# 3D Aorta Segmentation Project                                                        #
#                                                                                      #
# 1. Dicom Process with 3D Overlap Patch                                                #
#                                                                                      #
# created by                                                                           #
# Shuai Chen                                                                           #
# PhD student                                                                          #
# Medical Informatics                                                                  #
#                                                                                      #
# P.O. Box 2040, 3000 CA Rotterdam, The Netherlands, internal postal address Na-2603   #
# Visiting address: office 2616, Wytemaweg 80, 3015 CN Rotterdam, The Netherlands      #
# Email s.chen.2@erasmusmc.nl | Telephone +31 6 334 516 99                             #
# www.erasmusmc.nl                                                                     #
#                                                                                      #
# created on 24/01/2018                                                                #
# Last update: 24/01/2018                                                              #
########################################################################################


from glob import glob
import Modules.Common_modules as cm
from keras.utils import plot_model, to_categorical
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


# load dcm file:
def loadFile(filename):
  ds = sitk.ReadImage(filename)
  img_array = sitk.GetArrayFromImage(ds)
  frame_num, width, height = img_array.shape
  return img_array, frame_num, width, height


# load dcm file imformation:
def loadFileInformation(filename):
  information = {}
  ds = dicom.read_file(filename)
  information['PatientID'] = ds.PatientID
  information['PatientName'] = ds.PatientName
  information['PatientBirthDate'] = ds.PatientBirthDate
  information['PatientSex'] = ds.PatientSex
  information['StudyID'] = ds.StudyID
  # information['StudyTime'] = ds.Studytime
  information['InstitutionName'] = ds.InstitutionName
  information['Manufacturer'] = ds.Manufacturer
  information['NumberOfFrames'] = ds.NumberOfFrames
  return information


# extract specific image array from dcm file:
def showImage(img_array, frame_num=0):
  img_bitmap = Image.fromarray(img_array[frame_num])
  return img_bitmap


# optimize image using Constrast Limit Adaptive Histogram Equalization (CLAHE):
def limitedEqualize(img_array, limit=2.0):
  img_array_list = []
  for img in img_array:
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
    img_array_list.append(clahe.apply(img))
  img_array_limited_equalized = np.array(img_array_list)
  return img_array_limited_equalized


# def writeVideo(img_array):
# 	frame_num, width, height = img_array.shape
# 	filename_output = filename.split('.')[0]+'.avi'
# 	video = cv2.VideoWriter(filename_output, -1, 16, (width, height))
# 	for img in img_array:
# 		video.write(img)
# 	video.release()

###########################################
########### Main Program Begin ############
###########################################


# Import dcm file and turn it into an image array:

# Get the file list:
# originFile_list = sorted(glob(cm.workingPath.originTrainingSet_path + 'vol*.npy'))
# maskFile_list = sorted(glob(cm.workingPath.maskTrainingSet_path + 'vol*.npy'))
originFile_list = sorted(glob(cm.workingPath.originTrainingSet_path + 'vol*.dcm'))[:]
maskAortaFile_list = sorted(glob(cm.workingPath.aortaTrainingSet_path + 'vol*.dcm'))[:]
maskPulFile_list = sorted(glob(cm.workingPath.pulTrainingSet_path + 'vol*.dcm'))[:]

filename = str(originFile_list[0])[24:-4]

filelist = open(cm.workingPath.trainingSet_path + "file.txt", 'w')
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
out_images = []
out_masks = []

for i in range(len(originFile_list)):

  # Read information from dcm file:

  # originVol = np.load(originFile_list[i])
  # maskVol = np.load(maskFile_list[i])
  originVol, originVol_num, originVolwidth, originVolheight = loadFile(originFile_list[i])
  maskAortaVol, maskAortaVol_num, maskAortaVolwidth, maskAortaVolheight = loadFile(maskAortaFile_list[i])
  maskPulVol, maskPulVol_num, maskPulVolwidth, maskPulVolheight = loadFile(maskPulFile_list[i])

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

  # # Make the Vessel class
  # for j in range(len(maskVol)):
  #   maskVol[j] = np.where(maskVol[j] != 0, 1, 0)

  for i in range(originVol.shape[0]):
    img = originVol[i, :, :]
    # new_img = resize(img, [512, 512])
    out_images.append(img)
  for i in range(maskVol.shape[0]):
    # new_mask = resize(img, [512, 512])
    img = maskVol[i, :, :]
    # img = to_categorical(img, num_classes=3)
    out_masks.append(img)

  vol_slices.append(originVol.shape[0])

nb_class = 3
# outmasks_onehot = to_categorical(out_masks, num_classes=nb_class)
final_images = np.ndarray([sum(vol_slices), 512, 512], dtype=np.int16)
final_masks = np.ndarray([sum(vol_slices), 512, 512, nb_class], dtype=np.int8)

for i in range(len(out_images)):
  final_images[i] = out_images[i]
  final_masks[i] = out_masks[i]


final_images = np.expand_dims(final_images, axis=-1)
# final_masks = np.expand_dims(final_masks, axis=-1)

num_file = range(0, len(vol_slices))

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
for num_vol in range(0, len(vol_slices)):
  num_patches = int(((vol_slices[num_vol] - slices) / gaps) + 1)
  if num_vol>0:
    count = count + vol_slices[num_vol-1]
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

random_scale = int(len(final_images_crop_save_last))

# rand_i = np.random.choice(range(random_scale), size=random_scale, replace=False)

print('Saving Images...')
print('-' * 30)

np.save(cm.workingPath.training3DSet_path + 'trainImages3D.npy', final_images_crop_save_last)
np.save(cm.workingPath.training3DSet_path + 'trainMasks3D.npy', final_masks_crop_save_last)

print('Training Images Saved')
print('-' * 30)

print("Finished")

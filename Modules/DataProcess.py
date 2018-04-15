import Modules.Common_modules as cm
import numpy as np
import SimpleITK as sitk
import dicom

#########################################################
# Data process

def train_split(nb_file, rand_i):
  """
  Set the validation split value and make list (By Shuai)
  """
  train_num = int(nb_file * 0.9)
  val_num = nb_file - train_num
  train_list = rand_i[0:train_num]
  val_list = rand_i[train_num:]

  return train_num, val_num, train_list, val_list


def train_val_split(nb_file, rand_i):
  """
  Set the validation split value and make list (By Shuai)
  """
  train_num = int(nb_file)
  train_list = rand_i[0:train_num]

  return train_num, train_list

def BatchGenerator(i, x_list, y_list, train_list):
  """
  Define how to load training data one by one from hard disk (By Shuai)
  """
  x = np.load(x_list[train_list[i]])
  y = np.load(y_list[train_list[i]])

  return x, y


def loadFile(filename):
  """
  Define how to load dicom file(By Shuai)
  """
  ds = sitk.ReadImage(filename)
  img_array = sitk.GetArrayFromImage(ds)
  frame_num, width, height = img_array.shape
  return img_array, frame_num, width, height


def loadFileInformation(filename):
  """
  Define how to load dicom information(By Shuai)
  """
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
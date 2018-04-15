import os
from glob import glob

img_rows_2d = 512
img_cols_2d = 512
smooth = 1

img_rows_3d = 256
img_cols_3d = 256
slices_3d = 16
gaps_3d = 2

val_split = 0.9
nb_batch_size = 1

class workingPath(list):
  # setup working paths:  (for Windows: \\)
  working_path = '/hdd2/PythonCodes/Aorta_Segmentation_2D_3D_Shuai/'
  home_path = "/home/schen/Desktop/"

  # setup model paths:
  model_path = os.path.join(working_path, 'Models/')
  best_model_path = os.path.join(model_path, 'Best_Models/')
  visual_path = os.path.join(model_path, 'Visual_files/')
  gradient_path = os.path.join(model_path, 'Gradient_files/')
  tensorboard_path = os.path.join(model_path, 'Tensorboard/')

  # setup training paths:
  trainingSet_path = os.path.join(working_path, 'trainingSet/')
  originTrainingSet_path = os.path.join(trainingSet_path, 'originSet/')
  maskTrainingSet_path = os.path.join(trainingSet_path, 'maskSet/')
  aortaTrainingSet_path = os.path.join(maskTrainingSet_path, 'Aorta/')
  pulTrainingSet_path = os.path.join(maskTrainingSet_path, 'Pul/')
  training3DSet_path = os.path.join(trainingSet_path, '3D/')
  trainingPatchesSet_path = os.path.join(training3DSet_path, 'patches/')
  trainingAugSet_path = os.path.join(trainingSet_path, 'originAugSet/')

  # setup validation paths:
  validationSet_path = os.path.join(working_path, 'validationSet/')
  originValidationSet_path = os.path.join(validationSet_path, 'originSet/')
  maskValidationSet_path = os.path.join(validationSet_path, 'maskSet/')
  aortaValidationSet_path = os.path.join(maskValidationSet_path, 'Aorta/')
  pulValidationSet_path = os.path.join(maskValidationSet_path, 'Pul/')
  validationPatchesSet_path = os.path.join(validationSet_path, 'patches/')

  # setup testing paths:
  testingSet_path = os.path.join(working_path, 'testingSet/')
  testingNPY_path = os.path.join(testingSet_path, 'npy/')
  testingResults_path = os.path.join(testingSet_path, 'Results/')
  originTestingSet_path = os.path.join(testingSet_path, 'originSet/')
  originAbnormalTestingSet_path = os.path.join(originTestingSet_path, 'abnormal/')
  originLidiaTestingSet_path = os.path.join(originTestingSet_path, 'Lidia_Data/')
  maskTestingSet_path = os.path.join(testingSet_path, 'maskSet/')
  maskLidiaTestingSet_path = os.path.join(maskTestingSet_path, 'Lidia_Lung_Segmentation/')
  aortaTestingSet_path = os.path.join(maskTestingSet_path, 'Aorta/')
  pulTestingSet_path = os.path.join(maskTestingSet_path, 'Pul/')

filename = 'vol3885_*.dcm'
start_slice = 110
# filename = 'Predicted_vol3885_*.dcm'

# modellist = glob(workingPath.model_path + 'weights.epoch_174*.hdf5')
#
modellist = glob(workingPath.model_path + 'Best_weights.187*.hdf5')

# modellist = glob(workingPath.model_path + 'Val.0*.hdf5')

# modellist = glob(workingPath.model_path + 'Val.09_Best_weights.48-0.00000.hdf5')

# modellist = glob(workingPath.model_path + 'unet.hdf5')

if len(modellist) == 0:
  pass
else:
  model_num = 77

# modelname = workingPath.model_path + 'Best_weights.03-0.00082.hdf5'


def mkdir(path):
  #import os

  path = path.strip()
  path = path.rstrip("\\")

  isExists = os.path.exists(path)

  if not isExists:
    os.makedirs(path)
    return True
  else:
    return False


def set_limit_gpu_memory_usage():

    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    import os
    DEFAULT_FRAC_GPU_USAGE = 0.5

    def get_session(gpu_fraction=DEFAULT_FRAC_GPU_USAGE):
        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        if num_threads:
            return tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    KTF.set_session(get_session())



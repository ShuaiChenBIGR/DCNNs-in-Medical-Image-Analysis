
import numpy as np
from keras import backend as K
import Modules.Common_modules as cm



######################################################
# Loss function:
def customloss(y_true, y_pred):
  pos = (y_true > 0) * 1  # get binary pos and ignore
  ignore = (y_true == 0) * 1

  return K.sum(-pos * K.log(y_pred)) / (cm.img_rows * cm.img_cols - K.sum(ignore) / 5 + 1e-10)


def customlossw(y_true, y_pred):
  pos = (y_true > 0) * 1  # get binary pos and ignore
  weightM = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])  # weights per class (for 4 classes)
  weightVTM = K.variable(value=weightM, dtype='float64', name='weightmatrix')

  return K.sum(K.dot((-pos * K.log(y_pred)), weightVTM)) / (K.sum(K.dot(pos, weightVTM)))


def EvaluatedVoxelAccuracy(y_true, y_pred):
  Ytrue_int = K.argmax(y_true, axis=2)
  Ypred_int = K.argmax(y_pred, axis=2)
  NotIgnore = K.abs(K.max(y_true, axis=2))
  nom = K.sum((K.equal(Ytrue_int, Ypred_int)) * 1 * NotIgnore)
  den = K.sum(NotIgnore)
  return nom / den


# Dice coefficients:
def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + cm.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + cm.smooth)


def dice_coef_np(y_true, y_pred):
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  intersection = np.sum(y_true_f * y_pred_f)
  return (2. * intersection + cm.smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + cm.smooth)


def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)


# Cross entropy:
def binary_crossentropy_loss(y_true, y_pred):
  return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
def binary_crossentropy(y_true, y_pred):
  return -binary_crossentropy_loss(y_true, y_pred)

# Multi Class Cross-entropy:
def categorical_crossentropy_loss(y_true, y_pred):
  return K.mean(K.categorical_crossentropy(y_true, y_pred), axis=-1)
def categorical_crossentropy(y_true, y_pred):
  return -categorical_crossentropy_loss(y_true, y_pred)*1.0e9



# Weighted Multi Class Cross-entropy:
def weighted_categorical_crossentropy_loss(weights):
  """
  A weighted version of keras.objectives.categorical_crossentropy

  Variables:
      weights: numpy array of shape (C,) where C is the number of classes

  Usage:
      weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
      loss = weighted_categorical_crossentropy(weights)
      model.compile(loss=loss,optimizer='adam')
  """

  weights = K.variable(weights)

  def loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

  return loss





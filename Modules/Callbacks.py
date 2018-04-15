import Modules.Common_modules as cm
import numpy as np
import keras.losses
import csv, codecs
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras import callbacks
from keras import backend as K


#########################################################
# Callback function:


class RecordLossHistory(callbacks.Callback):
  """
  Loss history in a same file by line(By Antonio)
  """

  def __init__(self):
    self.filename = cm.workingPath.model_path + 'lossHistory.txt'

  def on_train_begin(self, logs=None):
    self.fout = open(self.filename, 'w')
    self.fout.write('/epoch/ /loss/ /val_loss/\n')
    self.fout.close()

  def on_epoch_end(self, epoch, logs=None):
    self.fout = open(self.filename, 'a')
    newdataline = '%s %s %s\n' % (epoch, logs.get('loss'), logs.get('val_loss'))
    self.fout.write(newdataline)
    self.fout.close()


# Loss history with decreasing learning rate(By Shuai)
class LossHistory_lr(callbacks.Callback):
  """
  Loss history with decreasing learning rate(By Shuai)
  """

  def on_train_begin(self, logs={}):
    self.losses = []
    self.val_losses = []
    self.sd = []

  def on_epoch_end(self, epoch, logs={}):
    self.losses.append(logs.get('loss'))
    self.val_losses.append(logs.get('val_loss'))

    self.sd.append(step_decay(len(self.losses)))
    print('\nlr:', step_decay(len(self.losses)))
    lrate_file = list(self.sd)
    np.savetxt(cm.workingPath.model_path + 'lrate.txt', lrate_file, newline='\r\n')


def step_decay(losses):
  """
  Decreasing learning rate by step(By Shuai)
  """
  if len(LossHistory_lr.losses) == 0:
    lrate = 0.00001
    return lrate
  elif float(2 * np.sqrt(np.array(LossHistory_lr.losses[-1]))) < 1.0:
    lrate = 0.00001 * 1.0 / (1.0 + 0.1 * len(LossHistory_lr.losses))
    return lrate
  else:
    lrate = 0.00001
    return lrate


# Loss history in seperate files(By Shuai)
class LossHistory(callbacks.Callback):
  """
  Loss history in seperate files(By Shuai)
  """

  def on_train_begin(self, logs={}):
    self.losses = []
    self.val_losses = []

  def on_epoch_end(self, epoch, logs={}):
    self.losses.append(logs.get('loss'))
    loss_file = list(self.losses)
    np.savetxt(cm.workingPath.model_path + 'loss.txt', loss_file, newline='\r\n')

    self.val_losses.append(logs.get('val_loss'))
    val_loss_file = list(self.val_losses)
    np.savetxt(cm.workingPath.model_path + 'val_loss.txt', val_loss_file, newline='\r\n')


class LossHistory_Arno(callbacks.Callback):
  """
  Loss history (By Arno)
  """

  def __init__(self, ne):
    self.ne = ne

  def on_train_begin(self, logs={}):
    print('Start training ...')
    self.stats = ['loss']  # TODO: check
    self.logs = [{} for _ in range(self.ne)]

    with open(os.path.join(cm.workingPath.model_path, 'evolution.csv'), "w") as myfile:
      myfile.write(';'.join(self.stats + ['val_' + s for s in self.stats]) + "\n")

  def on_epoch_end(self, epoch, logs={}):
    print(logs)
    self.logs[epoch] = logs
    with open(os.path.join(cm.workingPath.model_path, 'evolution.csv'), "a") as myfile:

      plt.figure(figsize=(20, 60))
      plt.suptitle(cm.workingPath.model_path, fontsize=34, fontweight='bold')

      gs = gridspec.GridSpec(len(self.stats), 1)

      for idx, stat in enumerate(self.stats):
        plt.subplot(gs[idx])
        plt.ylabel(stat, fontsize=34)
        losses = [self.logs[e][stat] for e in range(epoch)]
        val_losses = [self.logs[e]['val_' + stat] for e in range(epoch)]
        plt.plot(range(0, epoch), losses, '-', color='b')
        plt.plot(range(0, epoch), val_losses, '-', color='r')
        plt.tick_params(axis='x', labelsize=30)
        plt.tick_params(axis='y', labelsize=30)
        plt.grid(True)

      try:
        plt.savefig(os.path.join(cm.workingPath.model_path, 'loss.png'))
      except Exception as inst:
        print(type(inst))
        print(inst)
      plt.close()


class LossHistory_Gerda(callbacks.Callback):
  """
  Loss history(By Gerda)
  """

  def __init__(self, savePaths):

    # parameter of the csv and plot
    self.reports = [{'type': 'text',
                     'file': 'evolution.csv',
                     'outputs': [0],
                     'order': ['metric', 'set', 'output']},
                    {'type': 'plot',
                     'file': 'plot.png',
                     'outputs': [0],
                     'order': ['metric', 'set', 'output']}]

    self.savePaths = savePaths

    metrics = []

    self.dimSpecs = {'set': ['', 'val_'], 'output': [0], 'metric': ['loss'] + metrics}

    outputs = ['']

    #        addVars = [{'name': 'loss', 'kerasName': 'loss', 'val': []},
    #                    {'name': 'val_loss', 'kerasName': 'val_loss', 'val': []}]

    addVars = []

    #        addNesting = [('overall loss', [('overall loss', [0, 1])])]

    addNesting = []

    def getKerasName(v):

      return '%s%s%s' % (v['set'], outputs[v['output']], v['metric'])

    for r in self.reports:

      # list of variables to monitor
      rvars = []

      # nesting: primarily for plots (to e.g. plot several variables on the same plot)
      nesting = []

      # only consider specified outputs
      dims = self.dimSpecs.copy()
      dims['output'] = r['outputs']

      dimSizes = [len(dims[d]) for d in r['order']]

      for d1 in range(dimSizes[0]):

        for d2 in range(dimSizes[1]):

          for d3 in range(dimSizes[2]):

            idx = [d1, d2, d3]
            v = [(r['order'][i], dims[r['order'][i]][idx[i]]) \
                 for i in range(3)]

            if d2 == 0 and d3 == 0:
              nesting.append((v[0][1], []))

            if d3 == 0:
              nesting[d1][1].append((v[1][1], []))

            nesting[d1][1][d2][1].append(len(rvars) + len(addVars))

            v = dict(v)

            v.update({'kerasName': getKerasName(v),
                      'name': '%s%s_%d' % (v['set'], v['metric'], v['output']), 'val': []})

            rvars.append(v)

      r['vars'] = addVars + rvars
      r['nesting'] = addNesting + nesting

  def on_epoch_end(self, epoch, logs={}):

    for r in self.reports:

      for idx, v in reversed(list(enumerate(r['vars']))):

        # save variables into the report objects

        if v['kerasName'] in logs.keys():
          v['val'].append(logs[v['kerasName']])
        elif epoch == 0:
          print
          idx
          del r['vars'][idx]

      # report

      if r['type'] == 'text':
        self.writeCSV(r, logs, True if epoch == 0 else False)
      elif r['type'] == 'plot':
        self.plot(r, logs)

  def writeCSV(self, report, logs, rewrite=False):

    for path in self.savePaths:

      with open(os.path.join(path, 'evolution.csv'), "w" if rewrite else "a") as myfile:

        writer = csv.writer(myfile, delimiter=';')

        rvars = report['vars']

        if rewrite:
          writer.writerow([rvar['name'] for rvar in rvars])

        writer.writerow([rvar['val'][-1] for rvar in rvars])

  def plot(self, report, logs):

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    lineStyle = ['dashed', 'solid', 'dotted']

    rvars = report['vars']
    nesting = report['nesting']

    plt.figure(figsize=(10, 10 * len(nesting)))
    #        plt.figure(figsize=(20, 60))
    gs = gridspec.GridSpec(len(nesting), 1, height_ratios=[1] * (len(nesting)))

    # plot
    for i1, l1 in enumerate(nesting):

      ax = plt.subplot(gs[i1])
      plt.title(l1[0])
      plt.ylabel(l1[0])

      for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(30)

      for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

      # color
      for i2, l2 in enumerate(l1[1]):

        # line fill
        for i3, l3 in enumerate(l2[1]):
          rvars[l3]['val']

          line, = plt.plot(range(0, len(rvars[l3]['val'])), rvars[l3]['val'], \
                           ls=lineStyle[i3], color=colors[i2], label=rvars[l3]['name'])

      # Shrink current axis by 20%
      box = ax.get_position()
      ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

      # Put a legend to the right of the current axis
      ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

    # ax.legend(bbox_to_anchor=(1.2, 0.5), fontsize = 20)

    try:

      plt.savefig(os.path.join(self.savePaths[0], report['file']))

    except Exception as inst:

      print(type(inst))
      print(inst)

    plt.close()


class LossHistory_Florian(callbacks.Callback):
  """
  LossHistory (By Florian)
  """
  def __init__(self, X_train, y_train, layer_index):
    super(callbacks.Callback, self).__init__()
    self.layer_index = layer_index
    if X_train.shape[0] >= 1000:
      mask = np.random.choice(X_train.shape[0], 1000)
      self.X_train_subset = X_train[mask]
      self.y_train_subset = y_train[mask]
    else:
      self.X_train_subset = X_train
      self.y_train_subset = y_train

  def on_train_begin(self, logs={}):
    self.train_batch_loss = []
    self.train_acc = []
    self.val_acc = []
    self.relu_out = []

  def on_batch_end(self, batch, logs={}):
    self.train_batch_loss.append(logs.get('loss'))

  def on_epoch_end(self, epoch, logs={}):
    self.relu_out.append(self.get_layer_out())
    val_epoch_acc = logs.get('val_acc')
    self.val_acc.append(val_epoch_acc)
    train_epoch_acc = self.model.evaluate(self.X_train_subset, self.y_train_subset,
                                          show_accuracy=True, verbose=0)[1]
    self.train_acc.append(train_epoch_acc)
    print('(train accuracy, val accuracy): (%.4f, %.4f)' % (train_epoch_acc, val_epoch_acc))


class recordGradients_Gerda(callbacks.Callback):
  """
  Record Gradients (By Gerda)
  """
  def __init__(self, train_set_x, savePaths, model):
    super(callbacks.Callback, self).__init__()

    self.train_set_x = train_set_x
    self.savePaths = savePaths
    self.model = model

  def on_epoch_end(self, epoch, logs={}):

    meanGrad, weights = self.compute_gradients(self.model, self.train_set_x)
    if epoch == 0:
      self.writeNamesCSV(weights)
    self.writeCSV(meanGrad, True if epoch == 0 else False)

  def compute_gradients(self, model, train_set_x):
    # define the function
    weights = model.trainable_weights  # weight tensors
    weights = [weight for weight in weights if model.get_layer(
      weight.name.split('/')[0]).trainable]  # filter down weights tensors to only ones which are trainable
    gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors

    input_tensors = [model.inputs[0],  # input data
                     model.sample_weights[0],  # how much to weight each sample by
                     model.targets[0],  # labels
                     K.learning_phase(),  # train or test mode
                     ]

    get_gradients = K.function(inputs=input_tensors, outputs=gradients)

    # run on the whole epoch and average
    nbr_layers = len(weights)
    meanGrad = np.zeros([1, nbr_layers])
    for image in train_set_x:
      inputs = [np.expand_dims(image, axis=0),  # X
                [1],  # sample weights
                [[1]],  # y
                0  # learning phase in TEST mode
                ]

      grad = get_gradients(inputs)

      # average gradients per layer
      current_epoch = 0
      for i, g in enumerate(grad):
        meanGrad[current_epoch, i] += np.mean(g)

    meanGrad = meanGrad * 1. / len(train_set_x)
    return meanGrad.tolist(), weights

  def writeCSV(self, gradients, rewrite=False):
    for path in self.savePaths:
      with open(os.path.join(path, 'gradients.csv'), "w" if rewrite else "a") as myfile:
        writer = csv.writer(myfile, delimiter=';')
        writer.writerow(gradients)  # gradient should be a list:  dimensions account for layers

  def writeNamesCSV(self, names):
    for path in self.savePaths:
      with open(os.path.join(path, 'gradients.csv'), "a") as myfile:
        writer = csv.writer(myfile, delimiter=';')
        writer.writerow(names)


# record Gradients in layers(By Florian)
class recordGradients_Florian(callbacks.Callback):
  """
  Record Gradients in layers(By Florian)
  """
  def __init__(self, train_set_x, savePaths, model, perSample):
    super(callbacks.Callback, self).__init__()

    self.train_set_x = train_set_x
    self.savePath = savePaths
    self.model = model
    self.perSample = perSample
    cm.mkdir(self.savePath + 'gradientsPerEpoch/')

  def on_epoch_end(self, epoch, logs={}):

    absGrad, layer_names, gradPerSample = self.compute_gradients(self.model, self.train_set_x)

    # save overall gradient
    path_overall = self.savePath + 'gradients.csv'
    if epoch == 0:
      self.writeNamesCSV(layer_names, path_overall)
    self.writeCSV(absGrad, path_overall)

    # save gradients of the current epoch
    if self.perSample:
      path_epoch = self.savePath + 'gradientsPerEpoch' + str(epoch) + '/'
      path_epoch_csv = path_epoch + 'gradients.csv'
      cm.mkdir(path_epoch)
      self.writeNamesCSV(layer_names, path_epoch_csv)
      for i in range(len(self.train_set_x)):
        self.writeCSV(gradPerSample[i], path_epoch_csv)

  def compute_gradients(self, model, train_set_x):
    # define the function
    weights = model.trainable_weights  # weight tensors
    #        weights = [weight for weight in weights if model.get_layer(weight.name.split('/')[0]).trainable]
    #       filter down weights tensors to only ones which are trainable
    gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors

    input_tensors = [model.inputs[0],  # input data
                     model.sample_weights[0],  # how much to weight each sample by
                     model.targets[0],  # labels
                     K.learning_phase(),  # train or test mode
                     ]

    get_gradients = K.function(inputs=input_tensors, outputs=gradients)

    # run on the whole epoch and average
    nbr_layers = len(weights)
    # meanGrad = np.zeros(nbr_layers)
    absGrad = np.zeros(nbr_layers)
    gradPerSample = np.zeros([len(train_set_x), nbr_layers])
    for j, image in enumerate(train_set_x):

      inputs = [np.expand_dims(image, axis=0),  # X
                [1],  # sample weights
                np.ones(np.shape(train_set_x)),  # y
                0  # learning phase in TEST mode
                ]

      grad = get_gradients(inputs)

      # average gradients per layer
      for i, g in enumerate(grad):
        # meanGrad[i] += np.mean(g)
        absGrad[i] += np.sum(np.absolute(g))
        # gradPerSample[j, i] = np.mean(g)
        gradPerSample[j, i] = np.sum(np.absolute(g))

    absGrad = absGrad * 1. / len(train_set_x)
    layer_names = [weight.name.split('/')[0] for weight in weights]
    return absGrad.tolist(), layer_names, gradPerSample

  def writeCSV(self, gradients, pathWrite):
    # with open(pathWrite, "a") as myfile:
    myfile = codecs.open(pathWrite, 'a')
    writer = csv.writer(myfile, delimiter=';')
    writer.writerow(gradients)  # gradient should be a list:  dimensions account for layers

  def writeNamesCSV(self, names, pathWrite):
    # with open(pathWrite, "w") as myfile:
    myfile = codecs.open(pathWrite, 'w')
    writer = csv.writer(myfile, delimiter=';')
    writer.writerow(names)

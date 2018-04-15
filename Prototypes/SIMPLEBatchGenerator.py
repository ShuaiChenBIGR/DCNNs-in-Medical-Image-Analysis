#
# created by
# Antonio Garcia-Uceda Juarez
# PhD student
# Medical Informatics
#
# created on 09/02/2018
# Last update: 09/02/2018
########################################################################################

from CommonUtil.FileDataManager import *
from keras.preprocessing import image
import numpy as np


class SimpleBatchGenerator(image.Iterator):

    def __init__(self, xData, yData, batch_size=32, shuffle=False, seed=None):
        self.xData = xData
        self.yData = yData

        size_data_total = self.xData.shape[0]
        super(SimpleBatchGenerator, self).__init__(size_data_total, batch_size, shuffle, seed)

        #reset
        self.on_epoch_end()


    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        indexes_array = self.index_array[self.batch_size * idx:
                                        self.batch_size * (idx + 1)]

        return (self.xData[indexes_array].astype(dtype=K.floatx()),
                self.yData[indexes_array].astype(dtype=K.floatx()))


    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        with self.lock:
            indexes_array = next(self.index_generator)

        return (self.xData[indexes_array].astype(dtype=K.floatx()),
                self.yData[indexes_array].astype(dtype=K.floatx()))

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]
import trsfile
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

import aes
import constants


class DataGenerator(Sequence):
    
    def __init__(self, tr_files, tr_indices, target, byte_idx, scaler, 
        batch_size, start_sample=None, stop_sample=None, cnn=False, 
        to_fit=True, shuffle_on_epoch_end=True):

        self.tr_files = tr_files
        self.tr_indices = tr_indices # "Rows" to be read from a single .TRS file. len(tr_indices) is the number of traces contained in a single file.
        self.tr_per_file = len(self.tr_indices) // len(tr_files)
        self.batch_size_per_file = batch_size // len(tr_files)
        self.n_batches_per_file = self.tr_per_file // self.batch_size_per_file
        
        self.target = target
        self.byte_idx = byte_idx
        self.scaler = scaler
        self.start_sample = start_sample
        self.stop_sample = stop_sample 
        self.cnn = cnn
        self.to_fit = to_fit
        self.shuffle_on_epoch_end = shuffle_on_epoch_end
         
        self.on_epoch_end()
        
        
    def __len__(self):
        # Generates the number of batches per epoch
        return self.n_batches_per_file
        
        
    def __getitem__(self, index):
        # Generates a batch of data from multiple files
        i = index * self.batch_size_per_file
        batch_indices = self.tr_indices[i : i+self.batch_size_per_file]
        
        x, y = self._read_traces(batch_indices)
        
        if self.cnn:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        
        if self.to_fit:
            return x, y
        else:
            return x
    
    
    def on_epoch_end(self):
        # Eventually shuffles the order of the traces
        if self.shuffle_on_epoch_end:
            np.random.shuffle(list(self.tr_indices))
    
    
    def _read_traces(self, batch_indices):

        x = []
        y = []

        start_none = self.start_sample is None
        stop_none = self.stop_sample is None
        
        for tfile in self.tr_files:
            with trsfile.open(tfile, 'r') as traces:
                batch_traces = [traces[i] for i in batch_indices]
                for tr in batch_traces:
                    if (not start_none) and (not stop_none):
                        s = tr.samples[self.start_sample:self.stop_sample]
                    elif start_none and (not stop_none):
                        s = tr.samples[:self.stop_sample]
                    elif (not start_none) and stop_none:
                        s = tr.samples[self.start_sample:]
                    else:
                        s = tr.samples
                    l = self._retrieve_label(tr)

                    x.append(s)
                    y.append(l)

        x = np.vstack(x)
        x = self.scaler.transform(x)
        y = np.vstack(y)

        return x, y
        
        
    def _retrieve_label(self, tr):
        # Retrieves the label associated to the given trace
        p = np.array(tr.get_input())
        k = np.array(tr.get_key())
        l = aes.labels_from_key(p, k, self.target)
        l = l[self.byte_idx]
        
        l = to_categorical(l, constants.N_CLASSES[self.target])
        
        return l
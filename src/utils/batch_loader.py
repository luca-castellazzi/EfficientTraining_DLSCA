import random
import trsfile
import numpy as np
from contextlib import ExitStack
from tensorflow.keras.utils import to_categorical

import sys
sys.path.insert(0, '../src/utils')
import aes
import constants

class BatchLoader():

    def __init__(self, tr_files, tr_start, tr_tot, batch_size):

        self.tr_files = tr_files

        # With more than one file it is possible that the actual batch_size is
        # different from the original one
        rows_per_file = batch_size // len(tr_files)
        actual_batch_size = rows_per_file * len(tr_files)
        self.actual_batch_num = tr_tot // actual_batch_size

        # Define where each batch starts and also where the last batch ends
        batch_steps = range(tr_start, tr_start+tr_tot+1, rows_per_file)
        # Define the trace indices of each single batch 
        self.batch_indices = [range(batch_steps[i], batch_steps[i+1])
                              for i in range(len(batch_steps)-1)]

        # Define the actual ranges to consider
        #
        # With more than one file the amount of traces forming a batch is spread
        # across multiple files, while batch_indices refers to a single file
        #
        # actual_batch_indices refers to the indices to consider over multiple files
        self.actual_batch_indices = self.batch_indices[:self.actual_batch_num]

    
    def load(self, batch_range_idx, start_sample, stop_sample, target, byte_idx):

        # print(self.actual_batch_indices)
        batch_range = self.actual_batch_indices[batch_range_idx]

        # Open multiple .TRS files
        with ExitStack() as stack:
    
            tracesets = [stack.enter_context(trsfile.open(tr_file, 'r')) 
                         for tr_file in self.tr_files]
    
            # Get the same amount of traces from each file
            # The total amount of traces is the batch_size (which can be slightly
            # different from the original one when multiple files are considered)
            batch_traces = [tset[i] 
                            for i in batch_range
                            for tset in tracesets]

        # Read the traces
        x = [tr.samples[start_sample:stop_sample]
             for tr in batch_traces]
        x = np.vstack(x)
        # x = self.scaler.transform(x)
            
        y = [self._retrieve_label(tr=tr, target=target, byte_idx=byte_idx)
             for tr in batch_traces]
        y = np.vstack(y)
        
        return x, y


    def _retrieve_label(self, tr, target, byte_idx):
        # Retrieves the label associated to the given trace
        p = np.array(tr.get_input())
        k = np.array(tr.get_key())
        l = aes.labels_from_key(p, k, target)
        l = l[byte_idx]
        
        l = to_categorical(l, constants.N_CLASSES[target])
        
        return l


    def shuffle_batch_ranges(self):

        random.shuffle(self.batch_indices)
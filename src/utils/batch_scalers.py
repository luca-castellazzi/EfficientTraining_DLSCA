# Basics
import gc
import trsfile
import numpy as np
from tqdm import tqdm
from contextlib import ExitStack


class BatchScaler():

    def __init__(self, tr_files, batch_indices, tr_len, start_sample, stop_sample):
        
        self.tr_files = tr_files
        self.batch_indices = batch_indices
        self.batch_size = np.vstack(batch_indices).shape[1]
        self.start_sample = start_sample if start_sample is not None else 0
        self.stop_sample = stop_sample if stop_sample is not None else tr_len

        self.tot_samples = self.stop_sample - self.start_sample
        

    def _compute_statistics(self, traces):
        pass
    
    def fit(self):
        pass    

    def transform(self, data):
        pass


class BatchStandardScaler(BatchScaler):
        
    def __init__(self, tr_files, batch_indices, tr_len, start_sample=None, stop_sample=None):
        
        super().__init__(tr_files, batch_indices, tr_len, start_sample, stop_sample)
        self.mean = np.zeros((self.tot_samples, ))
        self.var = np.zeros((self.tot_samples, ))
        

    def fit(self):
            
        for i, batch_range in tqdm(enumerate(self.batch_indices)):
            
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
                
            # Set previous and current number of traces
            m = i * self.batch_size
            n = self.batch_size
            
            # Get current batch of data
            new_x = [tr.samples[self.start_sample:self.stop_sample] 
                     for tr in batch_traces]
            new_x = np.vstack(new_x)
            
            # Consider old mean and compute the current one
            old_mean = self.mean
            new_mean = np.mean(new_x, axis=0)
            
            # Consider old var and compute the current one
            old_var = self.var
            new_var = np.var(new_x, axis=0)
            
            # Update mean and var
            self.mean = (m*old_mean + n*new_mean) / (m+n)
            var_m1 = (m*old_var + n*new_var) / (m+n)
            var_m2 = m * n / (m+n)**2 * (old_mean-new_mean)**2 # Precedence: (), **, *, /
            self.var = var_m1 + var_m2

            # Manually free memory
            del new_x
            gc.collect()
            
        # Compute std
        self.std = np.sqrt(self.var)


    def transform(self, data):

        scaled_data = (data - self.mean) / self.std

        return scaled_data


class BatchMinMaxScaler(BatchScaler):

    def __init__(self, tr_files, batch_indices, tr_len, start_sample=None, stop_sample=None, output_range=(0, 1)):

        super().__init__(tr_files, batch_indices, tr_len, start_sample, stop_sample)
        self.min = [np.inf for _ in range(self.tot_samples)]
        self.max = [-np.inf for _ in range(self.tot_samples)]
        self.range_min, self.range_max = output_range


    def fit(self):

        for i, batch_range in tqdm(enumerate(self.batch_indices)):
            
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
                
            # Set previous and current number of traces
            m = i * self.batch_size
            n = self.batch_size
            
            # Get current batch of data
            new_x = [tr.samples[self.start_sample:self.stop_sample]
                     for tr in batch_traces]
            new_x = np.vstack(new_x)

            # Compute min and max of the batch
            batch_min = np.min(new_x, axis=0)
            batch_max = np.max(new_x, axis=0)

            self.min = [batch_min[i] if batch_min[i] < self.min[i] else self.min[i] 
                        for i in range(len(self.min))]
            self.max = [batch_max[i] if batch_max[i] > self.max[i] else self.max[i] 
                        for i in range(len(self.max))]

            # Manually free memory
            del new_x
            gc.collect()

        self.min = np.array(self.min)
        self.max = np.array(self.max)


    def transform(self, data):

        scaled_data = (data - self.min) / (self.max - self.min)
        scaled_data = scaled_data * (self.range_max - self.range_min) + self.range_min

        return scaled_data
import gc
import trsfile
import numpy as np


class BatchScaler():

    def __init__(self, tr_files, tr_tot, tr_original_len, batch_size, 
                 start_sample, stop_sample):

        self.tr_files = tr_files
        self.tr_per_file = tr_tot // len(tr_files)
        self.batch_size_per_file = batch_size // len(tr_files)
        self.n_batches_per_file = self.tr_per_file // self.batch_size_per_file

        self.start_sample = start_sample if start_sample is not None else 0
        self.stop_sample = stop_sample if stop_sample is not None else tr_original_len

        self.tot_samples = self.stop_sample - self.start_sample


    def _compute_statistics(self, traces):
        pass

    def _adjust_statistics(self):
        # All those operation on statistics that should be done after computing them
        # (e.g., to numpy array, sqrt variance, ...)
        pass
    
    def fit(self):

        for tr_file in self.tr_files:
            with trsfile.open(tr_file, 'r') as traces:    
                self._compute_statistics(traces)
        
        self._adjust_statistics()

    def transform(self, data):
        pass



class BatchStandardScaler(BatchScaler):

    def __init__(self, tr_files, tr_tot, tr_original_len, batch_size, 
                 start_sample=None, stop_sample=None):

        super().__init__(tr_files, tr_tot, tr_original_len, batch_size, 
            start_sample, stop_sample)
        self.mean = np.zeros((self.tot_samples, ))
        self.var = np.zeros((self.tot_samples, ))
        

    def _compute_statistics(self, traces):

        for i in range(self.n_batches_per_file):

            # Set previous and current number of traces
            m = i * self.batch_size_per_file
            n = self.batch_size_per_file
            
            # Get current batch of data
            new_x = [tr.samples[self.start_sample:self.stop_sample] 
                     for tr in traces[m:m+n]]
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


    def _adjust_statistics(self):

        # Compute std from var
        self.std = np.sqrt(self.var) 


    def transform(self, data):

        scaled_data = (data - self.mean) / self.std

        return scaled_data



class BatchMinMaxScaler(BatchScaler):

    def __init__(self, tr_files, tr_tot, tr_original_len, batch_size, 
                 start_sample=None, stop_sample=None, output_range=(0, 1)):

        super().__init__(tr_files, tr_tot, tr_original_len, batch_size, 
            start_sample, stop_sample)
        self.min = [np.inf for _ in range(self.tot_samples)]
        self.max = [-np.inf for _ in range(self.tot_samples)]
        self.range_min, self.range_max = output_range


    def _compute_statistics(self, traces):

        for i in range(self.n_batches_per_file):

            # Set previous and current number of traces
            m = i * self.batch_size_per_file
            n = self.batch_size_per_file
            
            # Get current batch of data
            new_x = [tr.samples[self.start_sample:self.stop_sample] 
                     for tr in traces[m:m+n]]
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


    def _adjust_statistics(self):

        # min and max to numpy array
        self.min = np.array(self.min)
        self.max = np.array(self.max)


    def transform(self, data):

        scaled_data = (data - self.min) / (self.max - self.min)
        scaled_data = scaled_data * (self.range_max - self.range_min) + self.range_min

        return scaled_data
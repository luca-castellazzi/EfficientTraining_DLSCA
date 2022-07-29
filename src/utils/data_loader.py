# Basics
import trsfile
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm

# Custom
import aes
import constants


class DataLoader():

    def __init__(self, dev_key_configs, train_perc=0.8, target='SBOX_OUT'):
    
        self.trace_files = [f'{constants.CURR_TRACES_PATH}/{c}_500MHz + Resampled.trs' 
                            for c in dev_key_configs]
                            
        n_train_traces = int(train_perc * constants.TRACE_NUM)
        self.train_traces_per_dev = int(n_train_traces / len(dev_key_configs))
        
        n_test_traces = constants.TRACE_NUM - n_train_traces
        self.test_traces_per_dev = int(n_test_traces / len(dev_key_configs))
        
        self.target = target
        self.n_classes = constants.N_CLASSES[target]
        
    
    def _retrieve_data(self, tr, byte_idx):

        s = tr.samples # np.array
        p = np.array(tr.get_input()) # int list
        k = np.array(tr.get_key()) # int list
        l = aes.labels_from_key(p, k, self.target) # Compute the set of 16 labels

        if byte_idx is not None:
            l = l[byte_idx]
            p = p[byte_idx]
            k = k[byte_idx]

        l = to_categorical(l, self.n_classes)
        
        return s, l, p, k


    def load_train(self, byte_idx):
        
        samples = []
        labels = []
            
        for tfile in self.trace_files:
            with trsfile.open(tfile, 'r') as traces:
                for tr in tqdm(traces[:self.train_traces_per_dev]):
                    s, l, _, _ = self._retrieve_data(tr, byte_idx)
                    samples.append(s)
                    labels.append(l)
                
        train_data = list(zip(samples, labels))
        random.shuffle(train_data)
        
        samples, labels = zip(*train_data)
        samples = np.array(samples)
        labels = np.array(labels)

        return samples, labels
        
        
    def load_test(self, byte_idx):
        
        samples = []
        labels = []
        pltxt_bytes = []
        true_key_bytes = []
		
        for tfile in self.trace_files:
            with trsfile.open(tfile, 'r') as traces:
                for tr in tqdm(traces[-self.test_traces_per_dev:]):
                    s, l, p, k = self._retrieve_data(tr, byte_idx)
                    samples.append(s)
                    labels.append(l)
                    pltxt_bytes.append(p)
                    true_key_bytes.append(k)
        
        test_data = list(zip(samples, labels, pltxt_bytes, true_key_bytes))
        random.shuffle(test_data)
        
        samples, labels, pltxt_bytes, true_key_bytes = zip(*test_data)
        samples = np.array(samples)
        labels = np.array(labels)
        pltxt_bytes = np.array(pltxt_bytes)
        true_key_bytes = np.array(true_key_bytes)

        return samples, labels, pltxt_bytes, true_key_bytes
	
        
    def load_all(self, byte_idx=None):
        
        samples = []
        labels = []
        pltxt_bytes = []
        true_key_bytes = []
		
        for tfile in self.trace_files:
            with trsfile.open(tfile, 'r') as traces:
                for tr in tqdm(traces):
                    s, l, p, k = self._retrieve_data(tr, byte_idx)
                    samples.append(s)
                    labels.append(l)
                    pltxt_bytes.append(p)
                    true_key_bytes.append(k)
        
        all_data = list(zip(samples, labels, pltxt_bytes, true_key_bytes))
        random.shuffle(all_data)
        
        samples, labels, pltxt_bytes, true_key_bytes = zip(*all_data)
        samples = np.array(samples)
        labels = np.array(labels)
        pltxt_bytes = np.array(pltxt_bytes)
        true_key_bytes = np.array(true_key_bytes)
        
        return samples, labels, pltxt_bytes, true_key_bytes

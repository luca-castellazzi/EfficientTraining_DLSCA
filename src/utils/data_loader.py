# Basics
import trsfile
import numpy as np
from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

# Custom
import aes
import constants


class DataLoader():


    def __init__(self, dev_key_configs, n_tr_per_config, byte_idx=None, target='SBOX_OUT'):
        
        self.trace_files = [f'{constants.CURR_TRACES_PATH}/{c}_500MHz + Resampled.trs' 
                            for c in dev_key_configs]
                            
        self.n_tr_per_config = n_tr_per_config
        
        self.byte_idx = byte_idx
        
        self.target = target
        self.n_classes = constants.N_CLASSES[target]

    # def __init__(self, dev_key_configs, train_perc=0.8, target='SBOX_OUT'):
    
        # self.trace_files = [f'{constants.CURR_TRACES_PATH}/{c}_500MHz + Resampled.trs' 
                            # for c in dev_key_configs]
                            
        # n_train_traces = int(train_perc * constants.TRACE_NUM)
        # self.n_train_traces_per_dev = int(n_train_traces / len(dev_key_configs))
        # self.n_train_traces_per_dev = 10000
        
        # # Assumption: the test-config is always with ONLY 1 DEVICE
        # self.n_test_traces_per_dev = constants.TRACE_NUM - n_train_traces
        
        # self.target = target
        # self.n_classes = constants.N_CLASSES[target]
        
    
    def _retrieve_metadata(self, tr):

        p = np.array(tr.get_input()) # int list
        k = np.array(tr.get_key()) # int list
        l = aes.labels_from_key(p, k, self.target) # Compute the set of 16 labels

        if self.byte_idx is not None:
            l = l[self.byte_idx]
            p = p[self.byte_idx]
            k = k[self.byte_idx]

        l = to_categorical(l, self.n_classes)
        
        return l, p, k
        
        
    def _get_data_from_traces(self):
    
        samples = []
        labels = []
        pltxt_bytes = []
        true_key_bytes = []
        
        for tfile in tqdm(self.trace_files, desc='Loading data: '):
            with trsfile.open(tfile, 'r') as traces:
                for tr in traces[:self.n_tr_per_config]:
                    s = tr.samples
                    l, p, k = self._retrieve_metadata(tr)
                    samples.append(s)
                    labels.append(l)
                    pltxt_bytes.append(p)
                    true_key_bytes.append(k)
                    
        samples = np.array(samples) # (n_tr_per_config*len(train_configs) x trace_len)
        labels = np.array(labels) # (n_traces_per_config*len(train_configs) x n_classes)
        pltxt_bytes = np.array(pltxt_bytes) # (n_traces_per_config*len(train_configs) x 1)
        true_key_bytes = np.array(true_key_bytes) # (n_traces_per_config*len(train_configs) x 1)
        
        return samples, labels, pltxt_bytes, true_key_bytes
        
        
    def load(self):
        
        x, y, pbs, tkbs = self._get_data_from_traces()
        
        to_shuffle = list(zip(x, y, pbs, tkbs))
        random.shuffle(to_shuffle)
        x, y, pbs, tkbs = zip(*to_shuffle)
        
        x = np.array(x)
        y = np.array(y)
        pbs = np.array(pbs)
        if len(self.trace_files) == 1:
            tkbs = tkbs[0] # All true_key_bytes are equal because the train_config is unique 
        else:
            tkbs = np.array(tkbs)
            
        return x, y, pbs, tkbs
        
        
        
class SplitDataLoader(DataLoader):
   
    def __init__(self, dev_key_configs, n_tr_per_config, train_size, byte_idx, target='SBOX_OUT', shuffle=True):
        
        super().__init__(dev_key_configs, n_tr_per_config, byte_idx, target)
        
        self.train_size = train_size
        self.shuffle = shuffle
        
    
    def load(self):
        
        samples, labels, pltxt_bytes, true_key_bytes = self._get_data_from_traces()

        train_test_data = train_test_split(
                        samples,
                        labels,
                        pltxt_bytes,
                        true_key_bytes,
                        train_size=self.train_size,
                        shuffle=self.shuffle)
                        
        # train_test_data = [
        #    samples_train, samples_test,
        #    labels_train,  labels_test,
        #    pbs_train,     pbs_test,
        #    tkbs_train,    tkbs_test
        # ]
        #
        # Even indices = Train data (0, 2, 4, ...)
        # Odd indices = Test data (1, 3, 5, ...)
        
        even_indices = range(0, len(train_test_data), 2)
        odd_indices = range(1, len(train_test_data), 2)
        
        # train_test_data is a list of np.ndarrays of different shapes:
        # in order to cast it to np.array, "dtype=object" is needed
        train_data = np.array(train_test_data, dtype=object)[even_indices]
        test_data = np.array(train_test_data, dtype=object)[odd_indices]
    
        return train_data, test_data
        

    # class AllDataLoader(DataLoader):
        
        # def __init__(self, dev_key_configs, n_tr_per_config, byte_idx=None, target='SBOX_OUT')
            # super().__init__(dev_key_configs, n_tr_per_config, byte_idx, target)
    
    # def get_train_data(self):
    
        # x_train, y_train, pbs_train, tkbs_train = self.train_data
        
        # return x_train, y_train, pbs_train, tkbs_train
        
        
    # def get_test_data(self):
    
        # x_test, y_test, pbs_test, tkbs_test = self.test_data
        
        # return x_test, y_test, pbs_test, tkbs_test
        
        
    # def load_all(self):
        
        # samples = []
        # labels = []
        # pltxt_bytes = []
        # true_key_bytes = []
        
        # for tfile in tqdm(self.trace_files):
            # with trsfile.open(tfile, 'r') as traces:
                # for tr in traces[:self.n_tr_per_config]:
                    # s, l, p, k = self._retrieve_data(tr, None)
                    # samples.append(s)
                    # labels.append(l)
                    # pltxt_bytes.append(p)
                    # true_key_bytes.append(k)
                    
        # samples = np.array(samples)
        # labels = np.array(labels)
        # pltxt_bytes = np.array(pltxt_bytes)
        # true_key_bytes = np.array(true_key_bytes)
        
        # return samples, labels, pltxt_bytes, true_key_bytes
                                                            
        


    # def load_train(self, byte_idx):
        
        # samples = []
        # labels = []
            
        # for tfile in self.trace_files:
            # with trsfile.open(tfile, 'r') as traces:
                # for tr in tqdm(traces[:self.n_train_traces_per_dev]):
                    # s, l, _, _ = self._retrieve_data(tr, byte_idx)
                    # samples.append(s)
                    # labels.append(l)
                
        # train_data = list(zip(samples, labels))
        # random.shuffle(train_data)
        
        # samples, labels = zip(*train_data)
        # samples = np.array(samples)
        # labels = np.array(labels)

        # return samples, labels
        
        
    # def load_test(self, byte_idx):
        
        # samples = []
        # labels = []
        # pltxt_bytes = []
		
        # for tfile in self.trace_files:
            # with trsfile.open(tfile, 'r') as traces:
                # for tr in tqdm(traces[-self.n_test_traces_per_dev:]):
                    # s, l, p, k = self._retrieve_data(tr, byte_idx)
                    # samples.append(s)
                    # labels.append(l)
                    # pltxt_bytes.append(p)
        
        # # Assumption: the test-config is always with ONLY 1 DEVICE
        # # Meaning that the key is ALWAYS ONE (key-byte never changes)
        # test_data = list(zip(samples, labels, pltxt_bytes))
        # random.shuffle(test_data)
        
        # samples, labels, pltxt_bytes = zip(*test_data)
        # samples = np.array(samples)
        # labels = np.array(labels)
        # pltxt_bytes = np.array(pltxt_bytes)

        # return samples, labels, pltxt_bytes, k
	
        
    # def load_all(self, byte_idx=None):
        
        # samples = []
        # labels = []
        # pltxt_bytes = []
        # true_key_bytes = []
		
        # for tfile in self.trace_files:
            # with trsfile.open(tfile, 'r') as traces:
                # for tr in tqdm(traces):
                    # s, l, p, k = self._retrieve_data(tr, byte_idx)
                    # samples.append(s)
                    # labels.append(l)
                    # pltxt_bytes.append(p)
                    # true_key_bytes.append(k)
        
        # all_data = list(zip(samples, labels, pltxt_bytes, true_key_bytes))
        # random.shuffle(all_data)
        
        # samples, labels, pltxt_bytes, true_key_bytes = zip(*all_data)
        # samples = np.array(samples)
        # labels = np.array(labels)
        # pltxt_bytes = np.array(pltxt_bytes)
        # true_key_bytes = np.array(true_key_bytes)
        
        # return samples, labels, pltxt_bytes, true_key_bytes

# Basics
import trsfile
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import random

# Custom
import aes
import constants


class DataLoader():
    
    """
    Trace loading and labeling.
    
    Attributes:
        - trace_files (str list):
            Paths to the trace files.
        - n_tr_per_config (int):
            Number of traces to be collected per different device-key configuration.
        - target (str):
            Target of the attack.
        - n_classes (int):
            Number of possible labels.
        - byte_idx (int, default=None):
            Byte index to consider during the labeling of the traces.
        - scaler (sklearn.preprocessing.StandardScaler):
            Scaler used to scale the values of the traces.
           
    Methods:
        - _retrieve_metadata:
            Collects the plaintext and the key used during encryption and 
            produces the labels.
        - _shuffle:
            Shuffles the traces and their metadata (each trace mantains its own
            metadata).
        - load:
            Retrieves the values of the traces, the plaintexts and the keys and
            labels the traces.
    """


    def __init__(self, trace_files, n_tr_per_config, target, byte_idx=None):
    
        """
        Class constructor: and generates a DataLoader object (most of inputs are 
        not attributes).
        
        Parameters:
            - trace_files (str list):
            Paths to the trace files.
            - n_tr_per_config (int):
                Number of traces to retrieve per configuration file.
            - target (str):
                Target of the attack.
            - byte_idx (int, default=None):
                Byte index considered during the labeling of the traces.
            - mk_traces (bool, default=False):
                MultiKey flag.
                Indicates if the data to retrieve comes from a MultiKey traceset.
        """

        self.trace_files = trace_files
        
        self.n_tr_per_config = n_tr_per_config
        
        self.target = target
        self.n_classes = constants.N_CLASSES[target]
        
        self.byte_idx = byte_idx
        
        self.scaler = StandardScaler()
        
    
    def _retrieve_metadata(self, tr):

        """
        Collects the plaintext and the key used during encryption and produces 
        the labels.
        
        If a not-None byte index is specified in the constructor, then only a 
        single byte of the plaintext and of the key is retrieved.
        
        Parameters:
            - tr (trsfile.trace.Trace):
                Trace whose metadata must be retrieved.
        
        Returns:
            - l, p, k (tuple):
                l is the label associated to the trace.
                p is the plaintext used during the encryption (eventually single
                bytes).
                k is the key used during the encryption (eventually single bytes).
        """
        
        p = np.array(tr.get_input()) # int list
        k = np.array(tr.get_key()) # int list
        l = aes.labels_from_key(p, k, self.target) # Compute the set of 16 labels

        if self.byte_idx is not None:
            l = l[self.byte_idx]
            p = p[self.byte_idx]
            k = k[self.byte_idx]

        l = to_categorical(l, self.n_classes)
        
        return l, p, k
        
        
    @staticmethod
    def _shuffle(x, y, pbs, tkbs):
        
        """
        Shuffles the traces and their metadata (each trace mantains its own
        metadata).
        
        Parameters:
            - x (np.ndarray):
                Values of the traces.
            - y (np.ndarray):
                Labels of the traces (one-hot-encoded).
            - pbs (np.array or np.ndarray):
                Plaintexts (eventually single bytes).
            - tkbs (np.array or np.ndarray):
                Keys (eventually single bytes).
                
        Returns:
            - x, y, pbs, tkbs (tuple):
                Shuffled input where the relation between trace and metadata is
                kept (meaning that each trace is related to its own metadata).
        """

        to_shuffle = list(zip(x, y, pbs, tkbs))
        random.shuffle(to_shuffle)
        x, y, pbs, tkbs = zip(*to_shuffle)
        
        x = np.array(x)
        y = np.array(y)
        pbs = np.array(pbs)
        tkbs = np.array(tkbs)
        
        return x, y, pbs, tkbs
        
        
    def load(self):
    
        """
        Retrieves the values of the traces, the plaintexts and the keys and
        labels the traces.
        
        The values of the traces are scaled with a StandardScaler (resulting 
        data have mean=0, var=1).
        
        Returns:
            - x, y, pbs, tkbs (tuple):
                x contains the scaled values of the traces.
                y contains the one-hot-encoded labels of the traces.
                pbs contains the plaintexts of the traces (eventually single 
                bytes).
                tkbs contains the keys of the traces (eventually single bytes).
        """
    
        samples = []
        labels = []
        pltxt_bytes = []
        true_key_bytes = []
        
        for tfile in self.trace_files:
            with trsfile.open(tfile, 'r') as traces:
                # for tr in traces[:self.n_tr_per_config]:
                for tr in random.sample(list(traces), self.n_tr_per_config):
                    s = tr.samples
                    l, p, k = self._retrieve_metadata(tr)
                    samples.append(s)
                    labels.append(l)
                    pltxt_bytes.append(p)
                    true_key_bytes.append(k)
        
        # Define n_tot_traces as n_tr_per_config * len(trace_files)
        x = np.array(samples) # (n_tot_traces x trace_len)
        x = self.scaler.fit_transform(x)
        y = np.array(labels) # (n_tot_traces x n_classes)
        pbs = np.array(pltxt_bytes) # (n_tot_traces x 1)
        tkbs = np.array(true_key_bytes) # (n_tot_traces x 1)
        
        x, y, pbs, tkbs = self._shuffle(x, y, pbs, tkbs)
        
        if len(self.trace_files) == 1:
            tkbs = tkbs[0] # All true_key_bytes are equal because the config is unique 
            
        return x, y, pbs, tkbs
        
        
class SplitDataLoader(DataLoader):

    """
    Subclass of DataLoader used to directly split data into train and validation
    sets.
    
    Additional parameters:
        - n_train_tr_per_config (int):
            Number of train-traces to be collected per different device-key 
            configuration.

    Overwritten methods:
        - load:
            Retrieves the values of the traces, the plaintexts and the keys and
            labels the traces.
            In addition, splits the data into train and validation sets.
    """
   
    def __init__(self, trace_files, n_tr_per_config, train_size, target, byte_idx=None):
    
        """
        Class constructor: and generates a SplitDataLoader object (most of inputs 
        are not attributes).
        
        Parameters:
            - configs (str list):
                Device-keys configurations used during the encryption.
            - n_tr_per_config (int):
                Number of traces to retrieve per configuration file.
            - train_size (float):
                Size of the train-set expressed as percentage.
            - target (str):
                Target of the attack.
            - byte_idx (int, default=None):
                Byte index considered during the labeling of the traces.
        """
        
        super().__init__(trace_files, n_tr_per_config, target, byte_idx)
        
        self.n_train_tr_per_config = int(train_size * self.n_tr_per_config)
        
        
    def load(self):
    
        """
        Retrieves the values of the traces, the plaintexts and the keys and
        labels the traces.
        In addition, splits the data into train and validation sets.
        
        Proportions are kept in both sets: in the train set there are i traces
        per configuration, while in the validation set there are j < i traces 
        per configuration.
        
        Returns:
            - train_data, val_data (tuple):
                Train set and validation set with values of the traces, labels,
                plaintexts and keys (plaintexts and keys eventually as single 
                bytes).
        """
    
        x_train = []
        y_train = []
        pbs_train = []
        tkbs_train = []
        
        x_val = []
        y_val = []
        pbs_val = []
        tkbs_val = []
        
        for tfile in self.trace_files:
            
            config_s = []
            config_l = []
            config_p = []
            config_k = []
            
            with trsfile.open(tfile, 'r') as traces:
                # for tr in traces[:self.n_tr_per_config]:
                for tr in random.sample(list(traces), self.n_tr_per_config):
                    s = tr.samples
                    l, p, k = self._retrieve_metadata(tr)
                    
                    config_s.append(s)
                    config_l.append(l)
                    config_p.append(p)
                    config_k.append(k)
            
            x_train.append(config_s[:self.n_train_tr_per_config])
            x_val.append(config_s[self.n_train_tr_per_config:])
            
            y_train.append(config_l[:self.n_train_tr_per_config])
            y_val.append(config_l[self.n_train_tr_per_config:])
            
            pbs_train.append(config_p[:self.n_train_tr_per_config])
            pbs_val.append(config_p[self.n_train_tr_per_config:])
            
            tkbs_train.append(config_k[:self.n_train_tr_per_config])
            tkbs_val.append(config_k[self.n_train_tr_per_config:])
        
        # Reduce the lists of arrays to a single np.ndarray
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        pbs_train = np.concatenate(pbs_train)
        tkbs_train = np.concatenate(tkbs_train)
        
        x_val = np.concatenate(x_val)
        y_val = np.concatenate(y_val)
        pbs_val = np.concatenate(pbs_val)
        tkbs_val = np.concatenate(tkbs_val)
        
        # Scale the whole train-set (train + val) with the same scaler
        # Define n_tot_traces as n_tr_per_config * len(trace_files)
        n_tot_train = x_train.shape[0] # train_size * n_tot_traces
        x_tot = np.concatenate([x_train, x_val])
        x_tot = self.scaler.fit_transform(x_tot)
        x_train = x_tot[:n_tot_train]
        x_val = x_tot[n_tot_train:]
        
        # Shuffle the sets
        x_train, y_train, pbs_train, tkbs_train = self._shuffle(x_train, y_train, pbs_train, tkbs_train)
        x_val, y_val, pbs_val, tkbs_val = self._shuffle(x_val, y_val, pbs_val, tkbs_val) 
            
        # Create train and test packages
        train_data = (x_train, y_train, pbs_train, tkbs_train)
        val_data = (x_val, y_val, pbs_val, tkbs_val)
            
        return train_data, val_data
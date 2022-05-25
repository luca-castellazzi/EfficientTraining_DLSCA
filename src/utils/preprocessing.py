import trsfile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import aes128 
from helpers import hex_to_int


class TraceHandler():
    
    """
    Class dedicated to the manipulation and preprocessing of the traces.

    Attributes:
        - traces (np.ndarray):
            values of the trace.
        - plaintexts (np.ndarray):
            plaintexts used to produce each trace.
        - labels (np.ndarray):
            16-bytes labels relative to each plaintext.
        - target (str or TargetEnum):
            target of the attack.

    Methods:
        - get_traces:
            getter for the values of the traceset.
        - get_plaintexts:
            getter for the plaintexts of the traceset.
        - get_labels:
            getter for the 16-bytes labels of the traceset.
        - generate_train_val:
            generation of the train and validation sets splitting 

    """


    def __init__(self, path, target='SBO'):
        
        """
        Class constructor: given a path and a target, retrieve the traces, 
        retrieve the plaintexts and compute the 16-bytes labels for each trace
        w.r.t. the specified target.

        Parameters:
            - path (str):
                path to the traceset to be considered.
            - target (str or TargetEnum, Default: SBO):
                attack target to use during labels computation ('SBO' for SBox
                Output, 'HM' for Hamming Weight of SBox Output)
        """

        self._traces = []
        self._plaintexts = []
        self._labels = []
        self._target = target

        with trsfile.open(path, 'r') as tr_set:
                
            for i, tr in enumerate(tqdm(tr_set, desc='Labeling traces: ')):
                key = np.array(tr.get_key()) # int format by default
                   
                trace = np.array(tr.samples)
                plaintext = np.array(tr.get_input()) # int format by default
                labels = aes128.compute_labels(plaintext, key, self._target) # Compute the set of 16 labels
                
                self._traces.append(trace)
                self._plaintexts.append(plaintext)
                self._labels.append(labels)

        self._traces = np.array(self._traces)
        self._plaintexts = np.array(self._plaintexts)
        self._labels = np.array(self._labels)


    def get_traces(self):

        """
        Getter for the values of the traceset whose path is specified in the 
        constructor.

        Returns:
            float np.ndarray containing the values of the samples of each trace
            in the traceset.
        """

        return np.array(self._traces)


    def get_plaintexts(self):

        """
        Getter for the plaintexts of the traceset whose path is specified in the 
        constructor.

        Returns:
            int np.ndarray containing the int values of the bytes of each plaintext
            in the traceset.
        """

        return np.array(self._plaintexts)


    def get_labels(self):

        """
        Getter for the 16-bytes labels of the traceset whose path is specified
        in the constructor.
        
        Returns:
            int np.ndarray containing the 16-bytes labels of each trace in the 
            traceset.
        """

        return np.array(self._labels)


    def generate_train_val(self, byte_idx, val_perc, shuffle=True, seed=None):

        """
        Generates the train/validation split, where each trace is associated to 
        a single label, the one relative to the specified byte.

        Parameters: 
            - byte_idx (int): 
                0-based index relative to the byte to consider while labeling each trace.
            - val_perc (float): 
                percentage of data to consider as val-set. 
            - shuffle (bool, default: True): 
                whether or not to shuffle the data before splitting.
            - seed (int, default: None):
                value that controls the shuffle operation. 
        
        Returns:
            4-elements tuple containing the values of the train-traces, the values
            of the val-traces, the specific labels of the train-traces and the 
            specific labels of the val-traces (in this order).
        """

        specific_labels = np.array([l[byte_idx] for l in self._labels])

        x_train, x_val, y_train, y_val = train_test_split(self._traces,
                                                          specific_labels,
                                                          test_size=val_perc,
                                                          shuffle=shuffle, 
                                                          random_state=seed)

        return np.array(x_train), np.array(x_val), np.array(y_train), np.array(y_val)


    def generate_test(self, byte_idx):
        
        """ 
        Generates the test-set where each trace is associated to a single label,
        the one relative to the specified byte.

        Parameters:
            - byte_idx (int):
                0-based index relative to the byte to consider while labeling each trace.
        
        Returns:
            2-elements tuple containing the values of the test-traces and the relative
            specific labels.
        """

        specific_labels = np.array([l[byte_idx] for l in self._labels]) 
    
        return self._traces, specific_labels

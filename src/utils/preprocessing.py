import trsfile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import aes128 
from helpers import hex_to_int


class TraceHandler():
    
    """
    Class dedicated to the manipulation and preprocessing of the traces.

    Methods:
        - get_traces:
            getter for the values of the traceset.
        - get_plaintexts:
            getter for the plaintexts of the traceset.
        - get_labels:
            getter for the 16-bytes labels of the traceset.

    """


    def __init__(self, path, target='SBO'):
        
        """
        Class constructor: given a path and a target, retrieve the traces, 
        retrieve the plaintexts and compute the 16-bytes labels for each trace
        w.r.t. the specified target.

        Parameters:
            - path (str):
                path to the traceset to be considered.
            - target (str or TargetEnum):
                attack target to use during labels computation ('SBO' for SBox
                Output, 'HM' for Hamming Weight of SBox Output)
        """

        self.traces = []
        self.plaintexts = []
        self.labels = []
        self.target = target

        with trsfile.open(path, 'r') as tr_set:
                
            for i, tr in enumerate(tqdm(tr_set)):
                key = np.array(tr.get_key()) # int format by default
                   
                trace = np.array(tr.samples)
                plaintext = np.array(tr.get_input()) # int format by default
                labels = aes128.compute_labels(plaintext, key, target) # Compute the set of 16 labels
                
                self.traces.append(trace)
                self.plaintexts.append(plaintext)
                self.labels.append(labels)


    def get_traces(self):

        """
        Getter for the values of the traceset whose path is specified in the 
        constructor.

        Returns:
            float np.ndarray containing the values of the samples of each trace
            in the traceset.
        """

        return np.array(self.traces)


    def get_plaintexts(self):

        """
        Getter for the plaintexts of the traceset whose path is specified in the 
        constructor.

        Returns:
            int np.ndarray containing the int values of the bytes of each plaintext
            in the traceset.
        """

        return np.array(self.plaintexts)


    def get_labels(self):

        """
        Getter for the 16-bytes labels of the traceset whose path is specified
        in the constructor.
        
        Returns:
            int np.ndarray containing the 16-bytes labels of each trace in the 
            traceset.
        """

        return np.array(self.labels)

# -----------------------------------------------------------------------------

class Dataset:
    
    """ 
    A class to represent the whole dataset (both train and test sets).

    Methods:
        - build_train_val:
            Generates the train/validation split where each trace is associated to
            a single label, the one relative a specific byte.
        - build_test:
            Generates the test-set where each trace is associated to a single label, 
            the one relative to a specific byte.
    """

    def __init__(self, train_set_path, test_set_path, target='SBO', metadata=True, 
                 train_plaintext_list=None, test_plaintext_list=None, 
                 train_key=None, test_key=None):
        
        """ 
        Class constructor that collects the data relative to both train and test traces
        and generates the respective 16 labels.

        Parameters: 
            - train_set_path (str): 
                path to the set of traces that will be the train-set. 
            - test_set_path (str): 
                path to the set of traces that will be the test-set.
            - target (str, default 'SBO'): 
                target of the attack (either 'SBO', for SBox Output, or 'HW', 
                for Hamming Weight of the SBox Output).
            - metadata (bool, default True): 
                value to specify if the traces (both train and test) have metadata 
                relative to key and plaintext.
            - train_plaintext_list (str list, default None): 
                hex value of the train plaintexts (one per trace). 
                Useful only if metadata=True.
            - test_plaintext_list (str list, default None): 
                hex value of the test plaintexts (one per trace).
                Useful only if metadata=True.
            - train_key (str, default None):
                hex value of the encryption key used for the train set.
                Useful only if metadata=True.
            - test_key (str, default None): 
                hex value of the encryption key used for the test set.
                Useful only if metadata=True.
        """

        # Train set
        print('Reading train-set traces and extracting labels...')
        self._train_traces, self._train_labels = produce_labeled_traceset(train_set_path, 
                                                                          target,
                                                                          metadata,
                                                                          train_plaintext_list,
                                                                          train_key)
        print('Done')
        print()

        # Test set
        print('Reading test-set traces and extracting labels...')
        self._test_traces, self._test_labels = produce_labeled_traceset(test_set_path, 
                                                                        target,
                                                                        metadata,
                                                                        test_plaintext_list,
                                                                        test_key)
        print('Done')


    def build_train_val(self, byte_idx, train_size, shuffle=True, seed=None):
        
        """
        Generates the train/validation split, where each trace is associated to 
        a single label, the one relative to the specified byte.

        Parameters: 
            - byte_idx (int): 
                0-based index relative to the byte to consider while labeling each trace.
            - train_size (float, between 0 and 1): 
                percentage of data to consider as train-set. 
            - shuffle (bool, default: True): 
                whether or not to shuffle the data before splitting.
            - seed (int, default: None):
                value that controls the shuffle operation 
        
        Returns:
            4-elements tuple containing the values of the train-traces, the values
            of the val-traces, the specific labels of the train-traces and the 
            specific labels of the val-traces (in this order).
        """

        selected_byte_labels = [l[byte_idx] for l in self._train_labels] 
    
        return train_test_split(self._train_traces,     # returns:
                                selected_byte_labels,   # x_train, x_val, y_train, y_val
                                train_size=train_size,
                                shuffle=shuffle,
                                random_state=seed)

    
    def build_test(self, byte_idx):
        
        """ 
        Generates the test-set where each trace, is associated to a single label,
        the one relative to the specified byte.

        Parameters:
            - byte_idx (int):
                0-based index relative to the byte to consider while labeling each trace.
        
        Returns:
            2-elements tuple containing the values of the test-traces and the relative
            specific labels.
        """

        selected_byte_labels = [l[byte_idx] for l in self._test_labels] 
    
        return self._test_traces, selected_byte_labels

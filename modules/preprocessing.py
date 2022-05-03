import trsfile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

########## Enable imports from modules folder ##########
import sys
sys.path.insert(0, '/home/lcastellazzi/DL-SCA/modules')
########################################################

import aes


KEY = [int(c, 16) for c in ['CA', 'FE', 'BA', 'BE', 'DE', 'AD', 'BE', 'EF', '00', '01', '02', '03', '04', '05', '06', '07']]


def produce_labeled_traceset(trace_set_path, target, metadata, plaintext_list, key):

    """ 
    Reads .trs file containing the traces and generates the corresponding
    labels (all 16 bytes) either from traces metadata, or from given 
    plaintexts and key.

    Parameters:
        - trace_set_path (str): 
            path to the trace set to be read
        - target (str): 
            target of the attack (either 'SBO', for SBox Output, or 'HW', 
            for Hamming Weight of the SBox Output).
        - metadata (bool): 
            value to specify if the given traces have metadata relative to key 
            and plaintext.
        - plaintext_list (str list): 
            hex value of the plaintexts (one per trace).
            Useful only if metadata=True.
        - key (str): 
            hex value of the encryption key used for the trace set.
            Useful only if metadata=True.
    
    Returns: 
        2-elements tuple containing the values of each trace and the corresponding 
        set of 16 labels, one per byte of plaintext/key.
        The values and the labels are stored in lists of numpy arrays.
    """

    traces = []
    labels = []

    with trsfile.open(trace_set_path, 'r') as tr_set:
        # UNCOMMENT FOR "Edit parameters" from Inspector
        #if metadata: 
            #tr_set_parameters = tr_set.get_header(trsfile.common.Header.TRACE_SET_PARAMETERS)
            #key = np.array(trace_set_parameters.pop('KEY').value) 
        
        for i, tr in enumerate(tqdm(tr_set)):

            if metadata: # if the traces contain metadata for key and plaintext
                key = np.array(tr.get_key()) # int format by default
                plaintext = np.array(tr.get_input()) # int format by default
            else:
                assert plaintext_list is not None, 'If metadata=False, then provide a plaintext list!'
                assert len(plaintext_list[i]) == 32, 'Plaintext must be 32 characters!'
                plaintext = hex_str_to_int(plaintext_list[i])

                assert key is not None, 'If metadata=False, then provide a key!'
                assert len(key) == 32, 'Key must be 32 characters!'
                key = hex_str_to_int(key)

            tr_labels = aes.compute_labels(plaintext, key, target) # Compute the set of 16 labels
            #tr_labels = aes.compute_labels(plaintext, KEY, target)
            
            traces.append(tr.samples)
            labels.append(tr_labels)
    
    return traces, labels


def hex_str_to_int(hex_str):
    
    """ 
    Converts the given hex number into an array of int, each one relative to 
    a single byte of the input. 

    Parameters: 
        - hex_str (str):
            hex value to be converted.
    
    Returns: 
        Conversion of the given hex value as a list of int, each one relative to
        a single byte of the input.
    """

    split_hex_str = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]

    return [int(sb, 16) for sb in split_hex_str] 


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
        print()

        # Test set
        print('Reading test-set traces and extracting labels...')
        self._test_traces, self._test_labels = produce_labeled_traceset(test_set_path, 
                                                                        target,
                                                                        metadata,
                                                                        test_plaintext_list,
                                                                        test_key)


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

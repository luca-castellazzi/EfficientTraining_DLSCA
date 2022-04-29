import trsfile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Enable imports from modules
import sys
sys.path.insert(0, '/home/lcastellazzi/DL-SCA/modules')

import aes


KEY = [int(c, 16) for c in ['CA', 'FE', 'BA', 'BE', 'DE', 'AD', 'BE', 'EF', '00', '01', '02', '03', '04', '05', '06', '07']]


def produce_labeled_traceset(trace_set_path, target, metadata, plaintext_list, key):

    """ Read .trs file containing the traces and generate the corresponding
    labels (all 16 bytes) either from traces metadata, or from given 
    plaintexts and keys.

    Input: ...
    Output: ...
    """

    traces = []
    labels = []

    with trsfile.open(trace_set_path, 'r') as tr_set:
        # Get key if it is from TraceSet ##########################################################
        for i, tr in tqdm(enumerate(tr_set)):
            
            if metadata:
                # Get key if it is from Trace ##########################################################
                plaintext = np.array(tr.get_input()) # Int format by default
            else:
                assert plaintext_list is not None, 'If metadata=False, then provide a plaintext list!'
                assert len(plaintext_list[i]) == 32, 'Plaintext must be 32 characters!'
                plaintext = hex_str_to_int(plaintext_list[i])

                assert key is not None, 'If metadata=False, then provide a key!'
                assert len(key) == 32, 'Key must be 32 characters!'
                key = hex_str_to_int(key)

            # labels = aes.compute_labels(plaintext, key, target)
            labels_all_bytes = aes.compute_labels(plaintext, KEY, target)
            
            traces.append(tr.samples)
            labels.append(labels_all_bytes)
    
    return traces, labels


def hex_str_to_int(hex_str):
    
    """ Convert a given string corresponding to a hex number into an array
    of int, each one relative to a single byte of the hex input. 

    Input: ...
    Output: ...
    """

    split_hex_str = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]

    return [int(sb, 16) for sb in split_hex_str] 


class Dataset:
    
    """ Dataset class, allows to retrieve traces and corresponding labels
    and to produce train, val and test sets.
    """

    def __init__(self, train_set_path, test_set_path, target='SBO', metadata=True, 
                 train_plaintext_list=None, test_plaintext_list=None, 
                 train_key=None, test_key=None):
        
        """ Class constructor that allows to directly collect the data
        relative to both train and test traces and to generate the respective
        16-bytes labels.

        Input: ...
        Output: ...
        """

        # Train set
        print('Reading train-set traces and extracting all-bytes labels...')
        self._train_traces, self._train_labels = produce_labeled_traceset(train_set_path, 
                                                                          target,
                                                                          metadata,
                                                                          train_plaintext_list,
                                                                          train_key)
        print()

        # Test set
        print('Reading test-set traces and extracting all-bytes labels...')
        self._test_traces, self._test_labels = produce_labeled_traceset(test_set_path, 
                                                                        target,
                                                                        metadata,
                                                                        test_plaintext_list,
                                                                        test_key)
        print()


    def build_train_val(self, byte_idx, train_size, shuffle=True, seed=1234):
        
        """ Generation of the train/validation split where each trace
        is associated to a single label, the one relative to the provided
        byte.

        Input: ...
        Output: ...
        """

        selected_byte_labels = [l[byte_idx] for l in self._train_labels] 
    
        return train_test_split(self._train_traces, 
                                selected_byte_labels, 
                                train_size=train_size,
                                shuffle=shuffle,
                                random_state=seed)

    
    def build_test(self, byte_idx):
        
        """ Generation of the test set where each trace is associated to
        a single label, the one relative to the provided byte.

        Input: ...
        Output: ...
        """

        selected_byte_labels = [l[byte_idx] for l in self._test_labels] 
    
        return self._test_traces, selected_byte_labels

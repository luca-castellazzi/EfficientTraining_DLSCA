import json
import time
import numpy as np
from tensorflow.keras.utils import to_categorical
import random


class DataLoader():

    """
    Class dedicated to load traceset data from a JSON file.

    Attributes:
        - _x:
            samples of the retrieved traces.
        - _y:
            one-hot-encoding of the 16-bytes labels relative to the retrieved 
            traces.
        - _pltxt_bytes:
            specific byte of the plaintexts relative to the retrieved traces.
        - _true_key_byte:
            correct key-byte value for the retrieved traces.
        - _train_len:
            amount of traces dedicated to the train set.

    Methods:
        - get_true_key_byte:
            getter for the correct key-byte.
        - gen_train:
            generates the train set.
        - gen_test:
            generates the test set.
    """
    
    
    def __init__(self, json_path, byte_idx, train_perc=0.8):
        
        """
        Class constructor: given the path of the traceset JSON file, a
        specific byte index and a percentage of traces to be dedicated to the 
        train set, retrieve all data and metadata from the traceset and compute
        the amount of traces to dedicate to the train set.

        Parameters:
            - json_path (str):
                path of the traceset JSON file storing a tracest.
            - byte_idx (int):
                specific byte to consider to retrieve plaintexts bytes and
                correct key-byte.
            - train_perc (float, default: 0.8):
                value in [0, 1] representing the percentage of traces to be 
                considered as train set.
        """

        print('Loading the dataset... ')

        start_time = time.time()
        with open(json_path, 'r') as j_file:
            dataset = json.load(j_file)
        end_time = time.time()

        print(f'Dataset successfully loaded ({end_time-start_time:.2f} seconds).')

        traces = dataset['traces']

        random.seed(24)
        random.shuffle(traces)

        # Samples
        self._x = np.array([tr['samples'] for tr in traces])
        
        # Labels
        labels = [tr['labels'] for tr in traces]
        self._y = np.array([l[byte_idx] for l in labels])
        self._y = to_categorical(self._y)

        # Plaintexts
        plaintexts = [tr['pltxt'] for tr in traces]
        self._pltxt_bytes = np.array([pltxt[byte_idx] for pltxt in plaintexts])

        self._true_key_byte = dataset['key'][byte_idx]
        self._train_len = int(train_perc * len(traces))


    def get_true_key_byte(self):

        """
        Getter for the correct key-byte for the retrieved traces.

        Returns:
            int value representing the correct key-byte.
        """

        return self._true_key_byte
    

    def gen_train(self):

        """
        Generates the train set w.r.t. the percentage of traces specified in the
        constructor.

        Returns:
            2-element tuple containing train-traces values and train-traces
            labels (labels are one-hot-encoded).
        """
            
        x_train = self._x[:self._train_len]
        y_train = self._y[:self._train_len]
        #pltxt_train = self._pltxt_bytes[:self._train_len]

        return x_train, y_train#, pltxt_train


    def gen_test(self):

        """
        Generates the test set w.r.t. the percentage of train-traces specified 
        in the constructor.

        Returns:
            3-element tuple containing test-traces values, test-traces labels
            and test-traces plaintexts (labels are one-hot-encoded and 
            plaintexts relate only to a specific byte).
        """
        
        x_test = self._x[self._train_len:]
        y_test = self._y[self._train_len:]
        pltxt_test = self._pltxt_bytes[self._train_len:]

        return x_test, y_test, pltxt_test

# Basics
import pandas as pd
import time
import numpy as np
from tensorflow.keras.utils import to_categorical
import random
from ast import literal_eval

# Custom
import constants


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
        - gen_set:
            generates the train set or the test set.
    """
    
    
    def __init__(self, json_path, byte_idx=None):
        
        """
        Class constructor: given the path of the traceset JSON file, a
        specific byte index and a percentage of traces to be dedicated to the 
        train set, retrieve all data and metadata from the traceset and compute
        the amount of traces to dedicate to the train set.

        Parameters:
            - json_path (str):
                path of the traceset JSON file storing a tracest.
            - byte_idx (int, default: None):
                specific byte to consider to retrieve plaintexts bytes and
                correct key-byte.
                If None, allows to have all bytes of plaintext and key at once.
            - train_perc (float, default: 0.8):
                value in [0, 1] representing the percentage of traces to be 
                considered as train set.
        """

        self.json_path = json_path
        self.byte_idx = byte_idx

    
    def load_data(self):

        df = pd.read_json(self.json_path)
        
        # X
        x = np.array([df.at[i, 'samples'] 
                      for i in range(df.shape[0])])

        # Labels
        labels = np.array([df.at[i, 'labels'] 
                          for i in range(df.shape[0])])
        y = [l[self.byte_idx] for l in labels]
        y = to_categorical(y)
        
        # Plaintexts
        pltxts = [df.at[i, 'pltxt'] 
                  for i in range(df.shape[0])]
        pltxts_bytes = np.array([p[self.byte_idx] for p in pltxts])   

        # True key-byte
        true_kb = df.at[0, 'key'][self.byte_idx]
        
        return x, y, pltxts_bytes, true_kb

import json
import time
import numpy as np
from tensorflow.keras.utils import to_categorical
import random


class DataLoader():
    
    def __init__(self, json_path, byte_idx, train_perc=0.8):
        
        print('Loading the dataset... ')

        start_time = time.time()
        with open(json_path, 'r') as j_file:
            dataset = json.load(j_file)
        end_time = time.time()

        print(f'Dataset successfully loaded in {end_time-start_time:.2f} seconds.')

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


    def get_train(self):

        x_train = self._x[:self._train_len]
        y_train = self._y[:self._train_len]

        #pltxt_train = self._pltxt_bytes[:self._train_len]

        return x_train, y_train#, pltxt_train


    def get_test(self):
        
        x_test = self._x[self._train_len:]
        y_test = self._y[self._train_len:]
        pltxt_test = self._pltxt_bytes[self._train_len:]

        return x_test, y_test, pltxt_test


    def get_true_key_byte(self):
        return self._true_key_byte

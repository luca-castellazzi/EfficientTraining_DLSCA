import trsfile
import numpy as np
import random
from tqdm import tqdm

import sys
sys.path.insert(0, '../utils')
import aes
import constants
from helpers import to_coords
sys.path.insert(0, '../src/modeling')
from network import Network


class Evaluator():

    def __init__(self, x_test, pltxt_bytes, true_key_byte, model_type, target='SBOX_OUT'):
       
        self._x_test = x_test 
        self._key_bytes = np.array([aes.key_from_labels(pb, target) 
                                    for pb in tqdm(pltxt_bytes, desc='Recovering key-bytes: ')])
        self._true_key_byte = true_key_byte 
        self._model_type = model_type


    @staticmethod
    def compute_key_probs(probs, key_bytes):
        
        # For each array, element in position i is associated to target-byte i

        # probs: ordered from sbox-out=0 to sbox-out=255
        # key_bytes: ordered from sbox-out=0 to sbox-out=255

        # key_probs: ordered from key-byte=0 to key-byte=255

        key_probs = []

        # For each element in the association between key-bytes and sbox-probs...
        for kbs, ps in zip(key_bytes, probs):
            
            # ...associate each sbox-prob to its relative key-byte...
            curr_key_probs = list(zip(kbs, ps))
            
            # ...sort the sbox-probs w.r.t. their relative key-byte...
            curr_key_probs.sort(key=lambda x: x[0])
            
            # ...consider only the sorted predicions to "transform" sbox-probs
            # into key-byte-probs
            curr_key_probs = list(zip(*curr_key_probs))[1]

            key_probs.append(curr_key_probs)
            
        return np.array(key_probs)


    def guessing_entropy(self, n_exp, hp, x_train, y_train, epochs):

        net = Network(self._model_type)
        net.set_hp(hp)
        net.build_model()

        print('Training the model...')
        net.train_model(x_train, 
                        y_train, 
                        epochs=epochs, 
                        verbose=0)
        print('Model successfully trained.')

        
        ranks_per_exp = []
        for _ in tqdm(range(n_exp), desc='Computing GE: '):
            shuffled_data = list(zip(self._x_test, self._key_bytes))
            random.shuffle(shuffled_data)
            x_test_shuffled, key_bytes_shuffled = list(zip(*shuffled_data))
            
            probs = net.predict(np.array(x_test_shuffled))
            key_probs = self.compute_key_probs(probs, key_bytes_shuffled)
        
            log_probs = np.log10(key_probs + 1e-22)
            cum_tot_probs = np.cumsum(log_probs, axis=0)
        
            indexed_cum_tot_probs = [list(zip(range(256), tot_probs))
                                     for tot_probs in cum_tot_probs]
            
            sorted_cum_tot_probs = [sorted(el, key=lambda x: x[1], reverse=True) 
                                    for el in indexed_cum_tot_probs]

            sorted_key_bytes = [[el[0] for el in tot_probs]
                                for tot_probs in sorted_cum_tot_probs]

            true_key_byte_ranks = [skb.index(self._true_key_byte) 
                                   for skb in sorted_key_bytes]
            true_key_byte_ranks = np.array(true_key_byte_ranks)

            ranks_per_exp.append(true_key_byte_ranks)

        ranks_per_exp = np.array(ranks_per_exp)
        ge = np.mean(ranks_per_exp, axis=0)

        return ge

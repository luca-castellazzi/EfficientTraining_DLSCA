from tqdm import tqdm
import numpy as np

import aes128
import constants
from targets import TargetEnum
from helpers import int_to_hex


class Evaluator():
    
    def __init__(self, label_probs, plaintexts, byte_idx, target='SBO'):
        
        # Labels (SBox out) to key-bytes conversion
        self._key_byte_probs = []
        
        if target == TargetEnum.SBO:
            for i, plaintext in enumerate(tqdm(plaintexts, desc='Switching from labels to key-bytes: ')): 
                tmp_dict = {f'{sbox_out ^ plaintext[byte_idx]}': probs 
                            for sbox_out, probs in enumerate(label_probs[i])}
                self._key_byte_probs.append(tmp_dict)
        elif target == TargetEnum.HW: 
            pass ####################################################################################################################


    def rank_key_bytes(self):
        
        # Initialize a tempory dict where to store the probs of each key_byte
        tmp_dict = {str(key_byte): 0.0 for key_byte in range(256)}

        # Sum the probs (log10) relative to the same key-byte
        for probs in tqdm(self._key_byte_probs, desc='Computing final key-byte probabilities: '):
            for key_byte in tmp_dict.keys():
                tmp_dict[key_byte] += np.log10(probs[key_byte] + 1e-22)    

        # Sort the key-bytes w.r.t. their prob
        sorted_key_byte_probs = list(tmp_dict.items())
        sorted_key_byte_probs.sort(key=lambda x: -x[1]) # The "-" is used to sort values from the highest

        # Produce the final ranking of the key-bytes (hex)
        ranking = {int_to_hex([int(key_byte)]): prob for key_byte, prob in sorted_key_byte_probs}
        
        return ranking


    def get_true_key_byte_rank(self, ranking, true_key_byte):

        return list(ranking.keys()).index(true_key_byte)

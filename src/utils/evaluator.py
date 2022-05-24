from tqdm import tqdm
import numpy as np

import aes128
import constants
from targets import TargetEnum


class Evaluator():
    
    def __init__(self, label_probs, plaintexts, byte_idx, target='SBO'):
        
        self._key_byte_probs = []
        
        if target == TargetEnum.SBO:
            for i, plaintext in enumerate(tqdm(plaintexts)): 
                tmp_dict = {f'{sbox_out ^ plaintext[byte_idx]}': probs 
                            for sbox_out, probs in enumerate(label_probs[i])}
                self._key_byte_probs.append(tmp_dict)
        elif target == TargetEnum.HW: 
            pass ####################################################################################################################

        self._ranking = {str(key_byte): 0.0 for key_byte in range(256)}


    def rank_key_bytes(self):

        tmp_dict = {str(key_byte): 0.0 
                    for key_byte in range(256)} # Temporary dict used to store 
                                                # key-bytes and relative prob values

        # Sum the probs (log10) relative to the same key-byte
        for probs in tqdm(self._key_byte_probs):
            for key_byte in tmp_dict.keys():
                tmp_dict[key_byte] += np.log10(probs[key_byte] + 1e-22)    

        # Produce a ranking of the key-bytes (hex) w.r.t. their prob values
        self._ranking = sorted(self._ranking, 
                               key=lambda x: self._ranking[x], 
                               reverse=True)
        self._ranking = [hex(int(val))[2:] for val in key_byte_ranking]


    def get_true_key_byte_rank(self, true_key_byte):
        return self._ranking.index(true_key_byte)

from tqdm import tqdm
import numpy as np

import aes
from helpers import int_to_hex

class SingleByteEvaluator():

    """
    Class dedicated to the evaluation of the model predictions.

    Attributes:
        - _ranking (dict):
            ranking of the key-bytes w.r.t. their prediction value.
        - _mapping (dict list):
            mapping from each plaintext to the relative key-byte probabilities
            (single plaintext-byte).

    Methods:
        - get_ranking:
            getter for the ranking.
        - get_mapping:
            getter for the mapping.
        - get_predicted_key_byte:
            getter for the first key-byte in the ranking.
        - _sum_predictions:
            helper method to sum the predictions of same key-bytes (w.r.t. 
            different plaintexts).
        - _rank_key_bytes:
            helper method to create the ranking starting from the mapping.
        - rank:
            computation of the rank of a given key-byte value (usually the true
            key-byte).
    """
    

    def __init__(self, test_plaintexts, byte_idx, label_preds, target='SBOX_OUT'):
            
        """
        Class constructor: given the test plaintexts, a specific byte index, 
        the computed predictions and a target, produce the mapping from each 
        plaintext to the corresponding key-byte predictions

        Parameters:
            - test_plaintexts (int np.ndarray):
                test plaintexts used to generate the test traces (int values of
                each byte).
            - byte_idx (int):
                specific byte to consider in the evaluation.
            - label_preds (float np.ndarray):
                model predictions to be converted in key-byte predictions.
            - target (str, default: 'SBOX_OUT'):
                target of the attack ('SBOX_OUTPUT' for SBox Output).
                More targets in future (e.g. Hamming Weights, Key, ...).
        """

        self._mapping = [] # Complete mapping from plaintext to key-byte probabilities 
                           # (single plaintext-byte)

        # Create the mapping from plaintext to key-byte probability predictions
        num_labels = len(label_preds[0]) # Number of predicted classes
        for i, plaintext in enumerate(tqdm(test_plaintexts, desc='Switching from labels to key-bytes: ')):
            curr_key_bytes = aes.key_from_labels(plaintext, byte_idx, range(num_labels), target) # 1 x num_labels
            key_bytes_probs = dict(zip(curr_key_bytes, label_preds[i]))

            self._mapping.append(key_bytes_probs)

        self._ranking = {} # structure to save the ranking of all possible key-byte


    def get_mapping(self):

        """
        Getter for the mapping between each test plaintext and the corresponding
        key-byte predictions.

        Returns:
            dict list representing the mapping (w.r.t. the byte specified in the 
            constructor)
        """

        return self._mapping


    def get_ranking(self):

        """
        Getter for the key-byte ranking.

        Returns:
            dict containing, for each possible key-byte value, the corresponding
            prediction value.
            The dict is sorted w.r.t. the prediction value, from the highest to 
            the lowest.
        """

        return self._ranking
    

    def get_predicted_key_byte(self):

        """
        Getter for the predicted key-byte.

        Returns:
            hex key-byte at rank 0 (the one with highest prediction value).
            The returned value is the model's final output.
        """

        predicted_int_key_byte = int(list(self._ranking.keys())[0])
        predicted_key_byte = int_to_hex([predicted_int_key_byte])

        return predicted_key_byte 


    def _sum_predictions(self):
        
        """
        Sums the prediction values relative to the same key-byte (w.r.t. different
        plaintexts).

        Returns:    
            dict containing, for each key-byte, the total prediction value.
        """

        possible_key_byte_values = range(256)
        summed_preds = {kb_val: 0.0 for kb_val in possible_key_byte_values}
        
        for kb_preds in tqdm(self._mapping, desc='Summing the predictions: '):
            for kb_val in possible_key_byte_values: 
                summed_preds[kb_val] += np.log10(kb_preds[kb_val] + 1e-22)
        
        return summed_preds


    def _rank_key_bytes(self):

        """
        Produces the key-byte ranking sorting the key-bytes w.r.t. their total
        prediction value.
        """

        # Sum the obtained predictions
        summed_preds = self._sum_predictions()

        # Sort the summed predictions (high to low)
        sorted_summed_preds = list(summed_preds.items())
        sorted_summed_preds.sort(key=lambda x: -x[1]) # "-" used to sort from 
                                                      # highest to lowest

        # Produce the final ranking of the key-bytes
        for kb, kb_pred in sorted_summed_preds:
            self._ranking[kb] = kb_pred


    def rank(self, key_byte):
        
        """
        Generates the rank position of the specified key-byte.

        Parameters:
            - key_byte (int):
                key_byte whose rank is needed.
                Usually it is the true key-byte of the encryption key used to
                generate the test traces.

        Returns:
            int (from 0 to 255) relative to the rank of the specified key-byte.
        """

        self._rank_key_bytes()

        ranking_list = list(self._ranking.keys())

        return ranking_list.index(key_byte)

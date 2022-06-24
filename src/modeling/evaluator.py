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

    """
    Class dedicated to the evaluation of a model during the attack (test) phase.
    The evaluation is based on the Guessing Entropy (GE), the average rank of the 
    correct key among the predictions.

    Attributes:
        - _x_test (float np.ndarray):
            attack set (test set in DL pipeline).
        - _key_bytes (int np.ndarray):
            key-bytes relative to each possible value of the sbox-output.
        - _true_key_byte (int):
            correct target key-byte.
        - _model_type (str):
            type of model to be tested.

    Methods:
        - compute_key_probs (@staticmethod):
            converts sbox-out probabilities into key-byte probabilities.
        - guessing_entropy:
            computes the GE over a given number of experiments.


    """


    def __init__(self, x_test, pltxt_bytes, true_key_byte, model_type, target='SBOX_OUT'):
       
        """
        Class Constructor: given an attack set with test-traces and plaintexts,
        the correct byte of the encryption key and the model type and target, 
        compute the key-bytes related to each possible sbox-output and store
        attack set, correct encryption key-byte and model type.

        Parameters:
            - x_test (float np.ndarray):
                attack set (test set in DL pipeline).
            - pltxt_bytes (int np.array):
                specific plaintext byte (one specific byte for each attack 
                plaintext).
            - true_key_byte (int):
                correct target key-byte.
            - model_type (str):
                type of model to evaluate ('MLP' for Multi-Layer Perceptron, 
                'CNN' for Convolutional Neural Network).
            - target (str, default: 'SBOX_OUT'):
                target of the attack ('SBOX_OUTPUT' for SBox Output).
                More targets in future (e.g. Hamming Weights, Key, ...).
        """

        self._x_test = x_test 
        self._key_bytes = np.array([aes.key_from_labels(pb, target) 
                                    for pb in tqdm(pltxt_bytes, desc='Recovering key-bytes: ')])
        self._true_key_byte = true_key_byte 
        self._model_type = model_type


    @staticmethod
    def compute_key_probs(probs, key_bytes):
        
        """
        Converts sbox-output probabilities into key-byte probabilities.

        Parameters:
            - probs (float np.ndarray):
                model predictions w.r.t. an attack set  (probabilities of 
                each possible sbox-output).
                Predictions are ordered w.r.t. a given trace and the sbox-outpu
                t they relate to: in position i,j there is the probability that
                sbox-output j is the output for trace i.
            - key_bytes (int np.ndarray):
                key-bytes relative to each possible value of the sbox-output and
                each plaintext of an attack set.
                Key-bytes are ordered w.r.t. a given trace and the sbox-output 
                they relate to: in position i,j there is the key-byte associated
                to sbox-ouptut j, for trace i (meaning plaintext i).

        Returns:
            float np.array containing the probabilities that the output of the 
            model is each possible key-byte.
            The returned array is a permutation of the input array "probs", 
            where the elements are ordered w.r.t. the key-byte they relate to:
            in position i,j there is the probability that key-byte j is the 
            ouput for trace i.
        """


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

        """
        Computes the Guessing Entropy (GE) of the model w.r.t. an incremental 
        number of attack traces over a specified number of experiments.
        An "experiment" is defined as "test of a pre-trained model over a
        permutation of the whole attack set".
        The tested model is always the same (the one with the specified 
        hyperparameters) and it is trained only once over the whole train set 
        before computing the different permutations of the test set.

        Parameters:
            - n_exp (int):
                number of experiments.
            - hp (dict):
                hyperparameters used to generate the model under test.
            - x_train (float np.ndarray):
                whole train set.
            - y_train (0/1 list):
                one-hot-encoding of the labels relative to the whole train set  
                (all 0s but a single 1 in position i to represent label i).
            - epochs (int):
                number of epochs to use during the training of the model under 
                test.

        Returns:
            float np.array containing the GE for incremental number of traces:
            in position i there is the value of GE for i+1 test traces (i from 0).
        """

        # Define and train the model under test
        net = Network(self._model_type)
        net.set_hp(hp)
        net.build_model()

        print('Training the model...')
        net.train_model(x_train, 
                        y_train, 
                        epochs=epochs, 
                        verbose=0)
        print('Model successfully trained.')

        # Compute the ranks of the correct key-byte over different experiments
        ranks_per_exp = []
        for _ in tqdm(range(n_exp), desc='Computing GE: '):
            # Associate the attack traces with the relative key-bytes
            shuffled_data = list(zip(self._x_test, self._key_bytes))
            # Generate a permutation of the attack set
            random.shuffle(shuffled_data)
            x_test_shuffled, key_bytes_shuffled = list(zip(*shuffled_data))
            
            # Generate the sbox-output predictions for the given permutation
            probs = net.predict(np.array(x_test_shuffled))
            # Convert sbox-output predictions into key-byte predictions
            key_probs = self.compute_key_probs(probs, key_bytes_shuffled)
        
            # Compute the total probabilities
            log_probs = np.log10(key_probs + 1e-22)
            cum_tot_probs = np.cumsum(log_probs, axis=0)
        
            # Associate each total probability to the relative key-byte 
            # explicitly
            indexed_cum_tot_probs = [list(zip(range(256), tot_probs))
                                     for tot_probs in cum_tot_probs]
            
            # Sort the total probabilities w.r.t. their value (the higher the 
            # better)
            sorted_cum_tot_probs = [sorted(el, key=lambda x: x[1], reverse=True) 
                                    for el in indexed_cum_tot_probs]

            # Consider only the sorted key-bytes
            sorted_key_bytes = [[el[0] for el in tot_probs]
                                for tot_probs in sorted_cum_tot_probs]

            # Get the ranks of the correct key-byte
            true_key_byte_ranks = [skb.index(self._true_key_byte) 
                                   for skb in sorted_key_bytes]
            true_key_byte_ranks = np.array(true_key_byte_ranks)

            ranks_per_exp.append(true_key_byte_ranks)

        ranks_per_exp = np.array(ranks_per_exp)

        # Compute GE as average of the ranks
        ge = np.mean(ranks_per_exp, axis=0)

        return ge

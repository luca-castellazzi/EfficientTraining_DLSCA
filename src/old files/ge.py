import random
from tqdm import tqdm
import numpy as np

from network import Network

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


def guessing_entropy(x_test, key_bytes, true_key_byte, model_type, n_exp, hp, x_train, y_train, epochs, single_model=True):

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
            float np.artrue_key_byte, ray containing the GE for incremental number of traces:
            in position i there is the value of GE for i+1 test traces (i from 0).
        """

        # If the chosen approach is to consider a single model...
        if single_model:
            # ...then define the model and train it once
            net = Network(model_type)
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
        for i in range(n_exp):

            # If the chosen approach is to consider independent models...
            if not single_model:
                # ...then define the current model and train it
                net = Network(model_type)
                net.set_hp(hp)
                net.build_model()

                print(f'Training model {i+1}/{n_exp}...')
                net.train_model(x_train,
                                y_train,
                                epochs=epochs,
                                verbose=0)
                print(f'Model {i+1}/{n_exp} successfully trained.')

            # Associate the attack traces with the relative key-bytes and
            # generate a permutation
            shuffled_data = list(zip(x_test, key_bytes))
            random.shuffle(shuffled_data)
            x_test_shuffled, key_bytes_shuffled = list(zip(*shuffled_data))

            # Generate the sbox-output predictions for the given permutation
            # and convert them into key-byte predictions
            probs = net.predict(np.array(x_test_shuffled))
            key_probs = compute_key_probs(probs, key_bytes_shuffled)

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
            true_key_byte_ranks = [skb.index(true_key_byte)
                                   for skb in sorted_key_bytes]
            true_key_byte_ranks = np.array(true_key_byte_ranks)

            ranks_per_exp.append(true_key_byte_ranks)

        ranks_per_exp = np.array(ranks_per_exp)

        # Compute GE as average of the ranks
        ge = np.mean(ranks_per_exp, axis=0)

        return ge

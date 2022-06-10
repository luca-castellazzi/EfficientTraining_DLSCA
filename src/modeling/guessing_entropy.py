import sys
from tqdm import tqdm
import numpy as np

# Custom modules from modeling/
from network import Network

# Custom modules from utils/
sys.path.insert(0, '../utils')
from single_byte_evaluator import SingleByteEvaluator


def guessing_entropy(network_type, hp, epochs, x_train_tot, y_train_tot, x_test_tot, 
                     test_plaintexts_tot, true_key_byte, byte_idx, n_exp=100):

    """
    Computes the Guessing Entropy (GE, average rank of the correct key-byte) for
    a specific network type, in the specified conditions.

    Parameters:
        - network_type (str):
            type of network to consider computing the GE ('MLP' for Multi-Layer
            Perceptron, 'CNN' for Convolutional Neural Network).
        - hp (dict):
            hyperparameters used to build the networks.
        - epochs (int, default: 100):
            number of epochs for the training phase of each network.
        - x_train_tot (float np.ndarray):
            values of the traces of the whole train set.
        - y_train_tot (0/1 list):
            one-hot-encoding of the labels relative to the whole train set  
            (all 0s but a single 1 in position i to represent label i).
        - x_test_tot (float np.ndarray):
            values of the traces of the whole test set.
        - test_plaintexts_tot (np.ndarray):
            plaintexts used to generate the whole test set.
        - true_key_byte (int):
            true value of a specific key-byte, part of the key used to generate 
            the whole test set.
        - byte_idx (int):
            int (0 to 255) relative to the index of the provided true key-byte.
        - n_exp (int, default: 100):
            number of independent experiments (model-building + train + test) to
            perform in order to compute the GE.
    
    Returns:
        float np.array containing the GE values (with increasing number of 
        traces).
    """

    num_train_traces = int(len(x_train_tot) / n_exp)
    num_test_traces = int(len(x_test_tot) / n_exp)
    
    ranks = []
    for i in tqdm(range(n_exp)):
        start_train = i * num_train_traces
        stop_train = start_train + num_train_traces
        
        start_test = i * num_test_traces
        stop_test = start_test + num_test_traces

        network = Network(network_type)
        network.set_hp(hp)
        network.build_model()

        x_train = x_train_tot[start_train:stop_train]
        y_train = y_train_tot[start_train:stop_train]
        x_test = x_test_tot[start_test:stop_test]
        test_plaintexts = test_plaintexts_tot[start_test:stop_test]
        
        network.train_model(x_train,
                            y_train,
                            epochs=epochs)

        preds = network.predict(x_test)

        evaluator = SingleByteEvaluator(test_plaintexts=test_plaintexts,
                                        byte_idx=byte_idx,
                                        label_preds=preds)

        exp_ranks = []
        for j in range(num_test_traces):
            n_traces = j + 1
            exp_ranks.append(evaluator.rank(true_key_byte, n_traces))

        exp_ranks = np.array(exp_ranks)
        ranks.append(exp_ranks)

    ranks = np.array(ranks)
    guessing_entropy = np.mean(ranks, axis=0)

    return guessing_entropy

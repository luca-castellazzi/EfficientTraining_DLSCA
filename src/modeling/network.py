################################################################################                                      2 #                                                                              #
#                                                                              # 
# This code is based on the Genetic Algorithm for Hyperparameter Tuning        #
# implementation from harvitronix.                                             #
#                                                                              #
# The reference project is                                                     # 
# https://github.com/harvitronix/neural-network-genetic-algorithm,             # 
# which is licensed under MIT License.                                         #
#                                                                              #
################################################################################


import sys
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

# Custom modules in modeling/
from modeling_helpers import create_callbacks, guessing_entropy

# Custom modules in utils/
sys.path.insert(0, '../utils')
import constants
from single_byte_evaluator import SingleByteEvaluator


class Network():

    """
    Class used to represent a single individual of a population of networks.

    Attributes:
        - _model_type:
            type of model implemented by the individual.
        - _hp_choices:
            hyperparameter space, containing all the possible values for each
            hyperparameter.
        - _hp:
            actual hyperparameters considered by the individual.
        - _model:
            model implemented by the individual (keras.models.Sequential), 
            build with the hyperparameters stored in _hp.

    Methods:
        - set_hp:
            setter for the actual hyperparameters.
        - get_hp:
            getter for the actual hyperparameters. 
        - select_random_hp:
            selection of random hyperparameters values from the hyperparameter 
            space in order to initialize the individual.
        - build_model:
            construction of the model implemented by the individual.
        - train_and_val:
            train and Validation of the model implemented by the individual.
        - final_train:
            final Train of the model implemented by the individual performed
            over the whole train-set (no val).
        - save_model:
            generation of a file containing all model's info.
        - plot_guessing_entropy:
            computation and plot of the Guessing Entropy of the model implemented
            by the individual.
    """


    def __init__(self, model_type, hp_choices):

        """
        Class constructor: given a model type and the hyperparameter space, an
        individual is initialized with those values.
        In addition, the actual hyperparameters and the model implemented by the 
        individual are initialized as empty.

        Parameters:
            - model_type (str):
                type of model implemented by the individual ('MLP' for a 
                Multi-Layer Perceptron, 'CNN' for a Convolutional Neural Network).
            - hp_choices (dict):
                hyperparameter space containing all the possible values for each
                hyperparameter.
        """

        self._model_type = model_type
        self._hp_choices = hp_choices
        self._hp = {}
        self._model = Sequential()


    def set_hp(self, hp):

        """
        Setter for the actual hyperparameter of the model implemented by the 
        individual.

        Parameters:
            - hp (dict):
                structure containing all the specific hyperparameters considered
                by the individual.
        """

        self._hp = hp


    def get_hp(self, hp_name):

        """
        Getter for the actual hyperparameter of the model implemented by the
        individual.
        
        Parameters:
            - hp_name (str):
                name of the hyperparameter to get.

        Returns:
            specified hyperparameter considered by the individual (the type is
            variable due to the different types of the hyperparameters).
        """

        return self._hp[hp_name]


    def select_random_hp(self):

        """
        Initializes randomly the actual hyperparameters considered by the 
        individual w.r.t. the hyperparameter space.
        """

        for hp_name in self._hp_choices:
            self._hp[hp_name] = random.choice(self._hp_choices[hp_name])


    def build_model(self):

        """
        Builds the model implemented by the individual (it can be
        either an MLP or a CNN).
        The following layers are hardcoded due to experimental evidence of good
        performance:
            - MLP:  
                * BatchNormalization after input;
                * BatchNormalization before output;
        """

        if self._model_type == 'MLP':
            # Input
            self._model.add(Dense(constants.TRACE_LEN,
                            kernel_initializer=self._hp['kernel_initializer'],
                            activation=self._hp['activation']))

            # First BatchNorm
            self._model.add(BatchNormalization())

            # Hidden
            for _ in range(self._hp['hidden_layers']):
                self._model.add(Dense(self._hp['hidden_neurons'],
                                kernel_initializer=self._hp['kernel_initializer'],
                                activation=self._hp['activation']))

                # Dropout
                self._model.add(Dropout(self._hp['dropout_rate']))

            # Second BatchNorm
            self._model.add(BatchNormalization())

            # Output
            self._model.add(Dense(256, activation='softmax')) ########################### 256 to be changed if the target is changed (HW, ...)

            # Compilation
            self._model.compile(optimizer=self._hp['optimizer'](learning_rate=self._hp['learning_rate']),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
        else:
            pass # In future there will be CNN


    def train_and_val(self, x_train, y_train, x_val, y_val):

        """
        Trains and validates the model implemented by the individual.
        The validation metric is ... .
        
        Parameters:
            - x_train (float np.array):
                values of the train traces.
            - y_train (0/1 list):
                one-hot-encoding of the train labels (all 0s but a single 1 
                in position i to represent label i).
            - x_val (float np.array):
                values of the val traces.
            - y_val (0/1 list):
                one-hot-encoding of the val labels (all 0s but a single 1 
                in position i to represent label i).

        Returns:
            ...
        """

        callbacks = create_callbacks()

        # Default train and validation (w.r.t accuracy)
        self._model.fit(x_train,
                        y_train,
                        validation_data=(x_val, y_val),
                        epochs=100, # maybe as hp?
                        batch_size=self._hp['batch_size'],
                        callbacks=callbacks,
                        verbose=0)

        # Evaluation of the model over the val data in order to have the overal val-performance (w.r.t. accuracy)
        val_loss, val_acc = self._model.evaluate(x_val,
                                                 y_val,
                                                 verbose=0)
        return val_acc


    def final_train(self, x_train_tot, y_train_tot):
    
        """
        Trains the model implemented by the individual over the whole dataset
        (validation is not considered).

        Parameters:
            - x_train_tot (float np.array):
                values of all traces in the dataset.
            - y_train_tot (0/1 list):
                one-hot-encoding of the labels of all traces in the dataset
                (all 0s but a single 1 in position i to represent label i).
        """

        # Default train (w.r.t accuracy)
        self._model.fit(x_train_tot,
                        y_train_tot,
                        epochs=100, # maybe as hp?
                        batch_size=self._hp['batch_size'], 
                        verbose=1)


    def save_model(self, path):
        
        """
        Saves the model implemented by the individual as a SavedModel file at 
        the specified path.

        Parameters:
            - path (str):
                path where to save the model.
        """

        print('Saving the model...')
        self._model.save(path)


    def plot_guessing_entropy(self, x_train_tot, y_train_tot, x_test, test_plaintexts, true_key_byte, byte_idx, n_exp=10):
        
        """
        Computes and plots the Guessing Entropy (average rank of the correct
        key-byte w.r.t. a certain number of experiments).

        Parameters:
            - ...
        """

        ge = guessing_entropy(self._model, 
                              self._hp['batch_size'], 
                              x_train_tot, 
                              y_train_tot, 
                              x_test, 
                              test_plaintexts, 
                              true_key_byte, 
                              byte_idx, 
                              n_exp)

        plt.plot(ge[:50], marker='o')
        plt.set_title('Guessing Entropy')
        plt.grid()
        plt.show()
    

    def OLD_guessing_entropy(self, x_test, test_plaintexts, true_key_byte, byte_idx, n_exp=10):
        
        """
        Computes and plots the Guessing Entropy (average rank of the correct
        key-byte w.r.t. a certain number of experiments).

        Parameters:
            - ...
        """

        traces_per_exp = int(len(x_test) / n_exp)

        ranks = []
        for i in range(n_exp):
            start = i * traces_per_exp
            end = start + traces_per_exp

            curr_preds = self._model.predict(x_test[start:end])

            curr_plaintexts = test_plaintexts[start:end]
            curr_evaluator = SingleByteEvaluator(test_plaintexts=curr_plaintexts,
                                                 byte_idx=byte_idx,
                                                 label_preds=curr_preds)
            curr_ranks = []
            for j in tqdm(range(traces_per_exp)):
                n_traces = j + 1
                curr_ranks.append(curr_evaluator.rank(true_key_byte, n_traces))

            curr_ranks = np.array(curr_ranks)
            ranks.append(curr_ranks)

        ranks = np.array(ranks)

        guessing_entropy = np.round(np.mean(ranks, axis=0)) # .5 approximated to the next int

        plt.plot(guessing_entropy, marker='o')
        plt.grid()
        plt.show()

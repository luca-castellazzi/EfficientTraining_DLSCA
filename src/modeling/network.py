################################################################################
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom modules in utils/
sys.path.insert(0, '../utils')
import constants
from single_byte_evaluator import SingleByteEvaluator


class Network():

    """
    Class used to represent a single individual of a population of networks.

    Attributes:
        - _network_type:
            type of the network.
        - _hp:
            hyperparameters of the network.
        - _model:
            implementation of the network (keras.models.Sequential), built with 
            the hyperparameters stored in _hp.

    Methods:
        - set_hp:
            setter for the actual hyperparameters.
        - get_hp:
            getter for the actual hyperparameters.
        - build_model:
            construction of the network.
        - train_model:
            train and eventual validation of the network.
        - save_model:
            generation of a file containing all network's info.
        - predict:
            prediction of the labels of a given test set.
         """


    def __init__(self, network_type):

        """
        Class constructor: initialization of a network, given a network type.
        In addition, the actual hyperparameters and the network implementation
        are initialized as empty.

        Parameters:
            - network_type (str):
                type of network ('MLP' for Multi-Layer Perceptron, 'CNN' for
                Convolutional Neural Network).
        """

        self._network_type = network_type
        self._hp = {}
        self._model = Sequential()


    def set_hp(self, hp):

        """
        Setter for the hyperparameters of the network.

        Parameters:
            - hp (dict):
                structure containing all the specific hyperparameters.
        """

        self._hp = hp


    def get_hp(self):

        """
        Getter for the hyperparameters of the network.

        Returns:
            dict containing all the hyperparameters of the model.
        """

        return self._hp


    def build_model(self):

        """
        Builds the actual implementation of the model (either an MLP or a CNN).
        The following layers are hardcoded due to experimental evidence of good
        performance:
            - MLP:  
                * BatchNormalization after input;
                * BatchNormalization before output;
        """

        if self._network_type == 'MLP':
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

    
    @staticmethod
    def create_callbacks(cb):

        """
        Generates a set of callbacks for the network considering the provided
        ones (fixed parameters).

        Parameters:
            - cb (dict):
                set of callbacks to include.

        Returns:
            keras.callbacks.Callback list containing all the specified callbacks.
        """

        callbacks = []
        if len(cb) != 0:
            if cb['es']:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=5))
            if cb['reduceLR']:
                callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                                   factor=0.2,
                                                   patience=3, 
                                                   min_lr=1e-6))
    
        return callbacks


    def train_model(self, x_train, y_train, epochs, verbose=0, cb={}, validate=False, x_val=None, y_val=None):

        """
        Trains and eventually validates the network.
        
        Parameters:
            - x_train (float np.ndarray):
                values of the train traces.
            - y_train (0/1 list):
                one-hot-encoding of the train labels (all 0s but a single 1 
                in position i to represent label i).
            - epochs (int):
                number of epochs of the training phase.
            - verbose (0 or 1, default:0):
                whether or not printing training status.
            - cb (dict, default: {}):
                which callback to add to the model.
            - validate (bool, default: False):
                whether or not performing validation during training.
            - x_val (float np.ndarray, default: None):
                values of the val traces.
            - y_val (0/1 list, default: None):
                one-hot-encoding of the val labels (all 0s but a single 1 
                in position i to represent label i).

        Raises:
            Exception in the particular case of validate=True and x_val/y_val
            still None.

        Returns
            tensorflow.keras.callbacks.History object relative to the performed
            training.
        """

        callbacks = self.create_callbacks(cb)

        if not validate:
            history = self._model.fit(x_train,
                                      y_train,
                                      epochs=epochs, 
                                      batch_size=self._hp['batch_size'],
                                      callbacks=callbacks,
                                      verbose=verbose)
        else:
            try:
                history = self._model.fit(x_train,
                                          y_train,
                                          validation_data=(x_val, y_val),
                                          epochs=epochs,
                                          batch_size=self._hp['batch_size'],
                                          callbacks=callbacks,
                                          verbose=verbose)
            except:
                raise Exception('ERROR: x_val or y_val is None while validate=True')

        return history


    def save_model(self, path):
        
        """
        Saves the implementation of the network as a SavedModel file at the 
        specified path.

        Parameters:
            - path (str):
                path where to save the network.
        """

        print('Saving the model...')
        self._model.save(path)


    def predict(self, x_test):
        
        """
        Gives probabilities for each possible output label, given a test set.

        Parameters:
            - x_test (float np.ndarray):
                values of the test traces.
        
        Returns:
            float np.array containing the probabilities relative to each possible
            output label.
        """

        return self._model.predict(x_test)
    

class Individual(Network):

    def __init__(self, network_type, hp_choices):
        super().__init__(network_type)
        self._hp_choices = hp_choices

    
    def select_random_hp(self):

        """
        Initializes randomly the individual's hyperparameters w.r.t. the 
        hyperparameter space.
        """

        for hp_name in self._hp_choices:
            self._hp[hp_name] = random.choice(self._hp_choices[hp_name])


    def evaluate(self, x_val, y_val):

        """
        Evaluates the individual over a validation set (w.r.t. accuracy).

        Parameters:
            - x_val (np.array):
                values of the val traces.
            - y_val (0/1 list):
                one-hot-encoding of the val labels (all 0s but a single 1
                in position i to represent label i).

        Returns:
            float value representing the overall validation accuracy.
        """

        val_loss, val_acc = self._model.evaluate(x_val,
                                                 y_val,
                                                 verbose=0)
        return val_acc

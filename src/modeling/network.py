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
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom modules in utils/
sys.path.insert(0, '../utils')
import constants


class Network():

    """
    Class used to represent a single individual of a population of networks.

    Attributes:
        - network_type:
            type of the network.
        - hp:
            hyperparameters of the network.
        - model:
            implementation of the network (keras.models.Sequential), built with 
            the hyperparameters stored in _hp.

    Methods:
        - set_hp:
            setter for the actual hyperparameters.
        - get_hp:
            getter for the actual hyperparameters.
        - get_model:
            getter for the model.
        - build_model:
            construction of the network.
        - train_model:
            train and eventual validation of the network.
        - save_model:
            generation of a file containing all network's info.
        - predict:
            prediction of the labels of a given test set.
        - reset:
            resets the network.
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

        self.network_type = network_type
        self.hp = {}
        self.model = Sequential()


    def set_hp(self, hp):

        """
        Setter for the hyperparameters of the network.

        Parameters:
            - hp (dict):
                structure containing all the specific hyperparameters.
        """

        self.hp = hp


    def build_model(self):

        """
        Builds the actual implementation of the model (either an MLP or a CNN).
        The following layers are hardcoded due to experimental evidence of good
        performance:
            - MLP:  
                * BatchNormalization after input;
                * BatchNormalization before output;
        """

        if self.network_type == 'MLP':
    
            # Input
            self.model.add(Dense(constants.TRACE_LEN, activation='relu'))

            # First BatchNorm
            self.model.add(BatchNormalization())

            # Hidden
            for _ in range(self.hp['hidden_layers']):
                self.model.add(Dense(self.hp['hidden_neurons'], activation='relu'))

                # Dropout
                # self.model.add(Dropout(self.hp['dropout_rate']))

            # Second BatchNorm
            # self.model.add(BatchNormalization())

            # Output
            self.model.add(Dense(256, activation='softmax'))
            
            # Compilation
            lr = self.hp['learning_rate']
            if self.hp['optimizer'] == 'sgd':
                opt = SGD(learning_rate=lr)
            elif self.hp['optimizer'] == 'adam':
                opt = Adam(learning_rate=lr)
            else:
                opt = RMSprop(learning_rate=lr)
            
            self.model.compile(
                optimizer=opt, 
                loss='categorical_crossentropy',
                metrics=['accuracy']#, 'recall']
            )
    
        else:
            pass # In future there will be CNN


    def reset(self):
        
        """
        Resets the network.
        """

        self.model = Sequential()

    

class Individual(Network):

    """
    Class used to represent a single hyperparameter configuration during the 
    execution of a Genetic Algorithm.

    Superclass: Network

    Attributes:
        - hp_choices (dict):
            hyperparameter space.
    
    Methods:
        - select_random_hp:
            Initializes randomly the individual's hyperparameters w.r.t. the 
            hyperparameter space.
        - evaluate:
            Evaluates the individual over a validation set (w.r.t. accuracy).
    """ 


    def __init__(self, network_type, hp_choices):
        
        """
        Class constructor: initializes an individual as a Network object with 
        an hyperparameter space as additional attribute.
        """

        super().__init__(network_type)
        self.hp_choices = hp_choices

    
    def select_random_hp(self):

        """
        Initializes randomly the individual's hyperparameters w.r.t. the 
        hyperparameter space.
        """

        for hp_name in self.hp_choices:
            self.hp[hp_name] = random.choice(self.hp_choices[hp_name])


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

        val_loss, val_acc = self.model.evaluate(x_val,
                                                y_val,
                                                verbose=0)
        return val_acc

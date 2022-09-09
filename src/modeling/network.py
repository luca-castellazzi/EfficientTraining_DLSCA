# Basic
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import L1, L2, L1L2

# Custom
import sys
sys.path.insert(0, '../utils')
import aes
import constants


class Network():

    """
    Neural Network implementation.
    
    Attributes:
        - model_type (str):
            Type of Neural Network to implement.
        - hp (dict):
            Neural Network hyperparameters.
        - model (keras.models.Sequential):
            Neural Network model.
            By default, it is an empty Sequential object.
        - callbacks (keras.callbacks list):
            List of callbacks to use during model training.
            By default, EarlyStopping and ReduceLROnPlateau are considered.
            
    Methods:
        - add_checkpoint_callback:
            Adds ModelCheckpoint to the list of callbacks.
        - build model:
            Generates the Neural Network model.
        - _compute_key_preds:
            Converts target-predictions into key-predictions.
        - _compute_final_rankings:
            Generates the final ranking of all possible key-bytes.
        - ge:
            Computes the Guessing Entropy of an attack.
    """

    def __init__(self, model_type, hp):

        """
        Class constructor: takes as input all class attributes and generates a
        Network object.
        """
    
        self.model_type = model_type
        self.hp = hp
        self.model = Sequential()
        
        self.callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=15
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=7,
                min_lr=1e-7),
        ]
        

    def add_checkpoint_callback(self, model_path):
    
        """
        Adds ModelCheckpoint to the list of callbacks.
        
        ModelCheckpoint allows to save the best-performing model during training
        (performance is given by the validation loss (the lower, the better)).
        
        Parameters:
            - model_path (str):
                Path to where to store the model (model is a H5 file).
        """
        
        self.callbacks.append(
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True
            )
        )


    def build_model(self):
    
        """
        Generates the Neural Network model adding layer to the default empty
        Sequential object.
        
        Different models can be generated: MultiLayer Perceptron (MLP) or 
        Convolutional Neural Network (CNN).
        """

        if self.model_type == 'MLP':
        
            #### Architecture: ####
            # Input Dense         #
            # Input BatchNorm     #
            #                     #
            # repeat(             #
            #   Hidden Dropout    #
            #   Hidden Dense      #
            #   Hidden BatchNorm  #
            # )                   # 
            #                     #
            # Output Dropout      #
            # Output Dense        #
            # Output BatchNorm    #
            #######################
    
            # Input Dense
            self.model.add(Dense(constants.TRACE_LEN, activation='relu'))
            # Input BatchNorm
            self.model.add(BatchNormalization())

            # Hidden
            for _ in range(self.hp['hidden_layers']):
                # Hidden Dropout
                self.model.add(Dropout(self.hp['dropout_rate']))
                # Hidden Dense
                self.model.add(Dense(
                    self.hp['hidden_neurons'], 
                    activation='relu',
                    kernel_regularizer=L2(self.hp['l2']))
                )
                # Hidden BatchNorm
                self.model.add(BatchNormalization())

            # Output
            # Output Dropout
            self.model.add(Dropout(self.hp['dropout_rate']))
            # Output Dense with BatchNorm before activation
            self.model.add(Dense(256))
            self.model.add(BatchNormalization())
            self.model.add(Activation('softmax'))
            
            # Compilation
            lr = self.hp['learning_rate']
            if self.hp['optimizer'] == 'adam':
                opt = Adam(learning_rate=lr)
            elif self.hp['optimizer'] == 'rmsprop':
                opt = RMSprop(learning_rate=lr)
            else:
                opt = SGD(learning_rate=lr)
            
            self.model.compile(
                optimizer=opt, 
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    
        else:
            pass # In future there will be CNN


    def _compute_key_preds(self, preds, key_bytes):
    
        """
        Converts target-predictions into key-predictions (key-byte).

        Parameters:
            - preds (np.ndarray):
                Target-predictions (probabilities for each possible target value).
            - key_bytes (np.array):
                Key-bytes relative to each possible value in the target-predictions.
                Target-predictions cover all possible values (1 to 255), and
                each value leads to a differnt key-byte.
        
        Returns:
            - key_preds (np.array):
               Key-predictions (probabilities for each possible key-byte value).
               Sorted from key-byte=0 to key-byte=255.
        """
        
        # Associate each sbox-out prediction with its relative key-byte
        association = list(zip(key_bytes, preds))
        
        # Sort the association w.r.t. key-bytes (0 to 255, for alignment within all traces)
        association.sort(key=lambda x: x[0])
        
        # Consider the sorted sbox-out predictons as key-byte predictons
        key_preds = list(zip(*association))[1]
        
        return key_preds
        
        
    def _compute_final_rankings(self, key_preds):
    
        """
        Generates the ranking of the key-bytes starting from key-predictions.
        
        Parameters:
            - key_preds (np.ndarray):
                Predictions relative to the key-byte (probabilities relative to
                each possible key-byte).
                
        Returns:
            - sorted_kbs (np.array):
                Ranking of the possible key-bytes (from the most probable to the 
                least probable).
        """
    
        log_probs = np.log10(key_preds + 1e-22) # n_traces x 256
        
        cum_tot_probs = np.cumsum(log_probs, axis=0) # n_traces x 256
        
        indexed_cum_tot_probs = [list(zip(range(256), tot_probs))
                                 for tot_probs in cum_tot_probs] # n_traces x 256 x 2
        
        sorted_cum_tot_probs = [sorted(el, key=lambda x: x[1], reverse=True)
                                for el in indexed_cum_tot_probs] # n_traces x 256 x 2
                                
        final_rankings = [[el[0] for el in tot_probs]
                          for tot_probs in sorted_cum_tot_probs] # n_traces x 256
                      
        return final_rankings


    def ge(self, preds, pltxt_bytes, true_key_byte, n_exp, n_traces, target):
    
        """
        Computes the Guessing Entropy of an attack as the average rank of the 
        correct key-byte among the predictions.
        
        Parameters: 
            - preds (np.ndarray):
                Target predictions.
            - pltxt_bytes (np.array):
                Plaintext used during the encryption (single byte).
            - true_key_byte (int):
                Actual key used during the encryption (single byte).
            - n_exp (int):
                Number of experiment to compute the average value of GE.
            - n_traces (int):
                Number of traces to consider during GE computation.
            - target (str):
                Target of the attack.
                
        Returns:
            - ge (np.array):
                Guessing Entropy of the attack.
        """
        
        # Consider all couples predictions-plaintext
        all_preds_pltxt = list(zip(preds, pltxt_bytes))
        
        ranks_per_exp = []
        for _ in range(n_exp):
            
            # During each experiment:
            #   * Consider only a fixed number of target-predictions
            #   * Retrieve the corresponding key-predictions
            #   * Compute the final key-predictions
            #   * Rank the final key-predictions
            #   * Retrieve the rank of the correct key (key-byte)
            
            sampled = random.sample(all_preds_pltxt, n_traces)
            sampled_preds, sampled_pltxt_bytes = list(zip(*sampled))
            
            if target == 'KEY':
                # If the target is 'KEY', then key_preds is directly sampled_preds
                # because sampled_preds contains predictions related to each key-byte
                # already in order
                key_preds = preds
            else: 
                # SBOX-IN, SBOX-OUT, ... need further computations
                key_bytes = [aes.key_from_labels(pb, target) for pb in sampled_pltxt_bytes]
            
                key_preds = np.array([self._compute_key_preds(ps, kbs) 
                                      for ps, kbs in zip(sampled_preds, key_bytes)])
            
            # Compute the final ranking (for increasing number of traces)
            final_ranking = self._compute_final_rankings(key_preds)
            
            # Retrieve the rank of the true key-byte (for increasing number of traces)
            true_kb_ranks = np.array([kbs.index(true_key_byte)
                                      for kbs in final_ranking])

            ranks_per_exp.append(true_kb_ranks)
            
        # After the experiments, average the ranks
        ranks_per_exp = np.array(ranks_per_exp)
        ge = np.mean(ranks_per_exp, axis=0)
        
        return ge

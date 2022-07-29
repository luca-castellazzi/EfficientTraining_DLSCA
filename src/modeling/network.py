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


# Basic
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L1, L2, L1L2

# Custom
import sys
sys.path.insert(0, '../utils')
import aes
import constants


class Network():

    def __init__(self, model_type, hp):

        self.model_type = model_type
        self.hp = hp
        self.model = Sequential()
        

    def build_model(self):

        if self.model_type == 'MLP':
    
            # Input
            self.model.add(Dense(constants.TRACE_LEN, activation='relu'))

            # First BatchNorm
            if self.hp['first_batch_norm']:
                self.model.add(BatchNormalization())

            # Hidden
            for _ in range(self.hp['hidden_layers']):
                self.model.add(Dense(
                    self.hp['hidden_neurons'], 
                    activation='relu',
                    kernel_regularizer=L1L2(l1=self.hp['l1'], l2=self.hp['l2']))
                )

                # Dropout
                self.model.add(Dropout(self.hp['dropout_rate']))

            # Second BatchNorm
            if self.hp['second_batch_norm']:
                self.model.add(BatchNormalization())

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


    def _target_to_key(self, preds, key_bytes, process='ge'):
        
        # Associate each sbox-out prediction with its relative key-byte
        association = list(zip(key_bytes, preds))
        
        if process == 'ge':
            # Sort the association w.r.t. key-bytes (alignment for all traces)
            association.sort(key=lambda x: x[0])
            # Consider the sorted sbox-out predictons as key-byte predictons
            key_preds = list(zip(*association))[1]
            res = key_preds
        else:
            # Sort the association w.r.t. sbox-out predictions (higher to lower)
            association.sort(key=lambda x: x[1], reverse=True)
            # Get the rank of the true key-byte
            key_ranking = list(zip(*association))[0]
            res = key_ranking
        
        return res
        

    #def true_rank(self, preds, pltxt_byte, true_key_byte, target='SBOX_OUT'):
    #
    #    key_bytes = aes.key_from_labels(pltxt_byte, target)
    #    
    #    # Retrieve key-byte predicitons
    #    key_ranking = self._target_to_key(preds, key_bytes, process='ranking')
    #    
    #    # Get the rank of the true key-byte
    #    tkb_rank = key_ranking.index(true_key_byte)
    #    
    #    return tkb_rank
        
        
    def _compute_final_ranking(self, key_preds):
    
        log_probs = np.log10(key_preds + 1e-22)
        cum_tot_probs = np.cumsum(log_probs, axis=0)
        
        indexed_cum_tot_probs = [list(zip(range(256), tot_probs))
                                 for tot_probs in cum_tot_probs]
                                
        sorted_cum_tot_probs = [sorted(el, key=lambda x: x[1], reverse=True)
                                for el in indexed_cum_tot_probs]
                                
        sorted_kbs = [[el[0] for el in tot_probs]
                      for tot_probs in sorted_cum_tot_probs]
                      
        return sorted_kbs


    def ge(self, preds, pltxt_bytes, true_key_byte, n_exp, n_traces, target='SBOX_OUT'):
        
        # Consider all couples (predicitons, plaintext byte)
        # During each experiment, sample n_traces couples
        # Perform all GE-related computations w.r.t. the sampled couples only
        
        
        all_preds_pltxt = list(zip(preds, pltxt_bytes))
        
        ranks_per_exp = []
        for _ in range(n_exp):
            
            sampled = random.sample(all_preds_pltxt, n_traces)
            sampled_preds, sampled_pltxt_bytes = list(zip(*sampled))
            
            key_bytes = [aes.key_from_labels(pb, target) for pb in sampled_pltxt_bytes]
        
            key_preds = np.array([self._target_to_key(ps, kbs, process='ge') 
                                  for ps, kbs in zip(sampled_preds, key_bytes)])
                                
            final_ranking = self._compute_final_ranking(key_preds)
            
            true_kb_ranks = np.array([kbs.index(true_key_byte)
                                      for kbs in final_ranking])
            
            ranks_per_exp.append(true_kb_ranks)
            
        ranks_per_exp = np.array(ranks_per_exp)
        ge = np.mean(ranks_per_exp, axis=0)
        
        return ge

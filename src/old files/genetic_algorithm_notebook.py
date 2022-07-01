# In[1]:


# Basics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# TensorFlow/Keras (Keras layers below)
from tensorflow.keras.utils import set_random_seed, to_categorical
# set_random_seed(1234) # set the seeds for Python, NumPy, and TensorFlow in order to reproduce the results

# Custom
#import sys
#sys.path.insert(0, '/home/lcastellazzi/MDM32/src/utils')
from preprocessing import TraceHandler
from helpers import create_callbacks
from genetic_tuner import GeneticTuner


# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs


# In[2]:

def main():
    train_th = TraceHandler('/prj/side_channel/PinataTraces/CURR/D1-K1_50k_500MHz + Resampled at 168MHz.trs')

    BYTE_IDX = 0
    N_CLASSES = 256
    VAL_PERC = 0.1

    x_train_tot = train_th.get_traces()
    y_train_tot = train_th.get_specific_labels(BYTE_IDX)
    y_train_tot = to_categorical(y_train_tot, N_CLASSES)

    x_train, x_val, y_train, y_val = train_th.generate_train_val(BYTE_IDX, val_perc=VAL_PERC)
    y_train = to_categorical(y_train, N_CLASSES)
    y_val = to_categorical(y_val, N_CLASSES)


# In[3]:


    from tensorflow.keras.optimizers import Adam, RMSprop

    HP_CHOICES = {'kernel_initializer': ['random_normal', 'he_normal'],
                  'activation':         ['relu', 'tanh'],
                  'hidden_layers':      [1, 2, 3, 4, 5],
                  'hidden_neurons':     [100, 200, 300, 400, 500],
                  'dropout_rate':       [0.0, 0.2, 0.4],
                  'optimizer':          [Adam, RMSprop],
                  'learning_rate':      [1e-3, 1e-4, 1e-5]}  


# In[ ]:


    GENERATIONS = 10

    gt = GeneticTuner(hp_choices=HP_CHOICES, pop_size=6)
    population = gt.populate()

    for i in range(GENERATIONS):
        print()
        print(f'-------------------- Generation {i+1}/{GENERATIONS} --------------------')
    
        evaluation = gt.evaluate(population, x_train, y_train, x_val, y_val) # list of tuple Network-val_acc
    
        parents = gt.select(evaluation) # list of Network objects
    
        if i != GENERATIONS - 1:
            population = gt.evolve(parents) # list of Network objects
    
        print()
        print(f'Generation {i+1}/{GENERATIONS} Evaluation: {[ev[1] for ev in evaluation]}')


# In[5]:


# evaluation.sort(key=lambda x: -x[1])

    best_individual, best_val_acc = evaluation[0]

    print(f'Best val_acc: {best_val_acc}')


# In[9]:


# test_th = TraceHandler('/prj/side_channel/PinataTraces/CURR/D1-K2_50k_500MHz + Resampled at 168MHz.trs')
    test_th = TraceHandler('/prj/side_channel/PinataTraces/CURR/D2-K1_50k_500MHz + Resampled at 168MHz.trs')

    x_test, y_test = test_th.generate_test(BYTE_IDX) 
    y_test = to_categorical(y_test, N_CLASSES)

    test_plaintexts = test_th.get_plaintexts()
    true_key_byte = test_th.get_key()[BYTE_IDX]


# In[10]:


    x_train_tot = train_th.get_traces()
    y_train_tot = train_th.get_specific_labels(BYTE_IDX)
    y_train_tot = to_categorical(y_train_tot, N_CLASSES)

    best_individual.final_train(x_train_tot, y_train_tot)


# In[11]:


    best_individual.plot_guessing_entropy(x_test, y_test, test_plaintexts, true_key_byte, BYTE_IDX)


if __name__ == '__main__':
    main()

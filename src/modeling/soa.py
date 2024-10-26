# Basic
from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# Custom
import sys
sys.path.insert(0, '../utils')
import constants

SOA_EPOCHS = 50
SOA_BATCH_SIZE = 256


def build_model(layers, neurons):
    
    """
    Reproduces "Mind the Portability" MLP construction.
    
    Parameters:
        - layers (int):
            Number of MLP total layers.
        - neurons (int):
            Number of MLP neurons per layer.
    """

    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(constants.TRACE_LEN,)))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(256, activation='softmax'))

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def hp_tuning(x_train, y_train, x_val, y_val):
    
    """
    Reproduces "Mind the Portability" HP Tuning.
    
    Parameters:
        - x_train, x_val (np.ndarray):
            Train and validation samples.
        - y_train, y_val (np.ndarray):
            Train and validation labels.
    """

    res = []
    for layers in tqdm([1, 2, 3, 4], desc='HP Tuning: '):
        for neurons in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:

            model = build_model(layers, neurons)
            history = model.fit(
                x_train, 
                y_train, 
                validation_data=(x_val, y_val),
                epochs=SOA_EPOCHS,
                batch_size=SOA_BATCH_SIZE,
                verbose=0
            ).history

            val_acc = history['val_accuracy'][-1]

            res.append((val_acc, layers, neurons))
    
    res.sort(key=lambda x: x[0], reverse=True)

    chosen_hp = {'layers': res[0][1], 'neurons': res[0][2]}
    
    return chosen_hp


def fit(model, x_train, y_train, x_val, y_val):

    """
    Trains a model with "Mind The Portability" train-parameters.
    
    Parameters:
        - x_train, x_val (np.ndarray):
            Train and validation samples.
        - y_train, y_val (np.ndarray):
            Train and validation labels.

    Returns:
        - history (dict):
            Training history.
    """

    history = model.fit(
        x_train, 
        y_train, 
        validation_data=(x_val, y_val),
        epochs=SOA_EPOCHS,
        batch_size=SOA_BATCH_SIZE,
        verbose=0
    ).history

    return history
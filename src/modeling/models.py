# Basic
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import L2

# # Custom
# import sys
# sys.path.insert(0, '../utils')
# import constants


OPTIMIZERS = {
    'adam': Adam,
    'rmsprop': RMSprop,
    'sdg': SGD
}


def mlp(hp):

    """
    Builds and compiles an MLP network to attack Unmasked-AES-128, targeting SBOX_OUT.

    Parameters:
        - hp (dict):
            Hyperparameters of the network.
            Considered Hyperparameters are:
                * dropout_rate:  Dropout Rate
                * add_hlayers:   Number of Additional Hidden Layers (1 hidden layer by default)
                * add_hneurons:  Number of Neurons for the additional hidden layers
                * add_hl2:       L2-Regularization Coefficient for the additional hidden layers
                * learning_rate: Learning Rate
                * optimizer:     Optimizer
                * batch_size:    Batch Size

    Network Architecture:
        * Input
        * Dense
        * BatchNorm

        * (
            Hidden Dense
            Hidden BatchNorm
            Hidden Dropout
          )

        * Dense
        * BatchNorm
        * Softmax 
    """


    model = Sequential()

    # Input + First Hidden Dense
    model.add(Dense(1183, activation='relu', input_shape=(1183,)))
    # BatchNorm
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(hp['dropout_rate']))

    # Additional Hidden Layers
    for _ in range(hp['add_hlayers']):
        # Dense with L2-Regularization
        model.add(
            Dense(
                hp['add_hneurons'], 
                activation='relu',
                kernel_regularizer=L2(hp['add_hl2'])
            )
        )
        # BatchNorm
        model.add(BatchNormalization())
        # Dropout
        model.add(Dropout(hp['dropout_rate']))

    # Output Dense
    model.add(Dense(256))
    # BatchNorm (before activation)
    model.add(BatchNormalization())
    # Softmax
    model.add(Activation('softmax'))


    lr = hp['learning_rate']
    opt_str = hp['optimizer']
    
    model.compile(
        optimizer=OPTIMIZERS[opt_str](learning_rate=lr), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
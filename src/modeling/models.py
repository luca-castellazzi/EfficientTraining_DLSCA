# Basic
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import L2

# # Custom
# import sys
# sys.path.insert(0, '../utils')
# import constants


def mlp(hp):

    """
    Builds and compiles an MLP network to attack Unmasked-AES-128, targeting SBOX_OUT.

    Parameters:
        - hp (dict):
            Hyperparameters of the network.
            Considered Hyperparameters are:
                * Dropout Rate
                * Number of Additional Hidden Layers (1 hidden layer by default)
                * Number of Neurons for the additional hidden layers
                * L2-Regularization Coefficient for the additional hidden layers
                * Learning Rate
                * Optimizer
                * Batch Size

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
                kernel_regularizer=L2(hp['hl2'])
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
    opt = hp['optimizer'](learning_rate=lr)
    
    model.compile(
        optimizer=opt, 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
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
    'sgd': SGD
}


def mlp(hp, input_len, metrics=['accuracy']):

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
        - input_len (int):
            Length of the input trace (number of neurons in the input layer).
        - metrics (str list, default: ['accuracy']):
            List of metrics to compute during training.

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
    model.add(Dense(input_len, activation='relu', input_shape=(input_len,)))
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
        metrics=metrics
    )

    return model


# Masked-AES ###################################################################
def msk_mlp(hp, input_len, n_classes, metrics=['accuracy']):

    """
    Builds and compiles an MLP network to attack Unmasked-AES-128, targeting SBOX_OUT.
    The difference with mlp() function is that here the first hidden layer can have
    any number of neurons. In mlp() the number of neurons of the first hidden layer
    is exactly the length of the input.

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
        - input_len (int):
            Length of the input trace (number of neurons in the input layer).
        - n_classes (ing):
            Number of output neurons (number of classes in the problem).
        - metrics (str list, default: ['accuracy']):
            List of metrics to compute during training.

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
    model.add(Dense(hp['add_hneurons'], activation='relu', input_shape=(input_len,)))
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
    model.add(Dense(n_classes))
    # BatchNorm (before activation)
    model.add(BatchNormalization())
    # Softmax
    model.add(Activation('softmax'))


    lr = hp['learning_rate']
    opt_str = hp['optimizer']
    
    model.compile(
        optimizer=OPTIMIZERS[opt_str](learning_rate=lr), 
        loss='categorical_crossentropy',
        metrics=metrics
    )

    return model


def msk_cnn(hp, input_len, n_classes, metrics=['accuracy']):
    
    model = Sequential()

    model.add(
        Conv1D(
            filters=hp['filters'], 
            kernel_size=hp['filter_size'], 
            strides=hp['strides'],
            activation='relu', 
            input_shape=(input_len, 1)
        )
    )
    model.add(MaxPooling1D(pool_size=hp['pool_size']))

    for i in range(hp['add_hlayers']):
        coeff = i + 2
        model.add(
            Conv1D(
                filters=coeff*hp['filters'], 
                kernel_size=hp['filter_size'], 
                strides=hp['strides'],
                activation='relu'
            )
        )
        model.add(MaxPooling1D(pool_size=hp['pool_size']))
        model.add(Dropout(hp['dropout_rate']))

    model.add(Flatten())
    for i in range(hp['fc_layers']):
        model.add(Dense(hp['fc_neurons'], activation='relu'))
        model.add(Dropout(hp['dropout_rate']))
    model.add(Dense(n_classes, activation='softmax'))


    lr = hp['learning_rate']
    opt_str = hp['optimizer']
    
    model.compile(
        optimizer=OPTIMIZERS[opt_str](learning_rate=lr), 
        loss='categorical_crossentropy',
        metrics=metrics
    )

    return model
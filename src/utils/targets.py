import enum
import numpy as np


def hamming_weights(int_values):
    
    """
    Computes the Hamming Weights of the given int values.
    The Hamming Weight of a int value X is the number of 1s in the binary-conversion
    of X.
  
    Parameters:
        - int_values (int numpy array or list): 
            int values whose Hamming Weights should be computed.
  
    Returns:
        int list containing the Hamming Weights of the input values.
    """
  
    bin_values = [np.binary_repr(val) for val in int_values]
  
    return [bin_val.count('1') for bin_val in bin_values]


class TargetEnum(enum.Enum):

    """
    Enumeration class containing the supported targets of an attack.

    Attributes:
        - SBO: SBox Output
        - HW: Hamming Weight of the SBox Output
    """

    SBO = 'SBO'
    HW = 'HW'

import numpy as np

from targets import TargetEnum


SBOX_DEC = np.array([ # Int-version of the AES SBox 
    [ 99, 124, 119, 123, 242, 107, 111, 197,  48,   1, 103,  43, 254, 215, 171, 118],
    [202, 130, 201, 125, 250,  89,  71, 240, 173, 212, 162, 175, 156, 164, 114, 192],
    [183, 253, 147,  38,  54,  63, 247, 204,  52, 165, 229, 241, 113, 216,  49,  21],
    [  4, 199,  35, 195,  24, 150,   5, 154,   7,  18, 128, 226, 235,  39, 178, 117],
    [  9, 131,  44,  26,  27, 110,  90, 160,  82,  59, 214, 179,  41, 227,  47, 132],
    [ 83, 209,   0, 237,  32, 252, 177,  91, 106, 203, 190,  57,  74,  76,  88, 207],
    [208, 239, 170, 251,  67,  77,  51, 133,  69, 249,   2, 127,  80,  60, 159, 168],
    [ 81, 163,  64, 143, 146, 157,  56, 245, 188, 182, 218,  33,  16, 255, 243, 210],
    [205,  12,  19, 236,  95, 151,  68,  23, 196, 167, 126,  61, 100,  93,  25, 115],
    [ 96, 129,  79, 220,  34,  42, 144, 136,  70, 238, 184,  20, 222,  94,  11, 219],
    [224,  50,  58,  10,  73,   6,  36,  92, 194, 211, 172,  98, 145, 149, 228, 121],
    [231, 200,  55, 109, 141, 213,  78, 169, 108,  86, 244, 234, 101, 122, 174,   8],
    [186, 120,  37,  46,  28, 166, 180, 198, 232, 221, 116,  31,  75, 189, 139, 138],
    [112,  62, 181, 102,  72,   3, 246,  14,  97,  53,  87, 185, 134, 193,  29, 158],
    [225, 248, 152,  17, 105, 217, 142, 148, 155,  30, 135, 233, 206,  85,  40, 223],
    [140, 161, 137,  13, 191, 230,  66, 104,  65, 153,  45,  15, 176,  84, 187,  22]
]) 

SUPPORTED_TARGETS = ['SBO', 'HW'] # SBO = SBox Output
                                  # HW  = Hamming Weight of the SBox Output


def compute_labels(plaintext, key, target='SBO'):
    
    """ 
    Generates the labels associated to each byte of the given plaintext/key 
    w.r.t. the given target.

    Parameters: 
        - plaintext (int numpy array):
            plaintext considered as list of int values, each one relative to 
            a single byte of the hex version.
        - key (int numpy array): 
            encryption key considered as list of int values, each one relative
            to a single byte of the hex version.
        - target (str):
            target of the attack (either 'SBO', for SBox Output, or 'HW', for 
            Hamming Weight of the SBox Output).

    Returns:
        int numpy array containing the 16 labels (one per byte of plaintext/key)
        relative to the given target.
        In case of 'SBO' target, the output is the result of the SBox-lookup.
        In case of 'HW' target, the output is the Hamming Weight of the result 
        of the SBox-lookup.
    """

    sbox_out = compute_sbox_out(plaintext, key)

    if target == TargetEnum.SBO:
        return sbox_out
    elif target == TargetEnum.HW:
        return hamming_weights(sbox_out)


def compute_sbox_out(plaintext, key):

    """ 
    Computes the result of the SBox-lookup performed with the given plaintext and key.

    Parameters:
        - plaintext (int numpy array):
            plaintext considered as list of int values, each one relative to 
            a single byte of the hex version.
        - key (int numpy array): 
            encryption key considered as list of int values, each one relative
            to a single byte of the hex version.
    
    Returns: 
        int numpy array containing the SBox-lookup outputs relative to the given 
        plaintext and key.
    """

    sbox_in = add_round_key(plaintext, key) # AES AddRoundKey
    sbox_out = sub_bytes(sbox_in) # AES SubBytes

    return sbox_out
        
        
def add_round_key(plaintext, key):

    """ 
    Implements AES AddRoundKey (plaintext XOR key).

    Parameters:
        - plaintext (int numpy array):
            plaintext considered as list of int values, each one relative to 
            a single byte of the hex version.
        - key (int numpy array): 
            encryption key considered as list of int values, each one relative
            to a single byte of the hex version.

    Returns: 
        int numpy array containing each byte of the XOR between the given plaintext
        and key. 
    """

    return plaintext ^ key
    
    
def sub_bytes(sbox_in):

    """ 
    Implements AES SubBytes (SBox-lookup).
    
    Input:
        - sbox_in (int numpy array): 
            list of each byte of the XOR between a plaintext and a key. 

    Returns: 
        int list containing the result of the SBox-lookup.
    """

    sbox_in_hex = int_to_hex(sbox_in) # Convert the SBox input to well-formatted hex (each byte independently)

    rows = [int(byte[0], 16) for byte in sbox_in_hex] # The first 4 bits (of each byte) are the row index 
    cols = [int(byte[1], 16) for byte in sbox_in_hex] # The remaining 4 bits (of each byte) are the col index 

    return SBOX_DEC[rows, cols]
        
        
def int_to_hex(int_values):

    """ 
    Converts int values into hex values, where the eventual 0 in front is explicit.

    Parameters:
        - int_values (int numpy array):
            int values to be converted in hex.

    Returns:
        srt list where each value is the hex conversion of the int value in input.
        The eventual 0 in fron is expicit. 
    """

    hex_values = []
    for val in int_values:
        tmp = hex(val).replace('0x', '') # Get rid of '0x' that is generated by the cast to hex
        if len(tmp) == 1:
            tmp = f'0{tmp}' # Add 0 in front of the conversion if its len is 1
        hex_values.append(tmp)

    return hex_values
    
    
def hamming_weights(int_values):

    """ 
    Computes the Hamming Weights of the given int values.
    The Hamming Weight of a int value X is the number of 1s in the binary-conversion 
    of X.

    Parameters:
        - int_values (int numpy array):
            int values whose Hamming Weights should be computed.

    Returns:
        int list containing the Hamming Weights of the given int values.  
    """

    bin_values = [np.binary_repr(val) for val in int_values] # Binary-conversion of the input

    return [bin_val.count('1') for bin_val in bin_values] # HW = Number of 1s in a bin number

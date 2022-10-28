###############################################################################
#                                                                             #
#       Configuration file containing all constants used in the project       #
#                                                                             #
###############################################################################


import numpy as np


# All devices
DEVICES = ['D1', 'D2', 'D3']

# All keys (generated with key_gen.py script)
KEYS = {
    'K0':  ['95', '30', 'fc', 'd9', 'd6', 'fd', '1d', '9b', '62', '03', '28', '01', 'b6', '5c', '1c', '28'],
    'K1':  ['2a', 'd6', '1d', '54', 'ff', '8f', '73', '5c', '37', 'e0', '6c', '7b', '2e', 'be', '57', '94'],
    'K2':  ['ae', '80', 'b0', '7a', 'ab', 'bf', '3b', '84', '2b', '5c', '13', '8b', '31', 'b0', '3d', 'd5'],
    'K3':  ['ff', 'ad', 'a0', '62', 'c1', 'fb', '0c', 'f7', 'b4', 'b4', 'e5', '66', '17', '7f', '53', 'c2'],
    'K4':  ['48', '8b', '09', 'ac', 'b4', 'e1', '6c', '74', 'ce', '6f', '29', '1a', '26', 'bb', '9d', '18'],
    'K5':  ['03', '41', '12', '3c', 'c4', '14', 'd3', '9d', 'ec', '13', 'f9', 'ab', 'b9', '75', '82', 'c6'],
    'K6':  ['b8', '95', '57', '9c', 'dd', 'a3', '42', '6b', '77', 'bf', '23', 'b9', '70', 'fe', '21', 'e4'],
    'K7':  ['07', '36', '2b', 'ea', '1d', '97', '8d', '8c', 'a2', '9a', 'f4', '82', 'fc', 'e7', '99', 'cd'],
    'K8':  ['7d', 'c6', '7e', '9e', 'f5', '4a', '07', '56', '2b', '2c', 'bd', '4c', '84', '53', '32', '47'],
    'K9':  ['ae', '8d', 'e4', '29', '71', 'b7', '91', 'cd', 'd8', '60', '05', '5b', 'bd', '38', 'e7', 'e2'],
    'K10': ['fe', 'c0', 'ca', '1d', 'f3', 'f9', 'da', 'a1', '7f', 'f0', '32', 'fa', '4d', 'fa', '54', '65']
}

# Trace constants
TRACE_LEN = 1183
MSK_TRACE_LEN = 8871

TRACE_NUM = 50000

# Number of classes per target
N_CLASSES = {
    'KEY': 256,
    'SBOX_IN': 256,
    'SBOX_OUT': 256
}

# Int-version of AES SBox
SBOX_DEC = np.array(
    [
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
    ]
)

# Int-version of Reverse AES SBox
INV_SBOX_DEC = np.array(
    [
        [ 82,   9, 106, 213,  48,  54, 165,  56, 191,  64, 163, 158, 129, 243, 215, 251],
        [124, 227,  57, 130, 155,  47, 255, 135,  52, 142,  67,  68, 196, 222, 233, 203],
        [ 84, 123, 148,  50, 166, 194,  35,  61, 238,  76, 149,  11,  66, 250, 195,  78],
        [  8,  46, 161, 102,  40, 217,  36, 178, 118,  91, 162,  73, 109, 139, 209,  37],
        [114, 248, 246, 100, 134, 104, 152,  22, 212, 164,  92, 204,  93, 101, 182, 146],
        [108, 112,  72,  80, 253, 237, 185, 218,  94,  21,  70,  87, 167, 141, 157, 132],
        [144, 216, 171,   0, 140, 188, 211,  10, 247, 228,  88,   5, 184, 179,  69,   6],
        [208,  44,  30, 143, 202,  63,  15,   2, 193, 175, 189,   3,   1,  19, 138, 107],
        [ 58, 145,  17,  65,  79, 103, 220, 234, 151, 242, 207, 206, 240, 180, 230, 115],
        [150, 172, 116,  34, 231, 173,  53, 133, 226, 249,  55, 232,  28, 117, 223, 110],
        [ 71, 241,  26, 113,  29,  41, 197, 137, 111, 183,  98,  14, 170,  24, 190,  27],
        [252,  86,  62,  75, 198, 210, 121,  32, 154, 219, 192, 254, 120, 205,  90, 244],
        [ 31, 221, 168,  51, 136,   7, 199,  49, 177,  18,  16,  89,  39, 128, 236,  95],
        [ 96,  81, 127, 169,  25, 181,  74,  13,  45, 229, 122, 159, 147, 201, 156, 239],
        [160, 224,  59,  77, 174,  42, 245, 176, 200, 235, 187,  60, 131,  83, 153,  97],
        [ 23,  43,   4, 126, 186, 119, 214,  38, 225, 105,  20,  99,  85,  33,  12, 125]
    ]
)

# Default paths
PC_TRACES_PATH = '/prj/side_channel/Pinata/PC/swAES'
PC_MULTIKEY_PATH = '/prj/side_channel/Pinata/PC/swAES/MultiKeySplits'
EM_TRACES_PATH = '/prj/side_channel/Pinata/EM/swAES'
MSK_PC_TRACES_PATH = '/prj/side_channel/Pinata/PC/swMaskedAES'
MSK_EM_TRACES_PATH = '/prj/side_channel/Pinata/EM/swMaskedAES'
RESULTS_PATH = '/prj/side_channel/Pinata/results'

# Train-Attack permutations for DKTA (format: (train devs list, attack dev))
PERMUTATIONS = {
    1: [
        (['D1'], 'D2'), 
        (['D1'], 'D3'), 
        (['D2'], 'D1'),
        (['D2'], 'D3'),
        (['D3'], 'D1'),
        (['D3'], 'D2')
       ],
    2: [
        (['D1', 'D2'], 'D3'),
        (['D1', 'D3'], 'D2'),
        (['D2', 'D3'], 'D1')
       ]
}


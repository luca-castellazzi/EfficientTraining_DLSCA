import trsfile
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, '../utils')
import constants


def main():
   
    _, dev, n_keys, tr_per_key = sys.argv

    dev = dev.upper()
    n_keys = int(n_keys) # Number of keys
    tr_per_key = int(tr_per_key) # Number of traces per key
    
    IN_FILE = f'{constants.PC_TRACES_PATH}/{dev}-MultiKey_500MHz + Resampled.trs'
    OUT_FILE = '/home/lcastellazzi/Inspector/aesMultiKey.bin'
    
    with trsfile.open(IN_FILE, 'r') as traces:
        for i in tqdm(range(0, tr_per_key*n_keys, tr_per_key), desc='Retrieving keys: '):
            k = traces[i].get_key()
            with open(OUT_FILE, 'ab') as f:
                f.write(bytes(k))


if __name__ == '__main__':
    main()

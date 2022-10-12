import trsfile
from tqdm import tqdm

import sys
sys.path.insert(0, '../utils')
import constants

N_KEYS = 100


def main():
   
    _, dev, tr_per_key = sys.argv

    dev = dev.upper()
    tr_per_key = int(tr_per_key) # Number of traces per key
    
    IN_FILE = f'{constants.PC_TRACES_PATH}/{dev}-MultiKey_500MHz + Resampled.trs'
    OUT_FILE = '/home/lcastellazzi/Inspector/aesMultiKey.bin'
    
    with trsfile.open(IN_FILE, 'r') as traces:
    
        for i in tqdm(range(0, tr_per_key*N_KEYS, tr_per_key), desc='Retrieving keys: '):
           
            k = traces[i].get_key()
            
            if i == 0:
                mode = 'wb' # For the first key, create a new file
            else:
                mode = 'ab' # For all other keys, use the created file in append mode
                    
            with open(OUT_FILE, mode) as f:
                f.write(bytes(k))


if __name__ == '__main__':
    main()

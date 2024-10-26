import trsfile
from tqdm import tqdm
import os

import sys
sys.path.insert(0, '../utils')
import constants

N_TR = 50000
N_SAMPLES = 7700
ROOT = f'{constants.MSK_PC_TRACES_PATH}/second_order'

def main():
    
    for fname in tqdm(os.listdir(ROOT)):

        if '+ Resampled.trs' in fname:
            IN_FILE = f'{ROOT}/{fname}'
            OUT_FILE = f'{ROOT}/Trim/{fname.split(" +")[0]}.trs'
            
            with trsfile.open(IN_FILE, 'r') as trace_set:
                new_traces = trace_set[:N_TR]

                for tr in new_traces:
                    tr.samples = tr.samples[:N_SAMPLES]

                new_headers = trace_set.get_headers().copy() # A copy is needed
                new_headers[trsfile.Header.NUMBER_TRACES] = 0 
                new_headers[trsfile.Header.NUMBER_SAMPLES] = N_SAMPLES

                with trsfile.trs_open(OUT_FILE, 'w', headers=new_headers) as new_trace_set:
                    new_trace_set.extend(new_traces)


if __name__ == '__main__':
    main()

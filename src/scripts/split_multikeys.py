import trsfile
from tqdm import tqdm

import sys
sys.path.insert(0, '../utils')
import constants

N_KEYS = 100
TRACES_PER_KEY = 50000


def main():

    _, dev = sys.argv

    dev = dev.upper()
    IN_FILE = f'{constants.PC_TRACES_PATH}/{dev}-MultiKey_500MHz + Resampled.trs'

    with trsfile.open(IN_FILE, 'r') as trace_set:
        
        for k, start in enumerate(tqdm(range(0, TRACES_PER_KEY*N_KEYS, TRACES_PER_KEY), 
                desc='Splitting MultiKey trace: ')):
            
            OUT_FILE = f'{constants.PC_MULTIKEY_PATH}/{dev}-MK{k}.trs'

            stop = start + TRACES_PER_KEY
            
            new_traces = trace_set[start:stop]
                
            # The header values of the original trace-set must be kept also in
            # the new trace-set.
            # The idea is to take the original values and changing the number of
            # traces to 0 (because at the beginning, the new file must be empty)
            new_headers = trace_set.get_headers().copy() # A copy is needed
            new_headers[trsfile.Header.NUMBER_TRACES] = 0 

            with trsfile.trs_open(OUT_FILE, 'w', headers=new_headers) as new_trace_set:
                new_trace_set.extend(new_traces)




if __name__ == '__main__':
    main()

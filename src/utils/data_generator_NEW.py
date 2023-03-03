# Basics
from tensorflow.keras.utils import Sequence

# Custom
from batch_loader import BatchLoader


class DataGenerator(Sequence):
    
    def __init__(self, tr_files, tr_start, tr_tot, tr_len, batch_size, target, byte_idx, scaler, 
                 start_sample=None, stop_sample=None, to_fit=True, 
                 shuffle_on_epoch_end=True, cnn=False):

        self.tr_files = tr_files # Files from where to read
        self.tr_start = tr_start # Index of the first trace to consider
        self.tr_tot = tr_tot # Total amount of traces to consider (contiguous starting from tr_start)
        self.batch_size = batch_size # Original batch size
        self.target = target # Target of the attack
        self.byte_idx = byte_idx # Byte to attack
        self.scaler = scaler

        self.batch_loader = BatchLoader(tr_files, tr_start, tr_tot, batch_size)

        self.start_sample = start_sample if start_sample is not None else 0
        self.stop_sample = stop_sample if start_sample is not None else tr_len
        self.to_fit = to_fit
        self.shuffle_on_epoch_end = shuffle_on_epoch_end
        self.cnn = cnn

        self.on_epoch_end()
        
        
    def __len__(self):
        # Generates the number of batches per epoch
        return self.batch_loader.actual_batch_num
        
        
    def __getitem__(self, index):
        
        # Generates a batch of data
        x, y = self.batch_loader.load(
            batch_range_idx=index, 
            start_sample=self.start_sample,
            stop_sample=self.stop_sample,
            target=self.target,
            byte_idx=self.byte_idx
        )

        x = self.scaler.transform(x)
        
        if self.cnn:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        
        if self.to_fit:
            return x, y
        else:
            return x
    
    
    def on_epoch_end(self):
        # Eventually shuffles the order of the traces
        if self.shuffle_on_epoch_end:
            self.batch_loader.shuffle_batch_ranges()
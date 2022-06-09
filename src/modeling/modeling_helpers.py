import sys
from tensorflow.keras.callbacks import EarlyStopping

# Custom modules from utils/
sys.path.insert(0, '../utils')
from single_byte_evaluator import SingleByteEvaluator


def create_callbacks(es=True):

    """
    Generates the specified callbacks for a DL model.

    Parameters:
        - es (bool, default: True):
            Whether or not creating an EarlyStopping callback.

    Returns:
        Keras Callback list containing all the specified callbacks.
    """

    callbacks = []

    if es:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=5))
    else:
        pass # In future also other callbacks (e.g. the one used to save the
             # model in .h5)

    return callbacks


def guessing_entropy(model, batch_size, x_train_tot, y_train_tot, x_test, test_plaintexts, true_key_byte, byte_idx, n_exp):

    num_train_traces = int(len(x_train_tot) / n_exp)
    num_test_traces = int(len(x_test) / n_exp)
    
    ranks = []
    for i in range(n_exp):
        start_train = i * num_train_traces
        stop_train = start_train + num_train_traces
        
        start_test = i * num_test_traces
        stop_test = start_test + num_test_traces

        curr_x_train = x_train_tot[start_train:stop_train]
        curr_y_train = y_train_tot[start_train:stop_train]
        curr_x_test = x_test[start_test:stop_test]
        curr_plaintexts = test_plaintexts[start_test:stop_test]
        
        model.fit(curr_x_train,
                  curr_y_train,
                  epochs=100,
                  batch_size=batch_size,
                  verbose=1)

        curr_preds = model.predict(curr_x_test)

        curr_evaluator = SingleByteEvaluator(test_plaintexts=curr_plaintexts,
                                             byte_idx=byte_idx,
                                             label_preds=curr_preds)

        curr_ranks = []
        for j in tqdm(range(num_test_traces)):
            n_traces = j + 1
            curr_ranks.append(curr_evaluator.rank(true_key_byte, n_traces))

            curr_ranks = np.array(curr_ranks)
            ranks.append(curr_ranks)

    ranks = np.array(ranks)
    guessing_entropy = np.round(np.mean(ranks, axis=0)) # .5 approximated to the next int

    return guessing_entropy

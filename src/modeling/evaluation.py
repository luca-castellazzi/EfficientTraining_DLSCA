import random
import numpy as np
from tqdm import tqdm
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 


def compute_key_probs(probs, key_bytes):
    key_probs = []

    # For each element in the association between key-bytes and sbox-probs...
    for kbs, ps in zip(key_bytes, probs):

        # ...associate each sbox-prob to its relative key-byte...
        curr_key_probs = list(zip(kbs, ps))

        # ...sort the sbox-probs w.r.t. their relative key-byte...
        curr_key_probs.sort(key=lambda x: x[0])

        # ...consider only the sorted predicions to "transform" sbox-probs
        # into key-byte-probs
        curr_key_probs = list(zip(*curr_key_probs))[1]

        key_probs.append(curr_key_probs)

    return np.array(key_probs)


def compute_true_kb_ranks(probs, key_bytes, true_kb):
    key_probs = compute_key_probs(probs, key_bytes)
    log_probs = np.log10(key_probs + 1e-22)
    cum_tot_probs = np.cumsum(log_probs, axis=0)

    indexed_cum_tot_probs = [list(zip(range(256), tot_probs))
                             for tot_probs in cum_tot_probs]
    sorted_cum_tot_probs = [sorted(el, key=lambda x: x[1], reverse=True)
                            for el in indexed_cum_tot_probs]
    sorted_kbs = [[el[0] for el in tot_probs]
                  for tot_probs in sorted_cum_tot_probs]

    true_kb_ranks = [skb.index(true_kb)
                     for skb in sorted_kbs]
    true_kb_ranks = np.array(true_kb_ranks)

    return true_kb_ranks


# ------------------------------- #
# Guessing Entropy implementation #
# ------------------------------- #
def guessing_entropy(model, n_exp, test_data, n_traces):

    # Unpack test data
    x, kbs, true_kb = test_data
    
    x_kbs = list(zip(x, kbs))
    
    ranks_per_exp = []
    for _ in range(10):
        random.shuffle(x_kbs)
        shuffled_x, shuffled_kbs = list(zip(*x_kbs))
        shuffled_x = np.array(shuffled_x)

        probs = model.predict(shuffled_x)

        true_kb_ranks = compute_true_kb_ranks(probs[:n_traces], 
                                              shuffled_kbs[:n_traces], 
                                              true_kb)
        ranks_per_exp.append(true_kb_ranks)

    
    ge = np.mean(ranks_per_exp, axis=0)

    return ge


# -------------------------------------------------------------------------- #
# Custom Metric implementation:                                              #      
# Min number of traces to have GE=0 consistently (for at least N consecutiv  #
# steps)                                                                     #
# -------------------------------------------------------------------------- #
def ge_score(ge, n_consecutive_zeros, zero_threshold=0.3):
    
    iszero = (ge <= zero_threshold)
    score = len(ge) - 1

    if len(iszero) != 0:
        # Generate a vector with 1s where GE is 0 (all other elements are 0)
        tmp = np.concatenate(([0], iszero, [0]))

        # Generate a vector with 1s only at start_idx, end_idx of a 0-sequence in GE
        start_end_zeros = np.abs(np.diff(tmp))

        zero_slices = np.where(start_end_zeros==1)[0].reshape(-1, 2)
        for el in zero_slices:
            if (el[1] - el[0]) >= n_consecutive_zeros:
                score = el[0]
    
    return score


# -------------------------------- #
# Classic KFold XVal with Accuracy #
# -------------------------------- #
def xval_acc(kf, networks, train_data, hp_space):

    # Unpack the train data
    x, y, _, _, epochs, _ = train_data

    # Set the callbacks to use for the train (they are common to all models)
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', 
                                   patience=15))
    callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2,
                                       patience=8,
                                       min_lr=1e-7))

    # Iteratively evaluate all models
    results = []
    for i, net in enumerate(networks):
        
        print()
        print(f'HP Config {i+1}/{len(networks)}: ')

        random_hp = {k: random.choice(hp_space[k]) for k in hp_space}
        net.set_hp(random_hp)

        acc_per_exp = []
        for t_idx, v_idx in tqdm(kf.split(x)):
            net.build_model()
            model = net.get_model()

            x_t = x[t_idx]
            y_t = y[t_idx]
            
            x_v = x[v_idx]
            y_v = y[v_idx]

            model.fit(x_t,
                      y_t,
                      validation_data=(x_v, y_v),
                      epochs=epochs,
                      callbacks=callbacks,
                      verbose=0)

            _, acc = model.evaluate(x_v, y_v, verbose=0)

            acc_per_exp.append(acc)

            net.reset()
            keras_backend.clear_session()

        acc_per_exp = np.array(acc_per_exp)
        avg_acc = np.mean(acc_per_exp)

        print(f'Avg acc: {avg_acc}')
        
        results.append((i, avg_acc))
    
    results.sort(key=lambda x: x[1], reverse=True)
    print()
    print(f'KFold Top5 Results: {results[:5]}')

    best_net = networks[results[0][0]]

    return best_net.get_hp()


# ---------------------------------------------------------------- #
# Custom KFold XVal with GE as metric (computed for each val fold) #
# ---------------------------------------------------------------- #
def xval_ge(kf, networks, train_data, hp_space):

    # Unpack the train data
    x, y, kbs, true_kb, epochs, ge_tr = train_data

    # Set the callbacks to use for the train (they are common to all models)
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', 
                                   patience=15))
    callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2,
                                       patience=8,
                                       min_lr=1e-7))

    # Iteratively evaluate all models
    results = []
    for i, net in enumerate(networks):

        print()
        print(f'HP Config {i+1}/{len(networks)}: ')

        random_hp = {k: random.choice(hp_space[k]) for k in hp_space}
        net.set_hp(random_hp)

        score_per_exp = []
        for t_idx, v_idx in tqdm(kf.split(x)):
            net.build_model()
            model = net.get_model()

            x_t = x[t_idx]
            y_t = y[t_idx]
            
            x_v = x[v_idx]
            y_v = y[v_idx]
            kbs_v = kbs[v_idx]

            model.fit(x_t,
                      y_t,
                      validation_data=(x_v, y_v),
                      epochs=epochs,
                      callbacks=callbacks,
                      verbose=0)
            
            ge = guessing_entropy(model, 
                                  n_exp=10, 
                                  test_data=(x_v, kbs_v, true_kb), 
                                  n_traces=ge_tr)
            score = ge_score(ge, 10)

            score_per_exp.append(score)

            net.reset()
            keras_backend.clear_session()

        score_per_exp = np.array(score_per_exp)
        avg_score = np.mean(score_per_exp)

        print(f'Avg score: {avg_score}')
        
        results.append((i, avg_score))
    
    results.sort(key=lambda x: x[1])
    print()
    print(f'KFold Top5 Results: {results[:5]}')

    best_net = networks[results[0][0]]
    
    return best_net.get_hp()


# ----------------------------------------------------------------- #
# Custom KFold XVal with GE as metric (computed as average over the #
# k iterations)                                                     #
# ----------------------------------------------------------------- #
def xval_ge_iter(kf, networks, train_data, hp_space):
    
    # Unpack the train data
    x, y, kbs, true_kb, epochs, ge_tr = train_data

    # Set the callbacks to use for the train (they are common to all models)
    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', 
                                   patience=15))
    callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2,
                                       patience=8,
                                       min_lr=1e-7))

    # Iteratively evaluate all models
    results = []
    for i, net in enumerate(networks):

        print()
        print(f'HP Config {i+1}/{len(networks)}: ')

        random_hp = {k: random.choice(hp_space[k]) for k in hp_space}
        net.set_hp(random_hp)

        ranks_per_exp = []
        for t_idx, v_idx in tqdm(kf.split(x)):
            net.build_model()
            model = net.get_model()

            x_t = x[t_idx]
            y_t = y[t_idx]
            
            x_v = x[v_idx]
            y_v = y[v_idx]
            kbs_v = kbs[v_idx]

            model.fit(x_t,
                      y_t,
                      validation_data=(x_v, y_v),
                      epochs=epochs,
                      callbacks=callbacks,
                      verbose=0)
            
            probs = model.predict(x_v)
            
            random_idx = random.sample(range(len(probs)), ge_tr)
            true_kb_ranks = compute_true_kb_ranks(probs[random_idx], 
                                                  kbs_v[random_idx],
                                                  true_kb)
            
            ranks_per_exp.append(true_kb_ranks)

            net.reset()
            keras_backend.clear_session()
        
        ranks_per_exp = np.array(ranks_per_exp)
        ge = np.mean(ranks_per_exp, axis=0) 
            
        score = ge_score(ge, 10)
        print(f'Score: {score}')
        
        results.append((i, score))
    
    results.sort(key=lambda x: x[1])
    print()
    print(f'KFold Top5 Results: {results[:5]}')

    best_net = networks[results[0][0]]

    return best_net.get_hp()   

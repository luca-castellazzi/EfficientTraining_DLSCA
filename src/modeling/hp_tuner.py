# Basics
from tqdm import tqdm
from tensorflow.keras.backend import clear_session
import random

# Custom
from network import Network
from genetic_tuner import GeneticTuner

class HPTuner():
    
    def __init__(self, model_type, hp_space, n_models, n_epochs):
        
        self.model_type = model_type
        self.hp_space = hp_space
        self.n_models = n_models
        self.n_epochs = n_epochs
                                
    
    def random_search(self, x_train, y_train, x_val, y_val):
    
        res = []
        for m in tqdm(range(self.n_models), desc='Random Search: '):
            random_hp = {k: random.choice(self.hp_space[k]) for k in self.hp_space.keys()}
            net = Network(self.model_type, random_hp)
            net.build_model()
            model = net.model
            history = model.fit(
                x_train, 
                y_train, 
                validation_data=(x_val, y_val),
                epochs=self.n_epochs,
                batch_size=net.hp['batch_size'],
                callbacks=net.callbacks,
                verbose=0
            ).history
            
            val_loss, _ = model.evaluate(x_val, y_val, verbose=0)                    
            res.append((val_loss, random_hp, history))
            
            clear_session()
            
        res.sort(key=lambda x: x[0])
        
        self.best_val_loss, self.best_hp, self.best_history = res[0] # Take track of the best results
        
        print(f'Best result: {self.best_val_loss}')
        
        return self.best_hp
        
        
    
    def random_search_xval(self, n_folds):
        return
    
    
    def genetic_algorithm(self, n_gen, selection_perc, second_chance_prob, mutation_prob,
                          x_train, y_train, x_val, y_val):
        
        gt = GeneticTuner(
            model_type='MLP', 
            hp_space=self.hp_space, 
            pop_size=self.n_models, 
            selection_perc=selection_perc, 
            second_chance_prob=second_chance_prob, 
            mutation_prob=mutation_prob
        )
        
        pop = gt.populate()
        
        for gen in range(n_gen):
            print(f'=====  Gen {gen+1}/{n_gen}  =====')
            evaluation = gt.evaluate(
                pop=pop, 
                x_train=x_train, 
                y_train=y_train, 
                x_val=x_val, 
                y_val=y_val, 
                n_epochs=self.n_epochs
            )
            
            print(f'Top 5: {[val_loss for val_loss, _, _ in evaluation[:5]]}')
            
            parents = gt.select(evaluation)
            
            if gen != n_gen-1:
                pop = gt.evolve(parents)
        
        self.best_val_loss, self.best_hp, self.best_history = evaluation[0]
    
        return self.best_hp

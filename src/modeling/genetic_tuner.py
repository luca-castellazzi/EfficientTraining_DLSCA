# Basics
import random
from tensorflow.keras.backend import clear_session
from tqdm import tqdm

# Custom
from network import Network


class GeneticTuner():

    def __init__(self, model_type, hp_space, pop_size, selection_perc, second_chance_prob, mutation_prob, metric):
    
        self.model_type = model_type
        self.hp_space = hp_space
        self.pop_size = pop_size
        self.selection_perc = selection_perc
        self.second_chance_prob = second_chance_prob
        self.mutation_prob = mutation_prob
        self.metric=metric
        
    def populate(self):
    
        pop = [{k: random.choice(self.hp_space[k]) for k in self.hp_space.keys()}
               for _ in range(self.pop_size)]
            
        return pop
        
        
    def evaluate(self, pop, x_train, y_train, x_val, y_val, n_epochs):
        
        res = []
        for hp_config in tqdm(pop, desc='Training the population: '):
            
            net = Network(self.model_type, hp_config)
            net.build_model()
            model = net.model
            history = model.fit(
                x_train, 
                y_train, 
                validation_data=(x_val, y_val),
                epochs=n_epochs,
                batch_size=net.hp['batch_size'],
                callbacks=net.callbacks,
                verbose=0
            ).history
            
            if self.metric == 'rank':
                pass
                #score = net.rank_key_byte(x_val) ##############################################################################
            else:
                val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
                if self.metric == 'loss':
                    score = val_loss
                else:
                    score = val_acc
                    
            res.append((score, hp_config, history))
            
            clear_session()
        
        if self.metric == 'acc':
            reverse = True
        else: 
            reverse = False
        res.sort(key=lambda x: x[0], reverse=reverse)
        
        return res
        
        
    def select(self, evaluation):
        
        # Sort the individuals w.r.t. their val accuracy (the higher the better)
        sorted_pop = [hp_config for _, hp_config, _ in evaluation]
        
        # Select the best-perfoming individuals as parents for the next population
        num_selected = int(self.selection_perc * self.pop_size)
        parents = sorted_pop[:num_selected]
        
        # Select also some bad-perfoming individuals to add diversity in the 
        # next population
        for hp_config in sorted_pop[num_selected:]:
            if random.random() < self.second_chance_prob:
                parents.append(hp_config)

        return parents
        
        
    def _produce_offsprings(self, parent_a, parent_b):
        
        # An offspring is an hp configuration where the values derive from the parents 
        # in a random way
        offspring = {k: random.choice([parent_a[k], parent_b[k]])
                     for k in self.hp_space.keys()}
                        
        return offspring
        
        
    def _mutate_offspring(self, offspring):
        
        to_mutate = random.choice(list(self.hp_space.keys()))
        offspring[to_mutate] = random.choice(self.hp_space[to_mutate])
        
        return offspring
        
    
    def evolve(self, parents):
        
        n_offsprings = self.pop_size - len(parents)
        
        offsprings = []
        for _ in range(n_offsprings):
            parent_a, parent_b = random.sample(parents, k=2)
            offspring = self._produce_offsprings(parent_a, parent_b)
            
            if random.random() < self.mutation_prob:
                offspring = self._mutate_offspring(offspring)
            
            offsprings.append(offspring)
        
        new_pop = parents + offsprings
        
        return new_pop    

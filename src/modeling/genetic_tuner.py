###################################################################################
#                                                                                 # 
#  This Genetic Algorithm hyperparameter tuning method for Neural Network is      #
#  based onthe implementation by Matt Harvey (https://github.com/harvitronix)     #
#  available at https://github.com/harvitronix/neural-network-genetic-algorithm), #
#  licensed under MIT License:                                                    #      
#                                                                                 #
#  The MIT License                                                                #
#                                                                                 #
#  Copyright (c) 2017 Matt Harvey                                                 #
#                                                                                 #
#  Permission is hereby granted, free of charge, to any person obtaining a copy   # 
#  of this software and associated documentation files (the "Software"), to deal  #
#  in the Software without restriction, including without limitation the rights   #
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      #
#  copies of the Software, and to permit persons to whom the Software is          #
#  furnished to do so, subject to the following conditions:                       #
#                                                                                 #
#  The above copyright notice and this permission notice shall be included in     #
#  all copies or substantial portions of the Software.                            #
#                                                                                 #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     #
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       #
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    # 
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         #  
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  #
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN      #
#  THE SOFTWARE.                                                                  #
#                                                                                 #
###################################################################################

# Basics
import random
from tqdm import tqdm
from tensorflow.keras.backend import clear_session

# Custom
from hp_tuner import HPTuner


class GeneticTuner(HPTuner):

    """
    Subclass of HPTuner (implements the "interface") which performs hyperparameter 
    tuning through Genetic Algorithm.
    
    Additional attributes:
        - pop_size (int):
            Size of each population of hyperparameters.
        - n_gen (int):
            Number of generations.
        - selection_perc (float):
            Percentage of best-performing hyperparameters to keep by default 
            for the next population.
        - second_chance_prob (float):
            Probability of keeping a bad-performing hyperparameters.
        - mutation_prob (float):
            Probability of mutating an offspring.

    Methods:
        - _populate:
            Generates a populations of hyperparameters with random values.
        - _evaluate:
            Evaluates the performance of a population of hyperparameters.
        - _select:
            Selects the parents for the next population.
        - _produce_offspring:
            Generates an offspring.
        - _mutate_offspring:
            Mutates an offspring.
        - _evolve:
            Generates the new population of hyperparameters.

    Overwritten methods:
        - tune:
            Implements a Genetic Algorithm.
    """

    def __init__(self, model_fn, trace_len, n_classes, hp_space, n_epochs, 
        pop_size, n_gen, selection_perc, second_chance_prob, mutation_prob):

        """
        Class constructor: takes as input all class attributes and generates a 
        GeneticTuner object.
        """

        super().__init__(model_fn, trace_len, n_classes, hp_space, n_epochs)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.selection_perc = selection_perc
        self.second_chance_prob = second_chance_prob
        self.mutation_prob = mutation_prob


    def tune(self, train_data, val_data, callbacks, batch_size=None, metrics=['accuracy'], use_gen=False):

        """
        Performs hyperparameter tuning with a Genetic Algorithm: populations of 
        hyperparameters are generated, tested and updated according to the 
        "natural selection" principle (the best-performing hyperparameters are 
        kept to generate the next population).
        
        Only the hyperparameters that show the best performance after a fixed 
        number of generations are considered as result.
        
        The performance of the hyperparameters is given by the validation loss 
        (the lower the better).
        
        Parameters:
            - train_data (tuple or DataGenerator):
                Train and validation data.
                If tuple, should be in (samples, labels) format.
            - val_data (tuple or DataGenerator):
                If tuple, should be in (samples, labels) format.
            - callbacks (keras.callbacks list):
                List of callbacks to use during model training.
            - batch_size (int, default: None):
                Batch size to use during model training.
                If None, it is considered as hyperparameter (and must be inclued
                in hyperparameter space).
            - metrics (list, default: ['accuracy']):
                List of metrics to use during model training.
            - use_gen (bool, default: False):
                Whether or not the provided train and val data are DataGenerators. 

        Returns:
            - best_hp (dict):
                Best-performing hyperparameters.
            - best_history():
                Training history for the best-performing hyperparameters.  
        """

        pop = self._populate()
    
        for gen in tqdm(range(self.n_gen), desc='Genetic Algorithm: '):
            
            evaluation = self._evaluate(
                pop=pop, 
                train_data=train_data, 
                val_data=val_data, 
                callbacks=callbacks,
                batch_size=batch_size,
                metrics=metrics,
                use_gen=use_gen
            )
            parents = self._select(evaluation)
            
            if gen != self.n_gen-1:
                pop = self._evolve(parents)
        
        _, best_hp, best_history = evaluation[0] # evaluation is already sorted

        return best_hp, best_history


    def _populate(self):
    
        """
        Generates a population of hyperparameters with random values.
        The values are randomly chosen for each hyperparameter from the 
        hyperparameter space.
        
        Returns:
            - pop (list of dict):
                Population of hyperparameters.
        """
    
        pop = [{k: random.choice(self.hp_space[k]) for k in self.hp_space.keys()}
               for _ in range(self.pop_size)]
            
        return pop

    
    def _evaluate(self, pop, train_data, val_data, callbacks, batch_size, metrics, use_gen):
    
        """
        Evaluates the given population of hyperparameters training and validating 
        multiple models.
        
        Parameters:
            - pop (list of dict):
                Population of hyperparameters to evaluate.
            - train_data (tuple or DataGenerator):
                Train and validation data.
                If tuple, should be in (samples, labels) format.
            - val_data (tuple or DataGenerator):
                If tuple, should be in (samples, labels) format.
            - callbacks (keras.callbacks list):
                List of callbacks to use during model training.
            - batch_size (int or None):
                Batch size to use during model training.
            - metrics (list):
                List of metrics to use during model training.
            - use_gen (bool):
                Whether or not the provided train and val data are DataGenerators. 
        
        Returns:
            - res (list of tuple):
                Result of the evaluation, containing, for each individual in
                the population, the validation score (validation loss), the 
                individual (hyperparameters) and the training history.
                The tuples are ordered from the best score to the worst score
                (lowest loss to highest loss).
        """
        
        res = []

        for hp_config in pop:
            
            clear_session() # Start a new keras session every new training
            
            model = self.model_fn(
                hp=hp_config, 
                input_len=self.trace_len, 
                n_classes=self.n_classes,
                metrics=metrics
            )

            if use_gen:
                # train_data and val_data are DataGenerators
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=self.n_epochs,
                    callbacks=callbacks,
                    verbose=0
                ).history
            else:
                if batch_size is None:
                    bs = hp_config['batch_size']
                else:
                    bs = batch_size
                history = model.fit(
                    train_data[0], # x_train
                    train_data[1], # y_train
                    validation_data=(val_data[0], val_data[1]), # (x_val, y_val)
                    epochs=self.n_epochs,
                    batch_size=bs,
                    callbacks=callbacks,
                    verbose=0
                ).history
 
            val_loss = history['val_loss'][-1]      
            res.append((val_loss, hp_config, history))
            
        res.sort(key=lambda x: x[0])
        
        return res


    def _select(self, evaluation):
    
        """
        Selects the parents for the next population.
        
        A subset of individuals is selected from the current population w.r.t.
        the result of the evaluation step.
        Also some bad-performing individuals are selected, in order to add 
        diversity in the new population.
        
        The cardinality of the best-performing subset is specified in the 
        constructor as "selection_perc", while the chance of being selected if 
        the performance is not good is specified by "second_chance_prob".
        
        Parameters:
            - evaluation (list of tuple):
                Result of a population-evaluation step.
        
        Returns:
            - parents (list of dict):
                Selected hyperparameters.
        """
        
        # Sort the individuals w.r.t. their val accuracy (the higher the better)
        sorted_pop = [hp_config for _, hp_config, _ in evaluation]
        
        # Select the best-performing individuals as parents for the next population
        num_selected = int(self.selection_perc * self.pop_size)
        parents = sorted_pop[:num_selected]
        
        # Select also some bad-performing individuals to add diversity in the 
        # next population
        for hp_config in sorted_pop[num_selected:]:
            if random.random() < self.second_chance_prob:
                parents.append(hp_config)

        return parents


    def _produce_offspring(self, parent_a, parent_b):
    
        """
        Generates an offspring considering two parents.
        An offspring is a hyperparameter configuration where the values derive
        from the parents in a random way.
        
        Parameters:
            - parent_a, parent_b (dict):
                Hyperparameters to randomly mix in order to generate the offspring.
                
        Returns:
            - offspring (dict):
                Generated hyperparameters.
        """
        
        offspring = {k: random.choice([parent_a[k], parent_b[k]])
                     for k in self.hp_space.keys()}
                        
        return offspring
        
        
    def _mutate_offspring(self, offspring):
    
        """
        Mutates an offspring.
        A mutation consists in the substitution of a hyperparameter value with
        another chosen from the hyperparameter space (both choices are random).
        
        Parameters:
            - offspring (dict):
                Offspring to be mutated.
                
        Returns:
            - offspring (dict):
                The mutated version of the starting offspring.
        """
        
        to_mutate = random.choice(list(self.hp_space.keys()))
        offspring[to_mutate] = random.choice(self.hp_space[to_mutate])
        
        return offspring
        
    
    def _evolve(self, parents):
    
        """
        Generates a new population considering all the parents, their offspring 
        (eventually mutated) and some bad-performing individuals from the 
        the previous population.
        
        The mutation probability is specified in the constructor as 
        "mutation_prob".
        
        Parameters:
            - parents (dict list):
                Selected individuals from the previous population.
                
        Returns:
            - new_pop (dict list):
                New population of hyperparameters.
        """
        
        n_offsprings = self.pop_size - len(parents)
        
        offsprings = []
        for _ in range(n_offsprings):
            parent_a, parent_b = random.sample(parents, k=2) # Always different
            offspring = self._produce_offspring(parent_a, parent_b)
            
            if random.random() < self.mutation_prob:
                offspring = self._mutate_offspring(offspring)
            
            offsprings.append(offspring)
        
        new_pop = parents + offsprings
        
        return new_pop
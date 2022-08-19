##################################################################################
#                                                                                # 
#  This Genetic Algorithm for Neural Network hyperparameter tuning is based on   #
#  the implementation by Matt Harvey (https://github.com/harvitronix) available  # 
#  at https://github.com/harvitronix/neural-network-genetic-algorithm),          #
#  licensed under MIT License.                                                   #
#                                                                                #
##################################################################################

# Basics
import random
from tensorflow.keras.backend import clear_session
from tqdm import tqdm

# Custom
from network import Network


class GeneticTuner():

    """
    Genetic Algorithm for hyperparameter tuning.
    
    Populations of hyperparameters are generated, tested and updated
    according to the "natural selection" principle: the best-performing
    hyperparameters are kept to generate the next population.
    
    Attributes:
        - model_type (str):
           Type of Neural Network to implement.
        - hp_space (dict):
            Whole hyperparameter space.
        - pop_size (int):
            Size of each population of hyperparameters.
        - selection_perc (float):
            Percentage of best-performing hyperparameters to keep by default 
            for the next population.
        - second_chance_prob (float):
            Probability of keeping a bad-performing hyperparameter set.
        - mutation_prob (float):
            Probability of mutating an offspring.
            
    Methods:
        - populate:
            Generates a populations of hyperparameters with random values.
        - evaluate:
            Evaluates the performance of a population of hyperparameters.
        - select:
            Selects the parents for the next population.
        - _produce_offspring:
            Generates an offspring.
        - _mutate_offspring:
            Mutates an offspring.
        - evolve:
            Generates the new population of hyperparameters.
    """


    def __init__(self, model_type, hp_space, pop_size, selection_perc, 
                 second_chance_prob, mutation_prob):
    
        """
        Class constructor: takes as input all class attributes and generates a
        GeneticTuner object.
        """
    
        self.model_type = model_type
        self.hp_space = hp_space
        self.pop_size = pop_size
        self.selection_perc = selection_perc
        self.second_chance_prob = second_chance_prob
        self.mutation_prob = mutation_prob
        
        
    def populate(self):
    
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
        
        
    def evaluate(self, pop, x_train, y_train, x_val, y_val, n_epochs):
    
        """
        Evaluates the given population of hyperparameters.
        A Neural Network of the type specified in the constructor is built, 
        trained and evaluated w.r.t. the specified train and validation data.
        
        Parameters:
            - pop (list of dict):
                Population of hyperparameters to evaluate.
            - x_train, x_val (np.ndarray):
                Train and validation samples.
            - y_train, y_val (np.ndarray):
                Train and validation labels.
            - n_epochs (int):
                Number of epochs to use during the training.
        
        Returns:
            - res (list of tuple):
                Result of the evaluation, containing, for each individual in
                the population, the validation score (validation loss), the 
                individual (hyperparameters) and the training history.
                The tuples are ordered from the best score to the worst score
                (lowest to highest because loss is considered rank).
        """
        
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
            
            val_loss, _ = model.evaluate(x_val, y_val, verbose=0)        
            res.append((val_loss, hp_config, history))
            
            clear_session()
            
        res.sort(key=lambda x: x[0])
        
        return res
        
        
    def select(self, evaluation):
    
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
        
    
    def evolve(self, parents):
    
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

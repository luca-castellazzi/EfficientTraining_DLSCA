################################################################################                                      2 #                                                                              #
#                                                                              #
# This code is based on the Genetic Algorithm for Hyperparameter Tuning        #
# implementation from harvitronix.                                             #
#                                                                              #
# The reference project is                                                     #
# https://github.com/harvitronix/neural-network-genetic-algorithm,             #
# which is licensed under MIT License.                                         #
#                                                                              #
################################################################################


import random

from network import Network


class GeneticTuner():
    
    """
    Class dedicated to the implementation of a Genetic Algorithm for Neural
    Network Hyperparameter Tuning.

    Attributes:
        - _model_type:
            type of model implemented by each individual of the population.
        - _pop_size:
            size of the population.
        - _hp_choices:
            hyperparameter space, containing all the possible values for each
            hyperparameter.
        - _selection_perc:
            percentage of individuals to be considered in order to generate the 
            offsprings for the next population.
        - _mutation_chance:
            probability of mutation during the generation of an offspring.

    Methods:
        - populate:
            generation of the starting population.
        - evaluate:
            evaluation of the population.
        - select:
            selection of the best individuals of the population (the ones with 
            best performance).
        - _produce_offspring:
            generation of an offspring.
        - _mutate_offspring:
            mutation of an offspring.
        - evolve:
            evolution of the population considering the best individuals and 
            their offsprings.
    """


    def __init__(self, hp_choices, model_type='MLP', pop_size=10, selection_perc=0.5, mutation_chance=0.2):
        
        """
        Class constructor: gidven the hyperparameter space, a model type, a
        population size, a selection percentage and a mutation probability, a 
        new GeneticTuner is initialized.

        Parameters:
            - hp_choices (dict):
                hyperparameter space containing all the possible values for each
                hyperparameter.
            - model_type (str, default: 'MLP'):
                type of model implemented by the individual ('MLP' for a
                Multi-Layer Perceptron, 'CNN' for a Convolutional Neural Network).
            - pop_size (int, default: 10):
                size of the population.
            - selection_perc (float, default: 0.5):
                percentage of individuals to to be considered in order to 
                generate the offsprings for the next population.
                The individuals are considered w.r.t. their performance.
            - mutation_chance (float, default: 0.2):
                probability of having a mutation during the generation of an
                offspring.
        """

        self._model_type = model_type
        self._pop_size = pop_size
        self._hp_choices = hp_choices
        self._selection_perc = selection_perc
        self._mutation_chance = mutation_chance
    
    
    def populate(self):

        """
        Generates a population of Network individuals.
        Population size, model type and hyperparameter space are specified in
        the initialization phase.
        
        Returns:
            Network list representing the population.
        """

        pop = []
        for _ in range(self._pop_size):
            individual = Network(self._model_type, self._hp_choices)
            individual.select_random_hp()
            pop.append(individual)
            
        return pop
    
        
    def evaluate(self, pop, x_train, y_train, x_val, y_val):
        
        """
        Evaluates the given population training it over the specified train data
        and validating it over the specified val data.
        The metric is ... .

        Parameters: 
            - pop:
                population to be evaluated.
            - x_train (float np.array):
                values of the train traces.
            - y_train (0/1 list):
                one-hot-encoding of the train labels (all 0s but a single 1
                in position i to represent label i).
            - x_val (float np.array):
                values of the val traces.
            - y_val (0/1 list):
                one-hot-encoding of the val labels (all 0s but a single 1
                in position i to represent label i).
        
        Returns:
            tuple list representing the evaluation.
            Each tuple contains a Network, which represents an individual, and
            a ..., which represents the individual's performance.
            The tuples are sorted w.r.t. the performance (best to worst).
        """

        fitness_values = []
        for i, individual in enumerate(pop):
            print()
            print(f'***** Individual {i+1}/{self._pop_size} *****')
            
            individual.build_model()
            val_acc = individual.train_and_val(x_train, y_train, x_val, y_val) # Default train and val w.r.t. accuracy
            fitness_values.append(val_acc) 
        
        evaluation = list(zip(pop, fitness_values))
        
        return evaluation
    
    
    def select(self, evaluation):

        """
        Selects only a subset of individuals, the ones with best performance.
        The number of selections is defined in the initialization phase as a 
        percentage.

        Parameters:
            - evaluation (tuple list):
                ranking of the individuals of a population w.r.t. their 
                performance (best to worst).
        
        Returns:
            Network list containing the best-performing individuals of the
            population.
        """

        num_selected = int(self._selection_perc * self._pop_size)
        evaluation.sort(key=lambda x: -x[1]) # Sort the individuals w.r.t. their accuracy (the higher the better, so the "-" is needed)
        sorted_pop = [individual for individual, _ in evaluation]
        parents = sorted_pop[:num_selected]
        
        return parents
        
    
    def _produce_offspring(self, parentA, parentB):

        """
        Generation of an offspring starting from two individuals.

        Parameters:
            - parentA (Network):
                individual.
            - parentB (Network):
                individual.
        
        Returns:
            dict containing the hyperparameters of the offspring generated by
            the given individuals.
        """

        # An offspring contains hps from both parents (random selection)
        offspring_hp = {hp_name: random.choice([parentA.get_hp(hp_name), parentA.get_hp(hp_name)]) 
                        for hp_name in self._hp_choices}
        
        return offspring_hp
        
    
    def _mutate_offspring(self, offspring_hp):

        """
        Mutates the specified offspring.
        
        Parameters:
            - offspring_hp (dict):
                hyperparameters of an offspring.

        Returns:
            dict containing the hyperparameters of the mutated offspring (only
            a random hyperparameter is changed w.r.t. the ones given in input).
        """

        # A mutation is a random hp from the possible choices
        to_mutate = random.choice(list(self._hp_choices.keys()))
        offspring_hp[to_mutate] = random.choice(self._hp_choices[to_mutate])
        
        return offspring_hp
    
    
    def evolve(self, parents):

        """
        Evolves a population generating offsprings.
        A new population is generated starting from the best-performing individuals
        of a previous genaration.
        The remaining individuals are offspring of the best-performing individuals.
    
        Parameters:
            - parents (Network list):
                best-performing individuals of the previous generation.

        Returns:
            Network list representing a the population for the next generation,
            containing the best-performing individuals of the previous generation
            and their offsprings.
        """

        # Only the individuals with the best performances are kept and used to generate offsprings
        # The size of the population is the same
        num_offsprings = self._pop_size - len(parents)
        
        offsprings = []
        for _ in range(num_offsprings):
            parentA, parentB = random.sample(parents, k=2)
            offspring_hp = self._produce_offspring(parentA, parentB)
            
            if self._mutation_chance > random.random():
                offspring_hp = self._mutate_offspring(offspring_hp)
            
            offspring = Network(self._model_type, self._hp_choices)
            offspring.set_hp(offspring_hp)
            
            offsprings.append(offspring)
            
        new_pop = parents + offsprings
        
        return new_pop

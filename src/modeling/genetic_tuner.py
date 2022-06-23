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

from network import Individual


class GeneticTuner():
    
    """
    Class dedicated to the implementation of a Genetic Algorithm for Neural
    Network Hyperparameter Tuning.

    Attributes:
        - _network_type:
            type of network implemented by each individual of the population.
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


    def __init__(self, hp_choices, network_type='MLP', pop_size=10, 
                 selection_perc=0.5, random_selection_chance=0.1, 
                 mutation_chance=0.1):
        
        """
        Class constructor: gidven the hyperparameter space, a network type, a
        population size, a selection percentage and a mutation probability, a 
        new GeneticTuner is initialized.

        Parameters:
            - hp_choices (dict):
                hyperparameter space containing all the possible values for each
                hyperparameter.
            - network_type (str, default: 'MLP'):
                type of network implemented by the individual ('MLP' for
                Multi-Layer Perceptron, 'CNN' for Convolutional Neural Network).
            - pop_size (int, default: 10):
                size of the population.
            - selection_perc (float, default: 0.5):
                percentage of individuals to to be considered in order to 
                generate the offsprings for the next population (parents).
                The individuals are considered w.r.t. their performance.
            - random_selection_chance (float, default: 0.1):
                probability of selecting a bad-perfoming individual as parent for
                the next population.
            - mutation_chance (float, default: 0.1):
                probability of having a mutation during the generation of an
                offspring.
        """

        self._network_type = network_type
        self._pop_size = pop_size
        self._hp_choices = hp_choices
        self._selection_perc = selection_perc
        self._random_selection_chance = random_selection_chance
        self._mutation_chance = mutation_chance
    
    
    def populate(self):

        """
        Generates a population of Individuals.
        Population size, model type and hyperparameter space are specified in
        the initialization phase.
        
        Returns:
            Individual list representing the population.
        """

        pop = []
        for _ in range(self._pop_size):
            individual = Individual(self._network_type, self._hp_choices)
            individual.select_random_hp()
            pop.append(individual)
            
        return pop
    
        
    def evaluate(self, pop, x_train, y_train, epochs, x_val, y_val):
        
        """
        Evaluates the given population training it over the specified train data
        and validating it over the specified val data.
        The metric is the validation accuracy.

        Parameters: 
            - pop:
                population to be evaluated.
            - x_train (float np.array):
                values of the train traces.
            - y_train (0/1 list):
                one-hot-encoding of the train labels (all 0s but a single 1
                in position i to represent label i).
            - epochs (int):
                number of epochs of the training phase.
            - x_val (float np.array):
                values of the val traces.
            - y_val (0/1 list):
                one-hot-encoding of the val labels (all 0s but a single 1
                in position i to represent label i).
        
        Returns:
            tuple list representing the evaluation.
            Each tuple contains a Individual object, which represents an 
            individual, and a float number, which represents the individual's 
            performance.
            The tuples are sorted w.r.t. the performance (best to worst).
        """
        
        # Define a set of callbacks for the models
        cb = {'es': True,
              'reduceLR': True}
        
        # Train and evaluate the population
        fitness_values = []
        for i, individual in enumerate(pop):
            print(f'Training individual {i+1}/{self._pop_size}...')
            
            individual.build_model()
            _ = individual.train_model(x_train, 
                                       y_train, 
                                       epochs=epochs,
                                       cb=cb,
                                       validate=True,
                                       x_val=x_val, 
                                       y_val=y_val)
            val_acc = individual.evaluate(x_val, y_val)
            fitness_values.append(val_acc) 
        
        evaluation = list(zip(pop, fitness_values))
        
        return evaluation
    
    
    def select(self, evaluation):

        """
        Selects the parents for the next population as best-performing individuals
        of the current population and some randomly-chosen bad-performing 
        individuals.
        The number of best-performing individuals to be selected and the chance
        to select a bad-performing individual are defined in the initialization
        phase.

        Parameters:
            - evaluation (tuple list):
                ranking of the individuals of a population w.r.t. their 
                performance (best to worst).
        
        Returns:
            Individual list representing the parents for the next population 
            (individuals that will produce offsprings).
        """
        
        # Sort the individuals w.r.t. their val accuracy (the higher the better)
        evaluation.sort(key=lambda x: -x[1])
        sorted_pop = [individual for individual, _ in evaluation]
        
        # Select the best-perfoming individuals as parents for the next population
        num_selected = int(self._selection_perc * self._pop_size)
        parents = sorted_pop[:num_selected]
        
        # Select also some bad-perfoming individuals to add diversity in the 
        # next population
        for i in sorted_pop[num_selected:]:
            if self._random_selection_chance > random.random():
                parents.append(i)

        return parents
        
    
    def _produce_offspring(self, parentA, parentB):

        """
        Generation of an offspring starting from two individuals.

        Parameters:
            - parentA (Individual):
                parent individual.
            - parentB (Individual):
                parent individual.
        
        Returns:
            dict containing the hyperparameters of the offspring generated by
            the given individuals.
        """

        # An offspring contains hps from both parents (random selection)
        offspring_hp = {hp_name: 
                        random.choice([parentA.get_hp()[hp_name], parentA.get_hp()[hp_name]]) 
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
            - parents (Individual list):
                best-performing individuals of the previous generation.

        Returns:
            Individual list representing the population for the next generation,
            containing the best-performing individuals of the previous generation
            and their offsprings.
        """

        num_offsprings = self._pop_size - len(parents)
        
        offsprings = []
        for _ in range(num_offsprings):
            parentA, parentB = random.sample(parents, k=2)
            offspring_hp = self._produce_offspring(parentA, parentB)
            
            if self._mutation_chance > random.random():
                offspring_hp = self._mutate_offspring(offspring_hp)
            
            offspring = Individual(self._network_type, self._hp_choices)
            offspring.set_hp(offspring_hp)
            
            offsprings.append(offspring)
            
        new_pop = parents + offsprings
        
        return new_pop

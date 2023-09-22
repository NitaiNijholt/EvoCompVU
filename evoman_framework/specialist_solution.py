from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import sys

class Evolve:

    def __init__(self, experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, enemy=8):
        self.env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)
        
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = (self.env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

        self.tournament_lambda = tournament_lambda
        self.survivor_lambda = survivor_lambda
        self.recombination = recombination
        self.survivor_mode = survivor_selection
        self.k = k
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.dom_u = 1
        self.dom_l = -1
        self.mutation_probability = mutation_probability
        self.generations = generations
        self.population_size = population_size
        self.original_population_size = population_size

        self.population = self.initialize()
        self.fitness_population = self.get_fitness()
        self.best = np.argmax(self.fitness_population)
        self.mean = np.mean(self.fitness_population)
        self.std = np.std(self.fitness_population)

    def initialize(self):
        if self.survivor_mode != 'lambda,mu':
            self.survivor_lambda = 100
        return np.random.uniform(self.dom_l, self.dom_u, (self.population_size, self.n_vars))

    # Runs simulation, returns fitness f
    def simulation(self, individual):
        f,p,e,t = self.env.play(pcont=individual)
        return f
    
    # normalizes
    def norm(self, fitness_individual):
        return max(0.0000000001, (fitness_individual - min(self.fitness_population)) / (max(self.fitness_population) - min(self.fitness_population)))


    def similarity_score(self, individual1, individual2):
        '''
        Calculate the similarity score between two individuals using the Euclidean distance.
        
        Parameters:
        - individual1 (np.ndarray): The first individual's genotype (e.g., neural network weights).
        - individual2 (np.ndarray): The second individual's genotype (e.g., neural network weights).
        
        Returns:
        - float: The similarity score, which is inversely proportional to the Euclidean distance.
        '''
        
        # Compute the Euclidean distance between the two individuals
        return np.sqrt(np.sum((individual1 - individual2)**2))

    def share(self, d: np.array, sigma=1, alpha=1):
        '''
        Adjust the similarity score based on the distance between individuals.
        If the distance (d) is less than or equal to a threshold (sigma), then the similarity is adjusted using a scaling factor.
        Otherwise, the similarity is set to zero.
        
        Parameters:
        - d (np.array): The distances between individuals.
        - sigma (float, default=0): The threshold distance for similarity adjustment.
        - alpha (float, default=0): The scaling factor to adjust the similarity.
        
        Returns:
        - np.array: The adjusted similarity scores.
        '''
        
        # Vectorized operations to compute similarity scores
        similarity = np.where(d <= sigma, 1 - (d / sigma) ** alpha, 0)
        
        return similarity



    # evaluation
    def get_fitness(self, population=None, fitness_sharing=True):
        """Calculate the fitness of individuals in a population based on the simulation results. 
        If fitness sharing is enabled, the fitness of an individual is adjusted based on its similarity to others.

        Parameters:
        - population (list or np.ndarray, default=0): List of genotypes. A genotype represents the values for the weights in a neural network.
        - fitness_sharing (int, default=0): A flag to enable or disable fitness sharing. If set to 1, fitness sharing is enabled.

        Returns:
        - np.ndarray: Array containing the fitness values of the individuals in the population.
        """
        if population is None:
            population = self.population
        # Check if the provided population is a numpy array
        if type(population) == np.ndarray:
            # Calculate the fitness for each individual in the provided population using the simulation method
            fitness = np.array([self.simulation(individual) for individual in population])
        else:
            # Calculate the fitness for each individual in the default population using the simulation method
            fitness = np.array([self.simulation(individual) for individual in self.population])


        # If fitness sharing is enabled
        if fitness_sharing:
            # Initialize a zero vector to store the cumulative similarity scores for each individual
            similarity_vector = np.zeros(fitness.shape[0])
            # Loop through each individual in the population
            for index, individual_1 in enumerate(population):
                commulative_similarity = 0  # Initialize cumulative similarity for the current individual
                
                # Calculate the similarity score of the current individual with every other individual in the population
                for individual_2 in population:
                    commulative_similarity += self.share(self.similarity_score(individual_1, individual_2))
                
                # Store the cumulative similarity score for the current individual
                similarity_vector[index] = commulative_similarity
            
            # Adjust the fitness of each individual based on its cumulative similarity score
            fitness_shared = fitness/similarity_vector
            return fitness_shared
        return fitness


    def tournament(self):
        '''
        Implements the tournament selection algorithm. 
        It draws randomly with replacement k individuals and returns the fittest individual.
        '''
        
        # Select k random indexes from the population
        k_indexes = np.random.randint(0, len(self.population), k)
        selected_individuals = np.array([self.population[index] for index in k_indexes])

        # Compute the fitness of the selected individuals
        fitness_of_individuals = self.fitness_population[k_indexes]

        # Sort the individuals based on their fitness
        sorted_indices = np.argsort(fitness_of_individuals)[::-1]

        # Get the lambda best individuals
        return [selected_individuals[i] for i in sorted_indices[:self.tournament_lambda]]
    
    # limits
    def limits(self, x):
        if x > self.dom_u:
            return self.dom_u
        if x < self.dom_l:
            return self.dom_l
        return x

    def uniform_crossover(self, mating_pool, offspring):
        cross_prop = np.random.uniform()

        for j in range(len(offspring[0])):
            if np.random.uniform() < cross_prop:
                offspring[0][j] = mating_pool[0][j]
                offspring[1][j] = mating_pool[1][j]
            else:
                offspring[0][j] = mating_pool[1][j]
                offspring[1][j] = mating_pool[0][j]

        return offspring
    
    def line_recombination(self, mating_pool, offspring):
        for individual in offspring:
            alpha = np.random.uniform(-0.25, 1.25)
            for i in range(len(individual)):
                individual[i] = mating_pool[0][i] + alpha * (mating_pool[0][i] - mating_pool[1][i])
        return offspring

    def reproduce(self):
        total_offspring = []

        # Loop over number of reproductions
        for reproduction in range(int(self.survivor_lambda / 100 * len(self.population) / self.n_offspring)):

            # Make mating pool according to tournament selection
            mating_pool = np.array([self.tournament()[j] for _ in range(int(self.n_parents / self.tournament_lambda)) for j in range(self.tournament_lambda)])

            offspring =  np.zeros((self.n_offspring, self.n_vars))

            if self.recombination == 'uniform':
                offspring = self.uniform_crossover(mating_pool, offspring)
            elif self.recombination == 'line':
                offspring = self.line_recombination(mating_pool, offspring)

            for individual in offspring:
                # Mutates and ensures no weight is outside the range [-1, 1]
                individual = self.mutate(individual)
                individual = [self.limits(weight) for weight in individual]
                total_offspring.append(individual)

        return np.array(total_offspring)
    
    def mutate(self, individual):
        # Mutates the offspring
        for i in range(len(individual)):
            if np.random.uniform() <= self.mutation_probability:
                individual[i] += np.random.normal(0, 0.5)
        return individual
    

    def survivor_selection(self, offspring, fitness_offspring):
        if self.survivor_mode == 'lambda,mu':

            # in (mu, lamda_ survivor selection the offspring replaces all the parents
            self.population = offspring

            # find the lambda_ fittest individuals
            fittest = np.argsort(fitness_offspring)[::-1][:self.population_size]

            # select the fittest individuals from the population
            self.population = self.population[fittest]

            self.fitness_population = self.get_fitness(self.population)

        elif self.survivor_mode == 'roulette':

            # Add offspring to existing population. Population size is now much bigger
            self.population = np.vstack((self.population, offspring))
            self.fitness_population = np.append(self.fitness_population, fitness_offspring)

            # Find individual with the best fitness
            self.best = np.argmax(self.fitness_population)

            # Avoiding negative probabilities, as fitness is ranges from negative numbers
            fitness_population_normalized = np.array([self.norm(fitness_individual) for fitness_individual in self.fitness_population])

            # Calculate probability of surviving generation according to fitness individuals
            probs = fitness_population_normalized/sum(fitness_population_normalized)

            # Pick population_size individuals at random, weighted according to their fitness
            chosen = np.random.choice(len(self.population), self.population_size, p=probs, replace=False)

            # Delete first individual and replace it with the best individual. This raises the average fitness, as the best individual probably already was in the selection made, and now has a clone in the population
            chosen = np.append(chosen[1:], self.best)

            # Delete individuals in population which were not selected above
            self.population = self.population[chosen]
            self.fitness_population = self.fitness_population[chosen]

    def run(self):

        self.env.state_to_log() 
        ini_g = 0
        print(f"GENERATION {ini_g} {round(self.fitness_population[self.best], 6)} {round(self.mean, 6)} {round(self.std, 6)}")

        for i in range(ini_g + 1, self.generations):

            # New individuals by crossover
            offspring = self.reproduce()
            fitness_offspring = self.get_fitness(offspring)

            self.survivor_selection(offspring, fitness_offspring)

            self.best = np.argmax(self.fitness_population)
            self.std  =  np.std(self.fitness_population)
            self.mean = np.mean(self.fitness_population)

            print(f"GENERATION {i} {round(self.fitness_population[self.best], 6)} {round(self.mean, 6)} {round(self.std, 6)}")


os.environ["SDL_VIDEODRIVER"] = "dummy"
population_size = 100
generations = 30
mutation_probability = 0.2
n_hidden_neurons = 10

# 'line' or 'uniform'
recombination = 'line'

# 'lambda,mu' or 'roulette'
survivor_selection = 'roulette'
k = 5
tournament_lambda = 2
survivor_lambda = 120
n_parents = 2
n_offspring = 2
experiment_name = 'optimization_test'
evolve = Evolve(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda)
evolve.run()
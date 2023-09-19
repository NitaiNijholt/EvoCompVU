from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os


class Evolve:
    def __init__(self, experiment_name, n_hidden_neurons, population_size, generations, mutation_probability):
        self.env = Environment(experiment_name=experiment_name,
                  enemies=[3],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)
        
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = (self.env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

        self.dom_u = 1
        self.dom_l = -1
        self.mutation_probability = mutation_probability
        self.generations = generations
        self.population_size = population_size

        self.population = self.generate_population()
        self.fitness_population = self.get_fitness()
        self.best = np.argmax(self.fitness_population)
        self.mean = np.mean(self.fitness_population)
        self.std = np.std(self.fitness_population)
                
    def generate_population(self):
        return np.random.uniform(self.dom_l, self.dom_u, (self.population_size, self.n_vars))

    def simulation(self, individual):
        """
        Plays the game with the individual, returns the fitness according to proposed formula
        """ 
        f,p,e,t = self.env.play(pcont=individual)
        return f
    
    def norm(self, fitness_individual):
        """
        Normalizes the fitness functions to be within the range of 0 and 1
        """ 
        return max(0.0000000001, (fitness_individual - min(self.fitness_population)) / (max(self.fitness_population) - min(self.fitness_population)))

    def get_fitness(self, population=None):
        """
        Population is a list of genotypes, where the genotype is the values for the weights in the neural network
        returns array of the fitness of the individuals in the population
        """
        if type(population) == np.ndarray:
            return np.array([self.simulation(individual) for individual in population])
        return np.array([self.simulation(individual) for individual in self.population])

    def tournament(self, k):
        '''
        Implements the tournament selection algorithm. 
        It draws randomly with replacement k individuals and returns the fittest individual.
        '''
        
        # Choose a random individual and score it
        number_individuals = len(self.population)
        current_winner = np.random.randint(number_individuals)

        # Get the score which is the one to beat
        score = self.fitness_population[current_winner]
        
        # We already have one candidate, so we are left with k-1 to choose
        for i in range(k-1):
            new_score = self.fitness_population[candidate:=np.random.randint(number_individuals)]
            if new_score < score:
                current_winner = candidate
                score = new_score

        return self.population[current_winner]
    
    # limits
    def limits(self, x):

        if x > self.dom_u:
            return self.dom_u
        if x < self.dom_l:
            return self.dom_l
        return x

    # crossover
    def crossover(self):

        total_offspring = []
        n_offspring = 2
        k = 3

        # Loop over number of parent pairs/triplets/whatever
        for p in range(int(self.population.shape[0]/n_offspring)):

            # Make mating pool according to tournament selection
            mating_pool = [self.tournament(k) for i in range(n_offspring)]
            offspring =  np.zeros((n_offspring, self.n_vars))
            cross_prop = np.random.uniform()

            for j in range(len(offspring[0])):
                if np.random.uniform() < cross_prop:
                    offspring[0][j] = mating_pool[0][j]
                    offspring[1][j] = mating_pool[1][j]
                else:
                    offspring[0][j] = mating_pool[1][j]
                    offspring[1][j] = mating_pool[0][j]

            total_offspring.append(offspring[0])
            total_offspring.append(offspring[1])

        return np.array(total_offspring)
    

    def mutate(self, individual):
        # Mutates the offspring
        for i in range(len(individual)):
            if np.random.uniform() <= self.mutation_probability:
                individual[i] += np.random.normal(0, 1)

        # Ensures no weight is outside the range [-1, 1]
        individual = [self.limits(weight) for weight in individual]
        return individual

    def run(self):

        self.env.state_to_log() 
        ini_g = 1
        print(f"GENERATION {ini_g} {round(self.fitness_population[self.best],6)} {round(self.mean,6)} {round(self.std,6)}")

        for i in range(ini_g, self.generations + 1):

            # New individuals by crossover
            offspring = self.crossover()

            offspring = [self.mutate(individual) for individual in offspring]

            fitness_offspring = self.get_fitness(offspring)


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
            chosen = np.random.choice(self.population.shape[0], self.population_size, p=probs, replace=False)

            # Delete first individual and replace it with the best individual. This raises the average fitness, as the best individual probably already was in the selection made, and now has a clone in the population
            chosen = np.append(chosen[1:], self.best)

            # Delete individuals in population which were not selected above
            self.population = self.population[chosen]
            self.fitness_population = self.fitness_population[chosen]

            self.best = np.argmax(self.fitness_population)
            self.std = np.std(self.fitness_population)
            self.mean = np.mean(self.fitness_population)


            print(f"GENERATION {i} {round(self.fitness_population[self.best],6)} {round(self.mean,6)} {round(self.std,6)}")



os.environ["SDL_VIDEODRIVER"] = "dummy"
population_size = 100
generations = 30
mutation_probability = 0.2
experiment_name = 'optimization_test'
n_hidden_neurons = 10
evolve = Evolve(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability)
evolve.run()
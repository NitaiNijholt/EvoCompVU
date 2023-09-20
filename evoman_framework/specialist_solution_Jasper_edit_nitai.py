from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os


class Evolve:
    def __init__(self, experiment_name, n_hidden_neurons, population_size, generations, mutation_probability):
        self.env = Environment(experiment_name=experiment_name,
                  enemies=[8],
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
        self.original_population_size = population_size

        self.population = self.generate_population()
        self.fitness_population = self.get_fitness()
        self.best = np.argmax(self.fitness_population)
        self.mean = np.mean(self.fitness_population)
        self.std = np.std(self.fitness_population)
                
    def generate_population(self):
        return np.random.uniform(self.dom_l, self.dom_u, (self.population_size, self.n_vars))

    # Runs simulation, returns fitness f
    def simulation(self, individual):
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
    def crossover(self, n_offspring=2, k=3):

        total_offspring = []
        

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

            # Mutates the offspring
            for individual in offspring:
                for i in range(len(individual)):
                    if np.random.uniform() <= self.mutation_probability:
                        individual[i] += np.random.normal(0, 1)

                # Ensures no weight is outside the range [-1, 1]
                individual = [self.limits(weight) for weight in individual]

                total_offspring.append(individual)

        return np.array(total_offspring)
    
    def run(self):

        self.env.state_to_log() 
        ini_g = 1
        print(f"GENERATION {ini_g} {round(self.fitness_population[self.best],6)} {round(self.mean,6)} {round(self.std,6)}")

        # save original population size
        original_population_size = self.population.shape[0]

        # set lambda_ to the amounr of fittest individuals we want to keep each generation
        lambda_ = 5

        for i in range(ini_g, self.generations + 1):


            # New individuals by crossover
            offspring = self.crossover()
            fitness_offspring = self.get_fitness(offspring)
            # print(fitness_offspring)

            
            # in (mu, lamda_ survivor selection the offspring replaces all the parents
            self.population = offspring

            # find the lambda_ fittest individuals
            fittest = np.argsort(fitness_offspring)[::-1][:lambda_]

            # print statements, might come in handy for code understanding & debugging
            # print(fittest)
            # fittest_individual_index = fittest[0]
            # print(fittest_individual_index)
            # print(fitness_offspring[fittest_individual_index])

            # select the fittest individuals from the population
            self.population = self.population[fittest]

            # Pick population_size individuals at random, save the indicies
            chosen = np.random.choice(self.population.shape[0], self.original_population_size, replace=True) # replace=False means no duplicates, i.e. no individual can be selected twice. Maybe this should be True?

            # create new population from chosen individuals
            new_population = np.array([self.population[i] for i in chosen])

            # # Update population and fitness_population
            self.population = new_population
            self.fitness_population = self.get_fitness(self.population)

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
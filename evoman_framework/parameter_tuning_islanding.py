from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import sys
import optuna

class Evolve:

    def __init__(self, experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, migration_frequency, migration_amount, num_islands, enemy=8):
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
        self.migration_frequency = migration_frequency
        self.migration_amount = migration_amount

        self.population = self.initialize()
        self.fitness_population = self.get_fitness()
        self.best = np.argmax(self.fitness_population)
        self.mean = np.mean(self.fitness_population)
        self.std = np.std(self.fitness_population)

        self.num_islands = num_islands
        self.islands = [self.initialize() for _ in range(self.num_islands)]
        self.fitness_islands = [self.get_fitness(island) for island in self.islands]



    def initialize(self):
        if self.survivor_mode != 'lambda,mu':
            self.survivor_lambda = self.population_size
        return np.random.uniform(self.dom_l, self.dom_u, (self.population_size, self.n_vars))
    
    def migrate(self):
        for i in range(self.num_islands):
            # Select the top 10 fittest individuals from the current island
            top_indices = np.argsort(self.fitness_islands[i])
            
            # Randomly select 3 out of the top 10 for migration
            migrant_indices = np.random.choice(top_indices, self.migration_amount, replace=False)
            migrants = self.islands[i][migrant_indices]

            # Send migrants to the next island (circular migration)
            next_island = (i + 1) % self.num_islands

            # Replace 3 individuals in the next island with migrants
            replace_indices = np.random.choice(self.population_size, self.migration_amount, replace=False)
            try:
                self.islands[next_island][replace_indices] = migrants
            except IndexError:
                print(self.islands[next_island])
                print(replace_indices)
                print(migrants)
                sys.exit()

            # Update the fitness of the next island after migration
            self.fitness_islands[next_island] = self.get_fitness(self.islands[next_island])

    # Runs simulation, returns fitness f
    def simulation(self, individual):
        f,p,e,t = self.env.play(pcont=individual)
        return f
    
    # normalizes
    def norm(self, fitness_individual):
        return max(0.0000000001, (fitness_individual - min(self.fitness_population)) / (max(self.fitness_population) - min(self.fitness_population)))

    # evaluation
    def get_fitness(self, population=0):
        # population is a list of genotypes, where the genotype is the values for the weights in the neural network
        # returns array of the fitness of the individuals in the population
        if type(population) == np.ndarray:
            return np.array([self.simulation(individual) for individual in population])
        return np.array([self.simulation(individual) for individual in self.population])

    def tournament(self):
        '''
        Implements the tournament selection algorithm. 
        It draws randomly with replacement k individuals and returns the fittest individual.
        '''
        
        # Select k random indexes from the population
        k_indexes = np.random.randint(0, len(self.population), self.k)
        selected_individuals = np.array([self.population[index] for index in k_indexes])

        # Compute the fitness of the selected individuals
        fitness_of_individuals = self.fitness_population[k_indexes]

        # Sort the individuals based on their fitness
        sorted_indices = np.argsort(fitness_of_individuals)[::-1]

        # Get the lambda best individuals
        return [selected_individuals[i] for i in sorted_indices[:self.tournament_lambda]]
    
    # limits
    def limits(self, weight):
        if weight > self.dom_u:
            return self.dom_u
        if weight < self.dom_l:
            return self.dom_l
        return weight

    def uniform_crossover(self, mating_pool, offspring):
        cross_prop = np.random.uniform()
        try:
            for j in range(len(offspring[0])):
                if np.random.uniform() < cross_prop:
                    offspring[0][j] = mating_pool[0][j]
                    offspring[1][j] = mating_pool[1][j]
                else:
                    offspring[0][j] = mating_pool[1][j]
                    offspring[1][j] = mating_pool[0][j]
        except:
            print(self.population_size, self.recombination, self.survivor_mode, self.k, self.tournament_lambda)
            print(offspring)
            print(mating_pool)
            sys.exit()
        return offspring
    
    def line_recombination(self, mating_pool, offspring):
        for individual in offspring:
            alpha = np.random.uniform(-0.25, 1.25)
            try:
                for i in range(len(individual)):
                    individual[i] = mating_pool[0][i] + alpha * (mating_pool[1][i]-mating_pool[0][i])
            except IndexError:
                print(self.population_size, self.recombination, self.survivor_mode, self.k, self.tournament_lambda)
                print(individual)
                print(mating_pool)
                sys.exit()
        return offspring

    def reproduce(self):
        total_offspring = []

        # Loop over number of reproductions
        for reproduction in range(int(self.survivor_lambda / self.population_size * len(self.population) / self.n_offspring)):

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
        # print(f"GENERATION {ini_g} {round(self.fitness_population[self.best], 6)} {round(self.mean, 6)} {round(self.std, 6)}")

        for i in range(ini_g + 1, self.generations):
            print(f"GENERATION {i}")
            # Evolution for each island
            for j in range(self.num_islands):
                self.population = self.islands[j]
                self.fitness_population = self.fitness_islands[j]

                # New individuals by crossover
                offspring = self.reproduce()
                fitness_offspring = self.get_fitness(offspring)

                self.survivor_selection(offspring, fitness_offspring)

                self.best = np.argmax(self.fitness_population)
                self.std  =  np.std(self.fitness_population)
                self.mean = np.mean(self.fitness_population)

                # Update the island and its fitness values
                self.islands[j] = self.population
                self.fitness_islands[j] = self.fitness_population

                # print(f"ISLAND {j} - GENERATION {i} {round(self.fitness_population[self.best], 6)} {round(self.mean, 6)} {round(self.std, 6)}")

            # Migration between islands
            if i % self.migration_frequency == 0:
                print(f"{i}, Migration this generation")
                self.migrate()

        # Combine all islands into a single population at the end 
        # self.population = np.vstack(self.islands)
        # self.fitness_population = np.concatenate(self.fitness_islands)



def objective(trial):
    # Sample parameters
    population_size = trial.suggest_int('population_size', 50, 80)
    generations = trial.suggest_int('generations', 10, 11)
    mutation_probability = trial.suggest_float('mutation_probability', 0.01, 0.5)
    recombination = trial.suggest_categorical('recombination', ['line', 'uniform'])
    survivor_selection = trial.suggest_categorical('survivor_selection', ['lambda,mu', 'roulette'])
    k = trial.suggest_int('k', 3, 10)
    tournament_lambda = trial.suggest_int('tournament_lambda', 1, 2)
    survivor_lambda = trial.suggest_int('survivor_lambda', 100, 150)
    n_offspring = trial.suggest_int('n_offspring', 1, 5)
    migration_frequency = trial.suggest_int('migration_frequency', 1, 5)
    migration_amount = trial.suggest_int('migration_amount', 1, 5)
    num_islands = trial.suggest_int('num_islands', 2, 3)
    experiment_name = 'optimization_test'

    n_hidden_neurons = 10
    n_parents = 2
    # Run your evolutionary algorithm with the sampled parameters
    evolve = Evolve(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, migration_frequency, migration_amount, num_islands)
    evolve.run()

    # Return the negative value of the best fitness (since Optuna tries to minimize the objective)
    # Adjust this according to your needs
    return max(evolve.fitness_population)

os.environ["SDL_VIDEODRIVER"] = "dummy"
study = optuna.create_study(direction='maximize')  # or 'maximize' based on your needs
study.optimize(objective, n_trials=2)  # You can adjust n_trials based on your computational resources

print("Best trial:")
trial = study.best_trial
print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice

plot_optimization_history(study)
plot_parallel_coordinate(study)
plot_slice(study)


# os.environ["SDL_VIDEODRIVER"] = "dummy"
# population_size = 100
# generations = 30
# mutation_probability = 0.2
# n_hidden_neurons = 10

# # 'line' or 'uniform'
# recombination = 'line'

# # 'lambda,mu' or 'roulette' REPLACE WORST?
# num_islands = 4
# migration_frequency = 1
# migration_amount = 5
# survivor_selection = 'roulette'
# k = 5
# tournament_lambda = 2
# survivor_lambda = 120
# n_parents = 2
# n_offspring = 2
# experiment_name = 'optimization_test'
# evolve = Evolve(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, migration_frequency, migration_amount, num_islands)
# evolve.run()
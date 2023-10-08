from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os

class EvolveIsland:

    def __init__(self, experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, migration_frequency, migration_amount, num_islands, mutation_stepsize, enemies = [8]):
        self.env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)
        
        self.experiment_name = experiment_name
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = (self.env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

        self.enemies = enemies
        self.current_enemy = self.enemies[0]

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
        self.mutation_stepsize = mutation_stepsize
        self.generations = generations
        self.population_size = population_size
        self.original_population_size = population_size

        self.population = []
        self.best = 0
        self.mean = 0
        self.std = 0

        self.migration_frequency = migration_frequency
        self.migration_amount = migration_amount
        self.num_islands = num_islands
        self.islands = [self.initialize() for _ in range(self.num_islands)]

    def initialize(self):
        if self.survivor_mode != 'lambda,mu':
            self.survivor_lambda = self.population_size

        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.dom_l, self.dom_u, self.n_vars)
            population.append((individual, self.get_fitness(individual=individual)))
        return population
    
    def migrate(self):
        for i in range(self.num_islands):

            # Select the top 10 fittest individuals from the current island
            top_indices = sorted(range(self.population_size), key=lambda i: self.population[i][1], reverse=True)
            
            # Randomly select 3 out of the top 10 for migration
            migrant_indices = np.random.choice(top_indices, self.migration_amount, replace=False)
            migrants = [self.islands[i][index] for index in migrant_indices]

            # Send migrants to the next island (circular migration)
            next_island = (i + 1) % self.num_islands

            # Replace 3 individuals in the next island with migrants
            replace_indices = np.random.choice(self.population_size, self.migration_amount, replace=False)
            for i, index in enumerate(replace_indices):
                self.islands[next_island][index] = migrants[i]


    def simulation(self, individual):
        f,p,e,t = self.env.play(pcont=individual)
        return f, e
    
    def norm(self, fitness_individual):
        return max(0.0000000001, (fitness_individual - min(self.fitness_population)) / (max(self.fitness_population) - min(self.fitness_population)))

    def get_fitness(self, population=None, individual=[], return_dict=False, enemies=[]):
        """
        Calculate the fitness of individuals in a population based on the simulation results. 
        If fitness sharing is enabled, the fitness of an individual is adjusted based on its similarity to others.

        Parameters:
        - population (list or np.ndarray, default=0): List of genotypes. A genotype represents the values for the weights in a neural network.

        Returns:
        - np.ndarray: Array containing the fitness values of the individuals in the population.
        """

        if len(enemies) == 0:
            enemies = self.enemies

        if len(individual) > 0:

            fitness_vs_individual_enemy = []
            energy_per_enemy = []

            for enemy in enemies:
                self.env = Environment(experiment_name=self.experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(self.n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
                
                fitness, enemy_health = self.simulation(np.array(individual))

                # Append fitness of individual vs each enemy
                fitness_vs_individual_enemy.append(fitness)

                # Append energy each enemy after fighting individual
                energy_per_enemy.append(enemy_health)

            # Count the enemies beaten, add 1 to avoid fitness being 0
            beaten = [1 if energy == 0 else 0 for energy in energy_per_enemy]

            total_beaten = sum([1 for energy in energy_per_enemy if energy == 0]) + 0.0001

            # Modified fitness function accounting for enemies beaten
            fitness_individual_total = np.sum(fitness_vs_individual_enemy)/len(enemies)*total_beaten

            if return_dict:

                # Create dictionary with round info
                return {'total_fitness': fitness_individual_total, 'fitness_vs_individual_enemy': fitness_vs_individual_enemy, 'energy_per_enemy': energy_per_enemy, 'beaten': beaten, 'total_beaten': total_beaten}

            return fitness_individual_total

    def tournament(self):
        '''
        Implements the tournament selection algorithm. 
        It draws randomly with replacement k individuals and returns the fittest individual.
        '''
        # Select k random indexes from the population
        k_indexes = np.random.randint(0, len(self.population), self.k)

        # Extract the tuples corresponding to the selected indexes
        selected_individuals = [self.population[index] for index in k_indexes]

        # Convert the list of tuples to a NumPy array
        selected_individuals_array = np.array(selected_individuals, dtype=object)

        # Sort the selected individuals by score in descending order
        sorted_individuals = selected_individuals_array[selected_individuals_array[:, 1].argsort()][::-1]

        # Get the lambda best individuals
        return sorted_individuals[:self.tournament_lambda]
    
    def limits(self, weight):
        if weight > self.dom_u:
            return self.dom_u
        if weight < self.dom_l:
            return self.dom_l
        return weight

    def uniform_crossover(self, mating_pool, offspring):
        cross_prop = np.random.uniform()

        for j in range(len(offspring[0])):
            if np.random.uniform() < cross_prop:
                offspring[0][j] = mating_pool[0][0][j]
                offspring[1][j] = mating_pool[1][0][j]
            else:
                offspring[0][j] = mating_pool[1][0][j]
                offspring[1][j] = mating_pool[0][0][j]

        return offspring
    
    def line_recombination(self, mating_pool, offspring):
        for individual in offspring:
            alpha = np.random.uniform(-0.25, 1.25)
            for i in range(len(individual)):
                individual[i] = mating_pool[0][0][i] + alpha * (mating_pool[1][0][i] - mating_pool[0][0][i])
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
                total_offspring.append((np.array(individual), self.get_fitness(individual=individual)))

        return total_offspring

    
    def mutate(self, individual):

        # Mutates the offspring
        for i in range(len(individual)):
            if np.random.uniform() <= self.mutation_probability:
                individual[i] += np.random.normal(0, self.mutation_stepsize)
        return individual


    def survivor_selection(self, offspring):
        if self.survivor_mode == 'lambda,mu':

            # in (mu, lamda_ survivor selection the offspring replaces all the parents
            self.population = offspring[:]

            # select the fittest individuals from the population
            self.population = sorted(self.population, key=lambda x: x[1], reverse=True)[:self.population_size]

        elif self.survivor_mode == 'tournament':

            # Extend self.population with offspring
            self.population.extend(offspring)

            # Create a new list for the tournament selection
            self.population = [self.tournament()[j] for _ in range(int(self.population_size / self.tournament_lambda)) for j in range(self.tournament_lambda)]

    def run(self):

        self.env.state_to_log() 
        global_plot_data = {}  # To store plotting stats globally across all islands

        for i in range(1, self.generations):
            print(f"GENERATION {i}")

            # Holders for global stats
            global_best_fitness = float('-inf')
            global_mean_fitness = []
            global_std_fitness = []

            # Evolution for each island
            for j in range(self.num_islands):

                self.population = self.islands[j]

                # New individuals by crossover
                offspring = self.reproduce()
                self.survivor_selection(offspring)

                # For local stats
                fitness_values = [individual[1] for individual in self.population]
                local_best = max(fitness_values)
                local_std = np.std(fitness_values)
                local_mean = np.mean(fitness_values)

                # Update the island
                self.islands[j] = self.population

                print(f"ISLAND {j} - GENERATION {i} {round(local_best, 6)} {round(local_mean, 6)} {round(local_std, 6)}")

                # Update global stats
                global_best_fitness = max(global_best_fitness, local_best)
                global_mean_fitness.append(local_mean)
                global_std_fitness.append(local_std)

            # Calculate and store the global mean and std deviation for this generation
            global_mean = np.mean(global_mean_fitness)
            global_std = np.mean(global_std_fitness)

            global_plot_data[i] = (round(global_best_fitness, 6), round(global_mean, 6), round(global_std, 6))
            print(f"GLOBAL STATS - GENERATION {i} {round(global_best_fitness, 6)} {round(global_mean, 6)} {round(global_std, 6)}")

            # Migration between islands
            if i % self.migration_frequency == 0:
                print("Migration this generation")
                self.migrate()

        # Combine all islands into a single population list
        combined_population = [individual for island in self.islands for individual in island]

        # Find the global best individual and their fitness
        global_best_individual = max(combined_population, key=lambda x: x[1])

        return ((global_best_individual[0], global_best_individual[1]), global_plot_data)


    def save(self, filename, description):
        """
        description: THE MOST IMPORTANT variable, you should always include:
            - which EA was run
            - which fitness function was used

        filename: the name of the txt file, doesn't have to end in .txt

        """

        filepath = f"results/islanding/{filename}.txt"


        # Combine all islands into a single population list
        combined_population = [individual for island in self.islands for individual in island]

        # Find the global best individual and their fitness
        best_individual = max(combined_population, key=lambda x: x[1])[0]
        total_fitness, fitness_vs_individual_enemy, energy_per_enemy, beaten, total_beaten = self.get_fitness(individual=best_individual, return_dict=True, enemies=[1, 2, 3, 4, 5, 6, 7, 8]).values()

        with open(filepath, 'w') as f:

            # Write the description
            f.write(f"{description}\n")

            # Write the best individual
            f.write(f"{best_individual}\n")

            # Write the enemy-fitness dictionary
            f.write(f"Overall fitness:  {total_fitness}\n")
            f.write(f"The fitness per enemy: {fitness_vs_individual_enemy}\n")
            f.write(f"The energy per enemy after the simulation: {energy_per_enemy}\n")
            f.write(f"The beaten enemies are: {beaten}\n")
            f.write(f"Total beaten enemies: {int(total_beaten)}\n")
        

# Run the code below only when this script is executed, not when imported.
if __name__ == "__main__":

    os.environ["SDL_VIDEODRIVER"] = "dummy"

    population_size = 50
    generations = 10
    mutation_probability = 0.2
    n_hidden_neurons = 10
    num_islands = 1
    migration_amount = 1
    migration_frequency = 7
    mutation_stepsize = 0.215

    # 'line' or 'uniform'
    recombination = 'line'

    # 'lambda,mu' or 'tournament'
    survivor_selection = 'tournament'
    k = 4
    tournament_lambda = 1
    survivor_lambda = 141
    n_parents = 2
    n_offspring = 2
    experiment_name = 'optimization_test'
    enemies = [1, 2, 3, 4]
    evolve = EvolveIsland(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, migration_frequency, migration_amount, num_islands, mutation_stepsize, enemies)
    evolve.run()
    evolve.save('test2', 'test')
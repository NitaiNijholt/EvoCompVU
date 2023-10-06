from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import sys

class Evolve:

    def __init__(self, experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, mutation_sigma, recombination, survivor_selection, parent_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, sharing_alpha, sharing_sigma, enemies):
        self.env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)
        
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = (self.env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5
        
        self.enemies = enemies
        self.current_enemy = self.enemies[0]

        self.tournament_lambda = tournament_lambda
        self.survivor_lambda = survivor_lambda
        self.recombination = recombination
        self.survivor_mode = survivor_selection
        self.parent_selection = parent_selection
        self.k = k
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.dom_u = 1
        self.dom_l = -1
        self.mutation_probability = mutation_probability
        self.mutation_sigma = mutation_sigma
        self.generations = generations
        self.population_size = population_size
        self.original_population_size = population_size
        self.sharing_alpha = sharing_alpha
        self.sharing_sigma = sharing_sigma

        self.population = self.initialize()
        # indexing because the fitness function was changed to in addition to the fitness value, also return a dictionary with round info. So we index 1 to get the fitness value
        self.fitness_population = self.get_fitness()[0]
        self.best = np.argmax(self.fitness_population)
        self.mean = np.mean(self.fitness_population)
        self.std = np.std(self.fitness_population)

    def initialize(self):
        if self.survivor_mode != 'lambda,mu':
            self.survivor_lambda = self.population_size
        return np.random.uniform(self.dom_l, self.dom_u, (self.population_size, self.n_vars))

    # Runs simulation, returns fitness f
    def simulation(self, individual):
        f,p,e,t = self.env.play(pcont=individual)
        return f, e
    
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

    def share(self, d: np.array):
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
        similarity = np.where(d <= self.sharing_sigma, 1 - (d / self.sharing_sigma) ** self.sharing_alpha, 0)
        
        return similarity



    # evaluation
    def get_fitness(self, population=None, fitness_sharing=False, individual = []):
        """Calculate the fitness of individuals in a population based on the simulation results. 
        If fitness sharing is enabled, the fitness of an individual is adjusted based on its similarity to others.

        Parameters:
        - population (list or np.ndarray, default=0): List of genotypes. A genotype represents the values for the weights in a neural network.
        - fitness_sharing (int, default=0): A flag to enable or disable fitness sharing. If set to 1, fitness sharing is enabled.

        Returns:
        - np.ndarray: Array containing the fitness values of the individuals in the population.
        """
        if len(individual) > 0:

            fitness_vs_individual_enemy = []
            energy_per_enemy = []

            for enemy in self.enemies:
                self.env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
                
                fitness, enemy_health = self.simulation(np.array(individual))

                # append fitness of individual vs each enemy
                fitness_vs_individual_enemy.append(fitness)

                # append energy each enemy after fighting individual
                energy_per_enemy.append(enemy_health)
            
            beaten = [1 if energy == 0 else 0 for energy in energy_per_enemy]

            # count the enemies beaten, add 0.0001 to avoid fitness being 0

            total_beaten = sum([1 for energy in energy_per_enemy if energy == 0]) + 0.0001

            # modified fitness function accounting for enemies beaten
            fitness_individual_total = np.sum(fitness_vs_individual_enemy)/len(self.enemies)*total_beaten


            # create dictionary with round info
            dict_round_info = {'fitness_vs_individual_enemy':fitness_vs_individual_enemy, 'energy_per_enemy':energy_per_enemy, 'beaten':beaten, 'total_beaten': total_beaten}

            return fitness_individual_total, dict_round_info

        if population is None:
            population = self.population

        print(population)
        print(len(population))

        fitness_population = []
        dict_round_info_population = {}
        for individual in self.population:
            fitness_vs_individual_enemy = []
            energy_per_enemy = []

            for enemy in self.enemies:
                self.env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
                
                fitness, enemy_health = self.simulation(np.array(individual))

                # append fitness of individual vs each enemy
                fitness_vs_individual_enemy.append(fitness)

                # append energy each enemy after fighting individual
                energy_per_enemy.append(enemy_health)
                
            beaten = [1 if energy == 0 else 0 for energy in energy_per_enemy]

            # count the enemies beaten, add 0.0001 to avoid fitness being 0
            total_beaten = sum([1 for energy in energy_per_enemy if energy == 0]) + 0.0001

            # modified fitness function accounting for enemies beaten
            fitness_individual_total = np.sum(fitness_vs_individual_enemy)/len(self.enemies)*total_beaten

            # create dictionary with round info
            dict_round_info = {'fitness_vs_individual_enemy':fitness_vs_individual_enemy, 'energy_per_enemy':energy_per_enemy, 'beaten':beaten, 'total_beaten': total_beaten}

            # append fitness of individual to population
            fitness_population.append(fitness_individual_total)
            # append round info of individual to population
            dict_round_info_population[tuple(individual)] = dict_round_info 


    #    If fitness sharing is enabled, adjust the fitness of each individual based on its similarity to others
        if fitness_sharing:
                # Initialize a zero vector to store the cumulative similarity scores for each individual
                similarity_vector = np.zeros(len(fitness_population))
                print(similarity_vector[0])

                # Loop through each individual in the population
                for index, individual_1 in enumerate(population):
                    commulative_similarity = 0  # Initialize cumulative similarity for the current individual
                    
                    # Calculate the similarity score of the current individual with every other individual in the population
                    for individual_2 in population:
                        commulative_similarity += self.share(self.similarity_score(individual_1, individual_2))
                    
                    # Store the cumulative similarity score for the current individual
                    print(similarity_vector.shape)
                    similarity_vector[index] = commulative_similarity
                
                # Adjust the fitness of each individual based on its cumulative similarity score
                fitness_shared = fitness_population/similarity_vector


                # note that the fitness info in the dictionary is pre the adjustment to shared fitness
                return fitness_shared, dict_round_info_population   
                
            # note that the fitness is not shared
        return np.array(fitness_population), dict_round_info_population   


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
                individual[i] = mating_pool[0][i] + alpha * (mating_pool[1][i] - mating_pool[0][i])
        return offspring

    def reproduce(self):
        total_offspring = []

        # Loop over number of reproductions
        for reproduction in range(int(self.survivor_lambda / self.population_size * len(self.population) / self.n_offspring)):

            if self.parent_selection == 'tournament':
                # Make mating pool according to tournament selection
                mating_pool = np.array([self.tournament()[j] for _ in range(int(self.n_parents / self.tournament_lambda)) for j in range(self.tournament_lambda)])

            elif self.parent_selection == 'roulette':
                
                # Avoiding negative probabilities, as fitness is ranges from negative numbers
                fitness_population_normalized = np.array([self.norm(fitness_individual) for fitness_individual in self.fitness_population])

                # Calculate probability of surviving generation according to fitness individuals
                probs = fitness_population_normalized/sum(fitness_population_normalized)

                # Pick population_size individuals at random, weighted according to their fitness
                mating_pool = self.population[np.random.choice(self.population_size, self.n_parents, p=probs, replace=False)]

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
                individual[i] += np.random.normal(0, self.mutation_sigma)
        return individual
    

    def survivor_selection(self, offspring, fitness_offspring):
        if self.survivor_mode == 'lambda,mu':

            # in (mu, lamda_ survivor selection the offspring replaces all the parents
            self.population = offspring

            # find the lambda_ fittest individuals
            fittest = np.argsort(fitness_offspring)[::-1][:self.population_size]

            # select the fittest individuals from the population
            self.population = self.population[fittest]

            # indexing because the fitness function was changed to in addition to the fitness value, also return a dictionary with round info. So we index 1 to get the fitness value
            self.fitness_population = self.get_fitness(self.population)[0]

        elif self.survivor_mode == 'roulette':

            # Add offspring to existing population. Population size is now much bigger
            self.population = np.vstack((self.population, offspring))
            self.fitness_population = np.append(self.fitness_population, fitness_offspring)

            # Find individual with the best fitness

            print(self.fitness_population)
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

        plot_data = {1: (round(self.fitness_population[self.best], 6), round(self.mean, 6), round(self.std, 6))}
        print(f"GENERATION {ini_g} {round(self.fitness_population[self.best], 6)} {round(self.mean, 6)} {round(self.std, 6)}")

        for i in range(ini_g + 1, self.generations):

            # New individuals by crossover
            offspring = self.reproduce()

            # indexing because the fitness function was changed to in addition to the fitness value, also return a dictionary with round info. So we index 1 to get the fitness value
            fitness_offspring = self.get_fitness(offspring)[0]

            self.survivor_selection(offspring, fitness_offspring)

            self.best = np.argmax(self.fitness_population)
            self.std  =  np.std(self.fitness_population)
            self.mean = np.mean(self.fitness_population)

            plot_data[i+1] = (round(self.fitness_population[self.best], 6), round(self.mean, 6), round(self.std, 6))
            print(f"GENERATION {i} {round(self.fitness_population[self.best], 6)} {round(self.mean, 6)} {round(self.std, 6)}")
        
        # Return best individual with their fitness score
        return ((self.population[self.best], round(self.fitness_population[self.best])), plot_data)

    def save(self, filename, description, population_mode = True):
        """
        description: THE MOST IMPORTANT variable, you should always include:
            - which EA was run
            - which fitness function was used

        filename: the name of the txt file, doesn't have to end in .txt

        population_mode: if True, saves the whole population, if False, only saves the best individual

        """
        filepath = f"results/{filename}.txt"
        if filepath[-8:] == '.txt.txt':
            filepath = filepath[:-4]
        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        # Check that filename.txt is non-existent to avoid overwriting long computing work
        assert not os.path.exists(filepath), f"{filepath} already exists."


        if population_mode:
            with open(filepath, 'w') as f:
            # Write the description
                f.write(f"{description}\n")
                # Write the best individual
                f.write(f"{self.population[self.best]}\n")
                for individual in self.population:
                    # Printing rest of the population
                    f.write(f"{individual}\n")
                    fitness_vs_individual_enemy, energy_per_enemy, beaten, total_beaten  = self.get_fitness(individual=individual)[1].values()
                    total_fitness_value = self.get_fitness(individual=individual)[0]
                    # Write the enemy-fitness dictionary
                    f.write(f"total_fitness_value: {total_fitness_value}\n")
                    f.write(f"fitness_vs_individual_enemy: {fitness_vs_individual_enemy}\n")
                    f.write(f"energy_per_enemy:{energy_per_enemy}\n")
                    f.write(f"beaten: {beaten}\n")
                    f.write(f"beaten: {total_beaten}\n")
        else:
            with open(filepath, 'w') as f:
                # Write the description
                f.write(f"{description}\n")
                # Write the best individual
                f.write(f"{self.population[self.best]}\n")
                fitness_vs_individual_enemy, energy_per_enemy, beaten, total_beaten  = self.get_fitness(individual=self.population[self.best])[1].values()
                total_fitness_value = self.get_fitness(individual=self.population[self.best])[0]
                # Write the enemy-fitness dictionary
                f.write(f"total_fitness_value: {total_fitness_value}\n")
                f.write(f"fitness_vs_individual_enemy: {fitness_vs_individual_enemy}\n")
                f.write(f"energy_per_enemy:{energy_per_enemy}\n")
                f.write(f"beaten: {beaten}\n")
                f.write(f"beaten: {total_beaten}\n")




if __name__ == "__main__":

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    population_size = 30
    generations = 1
    mutation_probability = 0.043551807574295706
    mutation_sigma = 0.718031541166932
    n_hidden_neurons = 10

    # 'line' or 'uniform'
    recombination = 'line'

    # 'lambda,mu' or 'roulette'
    survivor_selection = 'roulette'

    # 'roulette' or 'tournament'
    parent_selection = 'roulette'
    k = 3
    tournament_lambda = 2
    survivor_lambda = 120
    n_parents = 2
    n_offspring = 2

    sharing_sigma = 4
    sharing_alpha = 2

    experiment_name = 'optimization_test'
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    evolve = Evolve(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, mutation_sigma, recombination, survivor_selection, parent_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, sharing_alpha, sharing_sigma, enemies)
    evolve.run()
    evolve.save('test2_new', 'Added the enemy fitness dict')
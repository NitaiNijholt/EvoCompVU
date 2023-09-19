import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os


# Set to False to generate visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# Folder where results are stored
experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10



# Initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment
# checks environment state
env.state_to_log() 


####   Optimization for controller solution (best genotype-weights for phenotype-network): Genetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# Number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
population_size = 1000
generations = 30
mutation_probability = 0.2
last_best = 0


# Runs simulation, returns fitness f
def simulation(env, individual):
    f,p,e,t = env.play(pcont=individual)
    return f

# normalizes
def norm(fitness_individual, pfit_pop):
    return max(0.0000000001, (fitness_individual - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop)))


# evaluation
def get_fitness(population):
    # population is a list of genotypes, where the genotype is the values for the weights in the neural network
    # returns array of the fitness of the individuals in the population
    return np.array([simulation(env, individual) for individual in population])



def tournament(population, k):
    '''
    Implements the tournament selection algorithm. 
    It draws randomly with replacement k individuals and returns the fittest individual.
    '''
    
    # First step: Choose a random individual and score it
    number_individuals = len(population)
    current_winner = np.random.randint(number_individuals)

    # Get the score which is the one to beat!
    score = fitness_population[current_winner]
    
    # We already have one candidate, so we are left with k-1 to choose
    for i in range(k-1):
        new_score = fitness_population[candidate:=np.random.randint(0, number_individuals)]
        if new_score < score:
            current_winner = candidate
            score = new_score

    return population[current_winner]


# limits
def limits(x):

    if x > dom_u:
        return dom_u
    if x < dom_l:
        return dom_l
    return x


# crossover
def crossover(population):

    total_offspring = []
    n_offspring = 2
    k = 5

    # Loop over number of parent pairs/triplets/whatever
    for p in range(int(population.shape[0]/n_offspring)):

        # Make mating pool according to tournament selection
        mating_pool = [tournament(population, k) for i in range(n_offspring)]

        offspring =  np.zeros((n_offspring, n_vars))


        cross_prop = np.random.uniform()

        for j in range(len(offspring[0])):
            if np.random.uniform() < cross_prop:
                offspring[0][j] = mating_pool[0][j]
                offspring[1][j] = mating_pool[1][j]
            else:
                offspring[0][j] = mating_pool[1][j]
                offspring[1][j] = mating_pool[0][j]

        for individual in offspring:
            # Mutates the offspring
            for i in range(len(individual)):
                if np.random.uniform() <= mutation_probability:
                    individual[i] += np.random.normal(0, 1)

            # Ensures no weight is outside the range [-1, 1]
            individual = [limits(weight) for weight in individual]

            total_offspring.append(individual)

    return np.array(total_offspring)


# Kills the worst genomes, and replace with new best/random solutions
def doomsday(population, fitness_population):
 
    order = np.argsort(fitness_population)
    orderasc = order[:int(population_size/4)]

    for o in orderasc:
        for j in range(n_vars):
            if np.random.uniform() <= 0.5:
                population[o][j] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist.
            else:
                population[o][j] = population[order[-1:]][0][j] # dna from best

        fitness_population[o] = get_fitness([population[o]])

    return population, fitness_population



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    get_fitness([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

# if not os.path.exists(experiment_name+'/evoman_solstate'):

print( '\nNEW EVOLUTION\n')

population = np.random.uniform(dom_l, dom_u, (population_size, n_vars))
fitness_population = get_fitness(population)
best = np.argmax(fitness_population)
mean = np.mean(fitness_population)
std = np.std(fitness_population)
ini_g = 0
solutions = [population, fitness_population]
env.update_solutions(solutions)

# else:

#     print( '\nCONTINUING EVOLUTION\n')

#     env.load_state()
#     population = env.solutions[0]
#     fitness_population = env.solutions[1]

#     best = np.argmax(fitness_population)
#     mean = np.mean(fitness_population)
#     std = np.std(fitness_population)

#     # finds last generation number
#     file_aux  = open(experiment_name+'/gen.txt','r')
#     ini_g = int(file_aux.readline())
#     file_aux.close()




# Saves results for first population
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fitness_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fitness_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

best_fitness = fitness_population[best]
notimproved = 0

for i in range(ini_g + 1, generations):

    # New individuals by crossover
    offspring = crossover(population)
    fitness_offspring = get_fitness(offspring)

    # population = offspring[:]


    # Add offspring to existing population. Population size is now much bigger
    population = np.vstack((population, offspring))
    fitness_population = np.append(fitness_population, fitness_offspring)

    # Find individual with the best fitness
    best = np.argmax(fitness_population)
    potential_best_fitness = fitness_population[best]


    # Avoiding negative probabilities, as fitness is ranges from negative numbers
    fitness_population_normalized = np.array([norm(fitness_individual, fitness_population) for fitness_individual in fitness_population])

    # Calculate probability of surviving generation according to fitness individuals
    probs = fitness_population_normalized/sum(fitness_population_normalized)

    # Pick population_size individuals at random, weighted according to their fitness
    chosen = np.random.choice(population.shape[0], population_size, p=probs, replace=False)

    # Delete first individual and replace it with the best individual. This raises the average fitness, as the best individual probably already was in the selection made, and now has a clone in the population
    chosen = np.append(chosen[1:], best)

    # Delete individuals in population which were not selected above
    population = population[chosen]
    fitness_population = fitness_population[chosen]


    # Keep track of progress
    if potential_best_fitness <= best_fitness:
        notimproved += 1
    else:
        best_fitness = potential_best_fitness
        notimproved = 0

    # If no progress is being made
    if notimproved >= 15:
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        # Run doomsday
        population, fitness_population = doomsday(population, fitness_population)
        notimproved = 0

    best = np.argmax(fitness_population)
    std  =  np.std(fitness_population)
    mean = np.mean(fitness_population)


    # Saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fitness_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fitness_population[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # Saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # Saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',population[best])

    # Saves simulation state
    solutions = [population, fitness_population]
    env.update_solutions(solutions)
    env.save_state()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state

###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from typing import List, Callable


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10



# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


# limits
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x



# tournament

def tournament(population: List[List[float]], k: int = 1, lambda_: int = 2, fitness_func: Callable[[List[float]], float] = fit_pop) -> List[List[float]]:
    '''Select k random individuals from the population and return the lambda best individuals'''

    # Select k random indexes from the population
    k_indexes = np.random.randint(0, population.shape[0], k)
    list_of_individuals = [population[index] for index in k_indexes]
    
    # Compute the fitness of the selected individuals
    fitness_of_individuals = [fitness_func(individual) for individual in list_of_individuals]
    
    # Sort the individuals based on their fitness
    sorted_indices = np.argsort(fitness_of_individuals)[::-1]

    # Get the lambda best individuals
    best_individuals = [list_of_individuals[i] for i in sorted_indices[:lambda_]]
    
    return best_individuals



def uniform_crossover(parent1: List[float], parent2: List[float]) -> List[float]:
# Initialize the child chromosome with zeros
    child = np.zeros(parent1.shape)
    
    # Iterate over each gene of the parents
    for i in range(parent1.shape[0]):
        # Randomly choose a gene from either parent1 or parent2
        if np.random.rand() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    
    return child

def population_crossover(population: List[List[float]], crossover_rate=0.5, k = 5, lambda_=1, fitness_func=fit_pop):
    """Performs crossover between the population. Returns a new population with the same size as the original with crossover performed on some individuals."""
    # Initialize a new population.
    new_population = np.zeros(population.shape)

    # Iterate over each individual in the population.
    for i in range(population.shape[0]):
        # With a probability determined by the crossover rate, perform crossover.
        if np.random.rand() < crossover_rate:
            # Select two parents using the tournament method.
            parent1 = tournament(population, k, lambda_, fitness_func)
            parent2 = tournament(population, k, lambda_, fitness_func)
            # Perform uniform crossover between the selected parents.
            child = uniform_crossover(parent1, parent2)
            new_population[i] = child
        else:
            # If no crossover is performed, retain the original individual.
            new_population[i] = population[i]
    
    return new_population

def population_mutation(population:List[List[float]], mutation_rate=0.1:float) -> List[List[float]]:
    """Performs mutation on the population."""
    new_population = np.zeros(population.shape)
    # Iterate over each individual in the population.
    for i in range(population.shape[0]):
        # Iterate over each gene in the individual.
        for j in range(population.shape[1]):
            # With a probability determined by the mutation rate, perform mutation.
            if np.random.rand() < mutation_rate:
                # Mutate the gene by adding a small random value to it.
                new_population[i][j] += np.random.normal(0, 1)
                # Still need to ensure gene is within the limits.
            else:
                # If no mutation is performed, retain the original gene.
                new_population[i][j] = population[i][j]
    return new_population

    # crossover
# def crossover(pop):

#     total_offspring = np.zeros((0,n_vars))


#     for p in range(0,pop.shape[0], 2):
#         p1 = tournament(pop)
#         p2 = tournament(pop)

#         n_offspring =   np.random.randint(1,3+1, 1)[0]
#         offspring =  np.zeros( (n_offspring, n_vars) )

#         for f in range(0,n_offspring):

#             cross_prop = np.random.uniform(0,1)
#             offspring[f] = p1*cross_prop+p2*(1-cross_prop)

#             # mutation
#             for i in range(0,len(offspring[f])):
#                 if np.random.uniform(0 ,1)<=mutation:
#                     offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

#             offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

#             total_offspring = np.vstack((total_offspring, offspring[f]))

#     return total_offspring


# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop,fit_pop):

    worst = int(npop/4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0,n_vars):
            pro = np.random.uniform(0,1)
            if np.random.uniform(0,1)  <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist.
            else:
                pop[o][j] = pop[order[-1:]][0][j] # dna from best

        fit_pop[o]=evaluate([pop[o]])

    return pop,fit_pop



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):

    offspring = crossover(pop)  # crossover
    fit_offspring = evaluate(offspring)   # evaluation
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    fit_pop_cp = fit_pop
    fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
    probs = (fit_pop_norm)/(fit_pop_norm).sum()
    chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    chosen = np.append(chosen[1:],best)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]


    # searching new areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:

        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop,fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state

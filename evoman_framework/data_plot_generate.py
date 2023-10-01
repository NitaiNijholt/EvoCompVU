'''
Here we train the algorithm 10 times for 3 enemies and save our 10 best individuals from those 10 training runs
per enemy, but for only one EA at a time.
'''

# Run this file for both our EA variants
from specialist_solution_fitness_sharing import Evolve
# from specialist_solution_islanding import Evolve

from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import json

def train(enemies, network_file, data_file, runs=10):
    '''
    Trains 

    enemies: list of 3 enemies that we pick to train and fight against
    network_file: place where we save our best individual
    data_file: place where we save the avg and best results per generation for all runs for plotting
    runs: the amount of runs that should be done (10!)
    '''
    # Check that network_file and data_file are non-existant to avoid overwriting long computing work
    assert not os.path.exists(network_file), f"{network_file} already exists."
    assert not os.path.exists(data_file), f"{data_file} already exists."

    # Parameters:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    population_size = 805 # TODO change back to 100
    generations = 30 # TODO change back to 30
    mutation_probability = 0.043551807574295706
    mutation_sigma = 0.718031541166932
    n_hidden_neurons = 10
    # 'line' or 'uniform'
    recombination = 'line'
    # 'lambda,mu' or 'roulette'
    survivor_selection = 'roulette'
    k = 3
    tournament_lambda = 2
    survivor_lambda = 116
    n_parents = 2
    n_offspring = 2
    sharing_sigma = 2
    sharing_alpha = 4
    parent_selection = "roulette"
    experiment_name = 'optimization_test'
    # Islanding specific:
    num_islands = 7
    migration_amount = 9
    migration_frequency = 7
    mutation_stepsize = 0.3420204523855799

    # Placeholder for storing best individuals and results for plotting
    best_individuals = dict()
    enemy_results = {}

    best_individual_per_run_per_enemy = {}

    # Call evolve, make sure it returns the rights things for the plotting and for saving the best individuals
    for enemy in enemies:
        best_individual_per_run = {}
        enemy_results[enemy] = []

        for run in range(runs):
            print(run, enemy)
            # Initialize the evolution algorithm class
            # FOR FITNESS SHARING:
            evolve = Evolve(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability,
                            mutation_sigma, recombination, survivor_selection, parent_selection, k, n_parents, n_offspring, tournament_lambda,
                            survivor_lambda, sharing_alpha, sharing_sigma, enemy=enemy)
            """
            CHANGE THIS: COMMENT OUT THE RIGHT PART
            """
            # FOR ISLANDING:
            
            # evolve =  Evolve(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability,
            #                  recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda,
            #                  survivor_lambda, migration_frequency, migration_amount, num_islands, mutation_stepsize,
            #                  enemy=enemy)

            # Run the evolution and get the results for plotting
            best_individual, run_results = evolve.run()
            best_individual = best_individual[0] # Because best_individual was a tuple: (weights, fitness)
        
            enemy_results[enemy].append(run_results)
            best_individual_per_run[run] = best_individual

        # Save 10 individuals for this enemy in the bigger nested dictionary
        best_individual_per_run_per_enemy[enemy] = best_individual_per_run

    # Write results for plotting to data_file
    with open(data_file, 'w') as df:
        for enemy, results in enemy_results.items():
            df.write(f'Enemy {enemy}\n')
            for res in results:
                df.write(str(res) + '\n')
            df.write('\n')

    # Save best individuals to network_file in JSON format
    # Convert ndarray to list before saving
    for enemy in best_individual_per_run_per_enemy:
        for run in best_individual_per_run_per_enemy[enemy]:
            best_individual_per_run_per_enemy[enemy][run] = best_individual_per_run_per_enemy[enemy][run].tolist()
    # Save
    with open(network_file, 'w') as nf:
        json.dump(best_individual_per_run_per_enemy, nf)

"""
CHANGE THE FILENAME TO THE FILES YOU WANT TO CREATE THIS TIME
"""
train([6, 7, 8], 'data_champion_fitness_test678.txt', 'data_lineplot_fitness_test678.txt', runs=10)
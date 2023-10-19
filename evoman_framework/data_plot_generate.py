'''
Here we train the algorithm 10 times for each `enemy group` and save our 10 best individuals from those 10 training runs
per enemy, but for only one EA at a time.
'''

# Run this file for both our EA variants
from generalist_solution_islanding import EvolveIsland

from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
import json

def train(network_file, data_file, runs=10):
    '''
    Trains 

    network_file: place where we save our best individual
    data_file: place where we save the avg and best results per generation for all runs for plotting
    runs: the amount of runs that should be done (10!)
    '''
    # Check that network_file and data_file are non-existant to avoid overwriting long computing work
    assert not os.path.exists(network_file), f"{network_file} already exists."
    assert not os.path.exists(data_file), f"{data_file} already exists."

    """
    CHANGE PARAMETERS FOR OTHER EA
    """
    # PARAMETERS
    population_size = 50
    generations = 30
    mutation_probability = 0.105
    n_hidden_neurons = 10
    num_islands = 5
    migration_amount = 10
    migration_frequency = 5
    mutation_stepsize = 0.324
    # 'line' or 'uniform'
    recombination = 'line'
    # 'lambda,mu' or 'tournament'
    survivor_selection = 'tournament'
    k = 4
    tournament_lambda = 1
    survivor_lambda = 41
    n_parents = 2
    n_offspring = 2
    experiment_name = 'optimization_test'
    enemygroups = {f'Team [1,4,6,7]': [1,4,6,7], 'Team [1,2,3,4,5,6,7,8]': [1, 2, 3, 4, 5, 6, 7, 8]}

    # Placeholder for storing best individuals and results for plotting
    best_individuals = dict()
    enemy_results = {}

    best_individual_per_run_per_enemy = {}

    # Call evolve, make sure it returns the rights things for the plotting and for saving the best individuals
    for groupname in enemygroups:
        best_individual_per_run = {}
        enemy_results[groupname] = []

        for run in range(runs):
            print(run, groupname)
            # Initialize the evolution algorithm class
            evolve = EvolveIsland(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, migration_frequency, migration_amount, num_islands, mutation_stepsize, enemygroups[groupname])
            """
            CHANGE EVOLVE CALL FOR OTHER EA
            """
            
            # Run the evolution and get the results for plotting
            best_individual, run_results = evolve.run()
            best_individual = best_individual[0] # Because best_individual was a tuple: (weights, fitness)
        
            enemy_results[groupname].append(run_results)
            best_individual_per_run[run] = best_individual

        # Save 10 individuals for this enemy in the bigger nested dictionary
        best_individual_per_run_per_enemy[groupname] = best_individual_per_run

    # Write results for plotting to data_file
    with open(data_file, 'w') as df:
        for groupname, results in enemy_results.items():
            df.write(f'{groupname}\n')
            for res in results:
                df.write(str(res) + '\n')
            df.write('\n')

    # Save best individuals to network_file in JSON format
    # Convert ndarray to list before saving
    for groupname in best_individual_per_run_per_enemy:
        for run in best_individual_per_run_per_enemy[groupname]:
            best_individual_per_run_per_enemy[groupname][run] = best_individual_per_run_per_enemy[groupname][run].tolist()
    # Save
    with open(network_file, 'w') as nf:
        json.dump(best_individual_per_run_per_enemy, nf)

"""
CHANGE THE FILENAME TO THE FILES YOU WANT TO CREATE THIS TIME

MAKE SURE runs=10
"""
os.environ["SDL_VIDEODRIVER"] = "dummy"
train('data_champion_gen_island.txt', 'data_lineplot_gen_island.txt', runs=10)
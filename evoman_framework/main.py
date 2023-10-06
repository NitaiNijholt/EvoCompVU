
from generalist_solution_islanding import EvolveIsland
import os
from generalist_solution_fitness_sharing import EvolveNiche


def evolve_island(settings):
    experiment_name = settings['experiment name']
    n_hidden_neurons = settings['number of hidden neurons']
    population_size = settings['population size']
    generations = settings['generations']
    mutation_probability = settings['mutation probability']
    recombination = settings['recombination mode']
    survivor_selection = settings['survivor selection mode']
    k = settings['k']
    n_parents = settings['number of parents per reproducion event']
    n_offspring = settings['number of offspring per reproduction event']
    tournament_lambda = settings['tournament lambda']
    survivor_lambda = settings['survivor selection lambda']
    migration_frequency = settings['migration frequency']
    migration_amount = settings['migration size']
    num_islands = settings['number of islands']
    mutation_stepsize = settings['mutation stepsize']
    enemies = settings['enemies']
    return EvolveIsland(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, migration_frequency, migration_amount, num_islands, mutation_stepsize, enemies)

def evolve_niche(settings):
    experiment_name = settings['experiment name']
    n_hidden_neurons = settings['number of hidden neurons']
    population_size = settings['population size']
    generations = settings['generations']
    mutation_probability = settings['mutation probability']
    recombination = settings['recombination mode']
    survivor_selection = settings['survivor selection mode']
    k = settings['k']
    n_parents = settings['number of parents per reproducion event']
    n_offspring = settings['number of offspring per reproduction event']
    tournament_lambda = settings['tournament lambda']
    survivor_lambda = settings['survivor selection lambda']
    mutation_stepsize = settings['mutation stepsize']
    enemies = settings['enemies']
    sharing_alpha = settings['sharing alpha']
    sharing_sigma = settings['sharing sigma']
    parent_selection = settings['parent selection mode']
    return EvolveNiche(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, mutation_stepsize, recombination, survivor_selection, parent_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, sharing_alpha, sharing_sigma, enemies)

def check_filename(filename):
    while True:
        filepath = f"results/{mode}/{filename}.txt"
        if filepath[-8:] == '.txt.txt':
            filepath = filepath[:-4]

        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('results/islanding'):
            os.makedirs('results/islanding')
        if not os.path.exists('results/fitness_sharing'):
            os.makedirs('results/fitness_sharing')

        # Check that filename.txt is non-existent to avoid overwriting long computing work
        if os.path.exists(filepath):
            print(f"The filepath {filepath} already exists.")
            print("Please try again. Type the new filename into the terminal. If you wish not to save the results, just press enter.")
            filename = input()

            # If no new filename is provided, the results will not be saved
            if not filename:
                return
            
        # Save results under provided filename. Function keeps repeating until a valid filename is given
        else:
            return filename
        

def run(mode, settings, filename = None):

    # Ensures provided filename is valid before the simulation starts
    if filename:
        filename = check_filename(filename)

    # Run correct evolutionary algorithm
    if mode == 'islanding':
        evolve = evolve_island(settings)
    elif mode == 'fitness_sharing':
        evolve = evolve_niche(settings)
    
    # Run the chosen algorithm
    evolve.run()
    
    # Give second chance to save results, as the results might be more interesting than previously thought
    if not filename:
        print("Saving is currently disabled.")
        print("This is a second chance to still save the results.")
        print("If you want to keep the results, type the name of the file you wish to save it in. Otherwise just press enter")
        filename = input()
        if filename:
            filename = check_filename(filename)
    
    # Save the results if a valid filename is provided
    if filename:
        evolve.save(filename, settings)

os.environ["SDL_VIDEODRIVER"] = "dummy"

# Pick i for islanding and f for fitness sharing
mode = 'f'

# General settings
settings = {
    'population size': 50,
    'generations': 3,
    'mutation probability': 0.2,
    'number of hidden neurons': 10,
    'mutation stepsize': 0.215,
    'recombination mode': 'line',
    'survivor selection mode': 'tournament',
    'survivor selection lambda': 141,
    'k': 4,
    'tournament lambda': 1,
    'number of parents per reproducion event': 2,
    'number of offspring per reproduction event': 2,
    'experiment name': 'optimization_test',
    'enemies': [1, 2, 3, 4]
}

# Additional settings for the island model
if mode in ['island', 'i', 'islanding']:
    mode = 'islanding'
    settings['number of islands'] = 2
    settings['migration size'] = 5
    settings['migration frequency'] = 7

# Additional settings for the fitness sharing model
elif mode in ['fitness_sharing', 'share_fitness', 'f', 'fs', 'f s', 'f_s', 'fitness sharing', 'share fitness']:
    mode = 'fitness_sharing'
    settings['sharing sigma'] = 4
    settings['sharing alpha'] = 2
    settings['parent selection mode'] = 'roulette'

run(mode, settings)
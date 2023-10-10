from generalist_solution_fitness_sharing import EvolveNiche
from generalist_solution_islanding import EvolveIsland
import os
import sys
import optuna



def evolve_island(settings, tuning, trial=None):
    
    if tuning:
        average = 0
        settings = get_tuning_parameters(settings=settings, mode=mode, trial=trial)

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

    if survivor_selection not in ['tournament', 'lambda,mu']:
        print(f"Invalid survivor selection mode: {survivor_selection}. Pick either 'tournament' or 'lambda,mu'.")
        sys.exit()

    for _ in range(settings['tuning sample size']):
        evolve = EvolveIsland(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, recombination, survivor_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, migration_frequency, migration_amount, num_islands, mutation_stepsize, enemies)
        evolve.run()

        if not tuning:
            return evolve

        best_fitness = sorted(evolve.population, key=lambda x: x[1], reverse=True)[:evolve.population_size][0][1]
        print(best_fitness)
        average += best_fitness / settings['tuning sample size']

    return average

def evolve_niche(settings, tuning, trial = None):

    if tuning:
        average = 0
        settings = get_tuning_parameters(settings, mode, trial=trial)

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

    for _ in range(settings['tuning sample size']):
        evolve= EvolveNiche(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability, mutation_stepsize, recombination, survivor_selection, parent_selection, k, n_parents, n_offspring, tournament_lambda, survivor_lambda, sharing_alpha, sharing_sigma, enemies)
        evolve.run()
        
        if not tuning:
            return evolve

        average += max(evolve.fitness_population) / settings['tuning sample size']
    return average

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
        
def get_tuning_parameters(settings, mode, trial):
    #settings['population size'] = trial.suggest_int('population size', 50, 150)
    settings['mutation probability'] = trial.suggest_float('mutation probability', 0.01, 0.15)
    settings['k'] = trial.suggest_int('k', 3, 10)
    settings['tournament lambda'] = trial.suggest_int('tournament lambda', 1, 2)
    settings['survivor selection lambda'] = settings['population size']
    settings['mutation stepsize'] = trial.suggest_float('mutation stepsize', 0.01, 1)

    if mode == 'islanding':
        settings['migration frequency'] = trial.suggest_int('migration frequency', 1, 10)
        settings['migration size'] = trial.suggest_int('migration size', 1, 20)
        settings['number of islands'] = trial.suggest_int('number of islands', 2, 6)
    elif mode == 'fitness_sharing':
        settings['sharing alpha'] = trial.suggest_float('sharing alpha', 1., 10.)
        settings['haring sigma'] = trial.suggest_float('sharing sigma', 1., 10.)
    return settings


def save(filename, evolve):
    # Give second chance to save results, as the results might be more interesting than previously thought
    if not filename:
        print("Saving is currently disabled.")
        print("This is a second chance to still save the results.")
        print("If you want to keep the results, type the name of the file you wish to save it in. Otherwise just press enter")
        filename = input()
        if filename:
            filename = check_filename(filename)
            print("There is also a probability to save more than 1 individual, but rather the best N. By default, N = 1.")
            print("Type the integer N if another N is desired.")
            N = input()

            if not N:
                N = 1
            else:
                N = int(N)
    
    # Save the results if a valid filename is provided
    if filename:
        evolve.save(filename, settings, N)

def run(mode, settings, filename = None, tuning = False):

    # Ensures provided filename is valid before the simulation starts
    if filename:
        filename = check_filename(filename)

    # Run correct evolutionary algorithm
    if mode == 'islanding':
        if tuning:
            study = optuna.create_study(direction='maximize')  # or 'maximize' based on your needs
            study.optimize(lambda trial: evolve_island(settings=settings, tuning=tuning, trial=trial), n_trials=settings['trials'])

            print("Best trial:")
            trial = study.best_trial
            print("Value: ", trial.value)
            print("Params: ")
            for key, value in trial.params.items():
                print(f"{key}: {value}")

        else:
            evolve = evolve_island(settings, tuning)
            while True:
                evolve.enemies = [1, 2, 3, 4, 5, 6, 7, 8]
                evolve.get_fitness()
                save(filename, evolve)
                print("The individuals are saved, or not depending on your choice. You can now continue this simulation.")
                print("Type anything to continue this simulation. Just type enter to quit")

                if len(input()) > 0:
                    print("Type the amount of generations wished to be added")
                    generations = int(input())
                    evolve.generations = generations
                    settings['generations'] = generations
                    print(f"Now type the enemies you wish to train on. The current enemies are: {settings['enemies']}. Press enter to use the same set")
                    enemies = input()

                    if enemies:
                        enemies = eval(enemies)
                        settings['enemies'] = enemies
                        evolve.enemies = enemies
                    else:
                        evolve.enemies = settings['enemies']
                    evolve.get_fitness()
                    evolve.run()

                else:
                    break



    elif mode == 'fitness_sharing':
        if tuning:
            study = optuna.create_study(direction='maximize')  # or 'maximize' based on your needs
            study.optimize(lambda trial: evolve_niche(settings=settings, tuning=tuning, trial=trial), n_trials=settings['trials'])

            print("Best trial:")
            trial = study.best_trial
            print("Value: ", trial.value)
            print("Params: ")
            for key, value in trial.params.items():
                print(f"{key}: {value}")

        else:
            evolve = evolve_niche(settings, tuning)
            evolve.enemies = [1, 2, 3, 4, 5, 6, 7, 8]
            evolve.get_fitness()



os.environ["SDL_VIDEODRIVER"] = "dummy"







# Pick i for islanding and f for fitness sharing
mode = 'i'

tuning = False

# General settings
settings = {
    'population size': 50,
    'generations': 5,
    'mutation probability': 0.0406,
    'number of hidden neurons': 10,
    'mutation stepsize': 0.30,
    'recombination mode': 'line',
    'survivor selection mode': 'tournament',
    'survivor selection lambda': 141,
    'k': 5,
    'tournament lambda': 2,
    'number of parents per reproducion event': 2,
    'number of offspring per reproduction event': 2,
    'experiment name': 'optimization_test',
    'enemies': [2, 4],
    'tuning sample size': 1
}

# Additional settings for the island model
if mode in ['island', 'i', 'islanding']:
    mode = 'islanding'
    settings['number of islands'] = 6
    settings['migration size'] = 3
    settings['migration frequency'] = 4

# Additional settings for the fitness sharing model
elif mode in ['fitness_sharing', 'share_fitness', 'f', 'fs', 'f s', 'f_s', 'fitness sharing', 'share fitness']:
    mode = 'fitness_sharing'
    settings['sharing sigma'] = 4
    settings['sharing alpha'] = 2
    settings['parent selection mode'] = 'roulette'

if tuning:
    settings['trials'] = 50
    settings['tuning sample size'] = 2


run(mode, settings, tuning=tuning)
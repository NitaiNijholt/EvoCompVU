from specialist_just_for_plot import Evolve
import matplotlib.pyplot as plt
import os

def create_boxplots_per_run(results):
    plt.figure(figsize=(12, 6))

    best_fitness = [run['best_fitness'] for run in results]
    std_deviation = [run['std_deviation'] for run in results]
    mean_fitness = [run['mean_fitness'] for run in results]

    # Create x-axis labels for the boxplots
    labels = [f'Run {i+1}' for i in range(len(best_fitness))]

    plt.subplot(3, 1, 1)
    plt.boxplot(best_fitness, labels=labels)
    plt.xlabel('Run')
    plt.ylabel('Value')
    plt.title('Boxplot of Best Individual Gain for 10 Runs')

    plt.subplot(3, 1, 2)
    plt.boxplot(std_deviation, labels=labels)
    plt.xlabel('Run')
    plt.ylabel('Value')
    plt.title('Boxplot of Standard Deviation for 10 Runs')

    plt.subplot(3, 1, 3)
    plt.boxplot(mean_fitness, labels=labels)
    plt.xlabel('Run')
    plt.ylabel('Value')
    plt.title('Boxplot of Mean Fitness for 10 Runs')

    plt.tight_layout()
    plt.show()

def create_plots_per_gen(generation_data):
    plt.figure(figsize=(12, 6))

    # Plot best fitness
    plt.subplot(2, 2, 1)
    plt.plot(generation_data['generation'], generation_data['best_fitness'], label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Plot mean fitness
    plt.subplot(2, 2, 2)
    plt.plot(generation_data['generation'], generation_data['mean_fitness'], label='Mean Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Plot standard deviation
    plt.subplot(2, 2, 3)
    plt.plot(generation_data['generation'], generation_data['std_deviation'], label='Standard Deviation')
    plt.xlabel('Generation')
    plt.ylabel('Std Deviation')
    plt.legend()

    plt.tight_layout()
    plt.show()

os.environ["SDL_VIDEODRIVER"] = "dummy"
population_size = 100
generations = 5
mutation_probability = 0.2
experiment_name = 'optimization_test'
n_hidden_neurons = 10
num_runs = 1  # Number of runs for each algorithm, adjust as needed

all_results = []

for _ in range(num_runs):
    evolve = Evolve(experiment_name, n_hidden_neurons, population_size, generations, mutation_probability)
    evolve.run()
    all_results.append(evolve.generation_data)  

if num_runs > 1:
    # Create and show boxplots for individual gain fitness, std deviation, and mean fitness
    create_boxplots_per_run(all_results)
else:
    # Create and show plots for individual gain fitness, std deviation, and mean fitness per generation
    create_plots_per_gen(all_results[0])
from two_sample_ttest_module import two_sample_ttest
import json
'''
Use files generated by data_plot_generate.py to make lineplots
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats
from matplotlib.ticker import MultipleLocator

def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines

def extract_data(lines):
    data = {}
    current_enemy = None
    for line in lines:
        line = line.strip()
        if line.startswith("Enemy"):
            current_enemy = int(line.split()[1])
            data[current_enemy] = []
        elif line:
            generations = eval(line)
            for gen, (best, avg, _) in generations.items():
                if len(data[current_enemy]) < gen:
                    data[current_enemy].append({'best': [], 'avg': []})
                data[current_enemy][gen-1]['best'].append(best)
                data[current_enemy][gen-1]['avg'].append(avg)
    return data


# Assymetrical CI, DONT USE:
# def bootstrap_ci(data, n_bootstrap=10000, alpha=0.05):
#     """Calculate the low and high percentiles for a bootstrap confidence interval."""
#     resamples = [np.random.choice(data, len(data), replace=True).mean() for _ in range(n_bootstrap)]
#     return np.percentile(resamples, 100 * alpha / 2), np.percentile(resamples, 100 * (1 - alpha / 2))

# def lineplot(data1, data2):
#     for enemy in data1.keys():
#         plt.figure()
        
#         for data, label_prefix in zip([data1, data2], ["20 pop ", "10 pop "]):
#             gen_numbers = list(range(1, len(data[enemy]) + 1))
            
#             best_means = [np.mean(gen['best']) for gen in data[enemy]]
#             avg_means = [np.mean(gen['avg']) for gen in data[enemy]]

#             # Calculate the lower and upper bounds of the confidence interval
#             ci_lowers, ci_uppers = zip(*[bootstrap_ci(gen['avg']) for gen in data[enemy]])
            
#             plt.plot(gen_numbers, best_means, label=f'{label_prefix}max')
#             plt.plot(gen_numbers, avg_means, label=f'{label_prefix}mean')
            
#             plt.fill_between(gen_numbers, 
#                              ci_lowers,
#                              ci_uppers, 
#                              alpha=0.2)
        
#         plt.title(f"Enemy {enemy}")
#         plt.xlabel('Generation')
#         plt.ylabel('Fitness')
#         plt.legend()
#         plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
#         plt.show()


# Use this:
def lineplot(data1, data2):
    num_enemies = len(data1.keys())
    fig, axs = plt.subplots(1, num_enemies, figsize=(5 * num_enemies, 5))
    hypothesis_tests_avg_total = [] 
    for ax, enemy in zip(axs, data1.keys()):
        hypothesis_test_per_enemy_mean = []
        for data, label_prefix in zip([data1, data2], [" Fitness sharing", " Islanding"]):
            gen_numbers = list(range(1, len(data[enemy]) + 1))
            
            best_means = [np.mean(gen['best']) for gen in data[enemy]]
            best_stds = [np.std(gen['best']) for gen in data[enemy]]
            best_values = [gen['best'] for gen in data[enemy]]
            avg_values = [gen['avg'] for gen in data[enemy]]
            avg_means = [np.mean(gen['avg']) for gen in data[enemy]]
            avg_stds = [np.std(gen['avg']) for gen in data[enemy]]
            avg_stderr = [stats.sem(gen['avg']) for gen in data[enemy]]
            # avg_ci = [1.96 * stderr / np.sqrt(len(data[enemy])) for stderr in avg_stderr] # TODO assumes gaussian distr

            print(len(best_values))
            print(len(best_values[-1]))
            # print(avg_values)
            
            if label_prefix == " Fitness sharing":
                colour = "orange"
            else:
                colour = "green"
   
            line1, = ax.plot(gen_numbers, best_means, label=f'{label_prefix} Max', linestyle='--', color=colour)
            line2, = ax.plot(gen_numbers, avg_means, label=f'{label_prefix} Mean', color=colour)

            
            # changed to std
            ax.fill_between(gen_numbers, 
                             [mean - std for mean, std in zip(avg_means, avg_stds)], 
                             [mean + std for mean, std in zip(avg_means, avg_stds)], 
                             alpha=0.2, color=line2.get_color())
            
            # added shaded area next to best
            ax.fill_between(gen_numbers, 
                             [mean - std for mean, std in zip(best_means, best_stds)], 
                             [mean + std for mean, std in zip(best_means, best_stds)], 
                             alpha=0.2, color=line1.get_color())
            hypothesis_test_per_enemy_mean.append(avg_values[-1])
        hypothesis_tests_avg_total.append(two_sample_ttest(hypothesis_test_per_enemy_mean[0], hypothesis_test_per_enemy_mean[1], alpha = 0.05))
            
        
        ax.set_title(f"Enemy {enemy}")
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(5))  # Set tick interval to 5
        ax.set_ylim(0, max(100, ax.get_ylim()[1]))  # Set y-axis upper limit to at least 100

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()
    return hypothesis_tests_avg_total


# Reading and parsing the data from the files
file1 = 'data_lineplot_fitness_final.txt'
file2 = 'data_lineplot_islanding_final.txt'

data1 = extract_data(read_file(file1))
data2 = extract_data(read_file(file2))

# extract mean and max in last generation


# save mean and max in last generation
# with open('results_mean_max.txt', 'w') as f:
#     json.dump({'mean_max': mean_max, 'max_max': max_max}, f)


# Creating combined plots
testresults = lineplot(data1, data2)
print(testresults)

# save the results of the statistical tests
with open('results_statistical_tests_lineplots.txt', 'w') as f:
    json.dump(testresults, f)


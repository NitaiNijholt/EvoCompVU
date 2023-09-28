'''
Use files generated by data_plot_generate.py to make lineplots
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats

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

# Assymetrical CI:
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

# Symmetrical CI
def lineplot(data1, data2):
    for enemy in data1.keys():
        plt.figure()
        
        for data, label_prefix in zip([data1, data2], ["20 pop ", "10 pop "]):
            gen_numbers = list(range(1, len(data[enemy]) + 1))
            
            best_means = [np.mean(gen['best']) for gen in data[enemy]]
            avg_means = [np.mean(gen['avg']) for gen in data[enemy]]
            avg_stds = [np.std(gen['avg']) for gen in data[enemy]]
            avg_stderr = [stats.sem(gen['avg']) for gen in data[enemy]]
            avg_ci = [1.96 * stderr / np.sqrt(len(data[enemy])) for stderr in avg_stderr] # TODO assumes gaussian distr
            
            plt.plot(gen_numbers, best_means, label=f'{label_prefix}max')
            plt.plot(gen_numbers, avg_means, label=f'{label_prefix}mean')
            
            plt.fill_between(gen_numbers, 
                             [mean - ci for mean, ci in zip(avg_means, avg_ci)], 
                             [mean + ci for mean, ci in zip(avg_means, avg_ci)], 
                             alpha=0.2)
        
        plt.title(f"Enemy {enemy}")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
        plt.show()

# Reading and parsing the data from the files
file1 = 'data_lineplot_fitnesssharing.txt'
file2 = 'data_lineplot_fitnesssharing2.txt'

data1 = extract_data(read_file(file1))
data2 = extract_data(read_file(file2))

# Creating combined plots
lineplot(data1, data2)

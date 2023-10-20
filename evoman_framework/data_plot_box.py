from two_sample_ttest_module import two_sample_ttest


from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd

"""
CHANGE FILENAMES IN `open()` FUNCTIONS
"""
# Read files with champions
with open('data_champion_islanding.txt', 'r') as f:
    fs_champions = json.load(f)
    # Change enemy number and run number to int and champions to np.arrays
    fs_champions = {k: {int(run): np.array(v) for run, v in runs.items()} for k, runs in fs_champions.items()}
with open('data_champion_sharing.txt', 'r') as f:
    isl_champions = json.load(f)
    # Change enemy number and run number to int and champions to np.arrays
    isl_champions = {k: {int(run): np.array(v) for run, v in runs.items()} for k, runs in isl_champions.items()}
# {'Team [...]' : {run_int : weights_array}}  
 

os.environ["SDL_VIDEODRIVER"] = "dummy"
env = Environment(experiment_name='test',
                    enemies=[1,2,3,4,5,6,7,8],
                    playermode="ai",
                    player_controller=player_controller(10), # 10 = n_hidden_neurons
                    enemymode="static",
                    level=2,
                    multiplemode='yes',
                    speed="fastest",
                    visuals=False)

gains_dict = {}

# Per Team, play both algorithms 5 times
i = 0
very_best = 0
for algorithm in fs_champions, isl_champions:
    i+=1
    print(f"EA {[0, 'Fitness Sharing', 'The Island Model'][i]}")
    gains_dict[i] = {}
    average = 0
    best = 0
    best_index = 0

    for team in algorithm.keys():
        print(f"{team}")
        
        gains_dict[i][team] = []

        for run in range(10):

            gain = 0
            current = 0
            for n in range(5):

                f, p, e, t = env.play(pcont=algorithm[team][run])
                gain += (p - e) / 5
                average += f / 5
                current += f / 5
            if current > best:
                best = current
                best_individual = algorithm[team][run]

            
            gains_dict[i][team].append(gain)

        if best > very_best:
            very_best = best
            very_best_individual = best_individual

        print(f"Very best score: {best}")
        print(f"Average over 10 runs: {average / 10}")
        np.savetxt("very_best_individual", very_best_individual)

# Create the boxplots
gains_teamsmall = [gains['Team [1,4,6,7]'] for gains in gains_dict.values()]
gains_teambig = [gains['Team [1,2,3,4,5,6,7,8]'] for gains in gains_dict.values()]

# Update the labels
labels=["Islanding Fitness Genotype", "Islanding Fitness Beaten", "Islanding Fitness Genotype", "Islanding Fitness Beaten"]

# Update the boxplot creation with the correct order
plt.figure(figsize=(12, 8))
bp = plt.boxplot([gains_teamsmall[0], gains_teamsmall[1], gains_teambig[0], gains_teambig[1]], labels=labels)

# plt.title("Combined Boxplot for all Enemies", y=1.08)  # Adjust the title's y position
plt.ylabel('Gain', fontsize=16)
plt.xlabel('EA', fontsize=16)

plt.xticks(rotation=45, fontsize = 14)
plt.tick_params(axis='x', labelsize=14)

# Calculate maximum y value to place algorithm labels just above the highest boxplot
ymax = max([item.get_ydata().max() for item in bp['whiskers']])

# Place the additional labels with increased vertical offset
algorithms = ['Enemy group [1,4,6,7]', 'Enemy group [1,2,3,4,5,6,7,8]']
for i, algorithm in enumerate(algorithms):
    plt.text(2*i + 1.5, ymax + 2.7 * ymax, algorithm, ha='center', va='center', fontsize=16)  # Increased the vertical offset

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the vertical spacing
plt.savefig('boxplots.png')
plt.show()


# TODO
# TODO Statistical test

# Extract data for each boxplot
data_boxplot1 = gains_teamsmall[0]
data_boxplot2 = gains_teamsmall[1]
data_boxplot3 = gains_teambig[0]
data_boxplot4 = gains_teambig[1]

# Compute means for each set of data
mean_boxplot1 = np.mean(data_boxplot1)
mean_boxplot2 = np.mean(data_boxplot2)
mean_boxplot3 = np.mean(data_boxplot3)
mean_boxplot4 = np.mean(data_boxplot4)

# Compute minimum and maximum for each set of data
min_boxplot1 = np.round(np.min(data_boxplot1),2)
max_boxplot1 = np.round(np.max(data_boxplot1),2)
min_boxplot2 = np.round(np.min(data_boxplot2),2)
max_boxplot2 = np.round(np.max(data_boxplot2),2)
min_boxplot3 = np.round(np.min(data_boxplot3),2)
max_boxplot3 = np.round(np.max(data_boxplot3),2)
min_boxplot4 = np.round(np.min(data_boxplot4),2)
max_boxplot4 = np.round(np.max(data_boxplot4),2)

# Compute standard deviation for each set of data
std_boxplot1 = np.std(data_boxplot1)
std_boxplot2 = np.std(data_boxplot2)
std_boxplot3 = np.std(data_boxplot3)
std_boxplot4 = np.std(data_boxplot4)

# Compute mean ± standard deviation
mean_std_boxplot1 = f"{mean_boxplot1:.2f} ± {std_boxplot1:.2f}"
mean_std_boxplot2 = f"{mean_boxplot2:.2f} ± {std_boxplot2:.2f}"
mean_std_boxplot3 = f"{mean_boxplot3:.2f} ± {std_boxplot3:.2f}"
mean_std_boxplot4 = f"{mean_boxplot4:.2f} ± {std_boxplot4:.2f}"


# Conduct t-test for Islanding Fitness Genotype
# Conduct t-test for Team Small Genotype vs Team Small Beaten
print("Team Small Genotype vs Team Small Beaten \n")
t_stat1, p_value1, decision1 = two_sample_ttest(data_boxplot1, data_boxplot2)

print("\nTeam Big Genotype vs Team Big Beaten \n")
t_stat2, p_value2, decision2 = two_sample_ttest(data_boxplot3, data_boxplot4)



testresults = {
    "Islanding Fitness Genotype: Enemies [1,4,6,7] vs Islanding Fitness Genotype: Enemies [1,2,3,4,5,6,7,8]": {
        "t-statistic": t_stat1,
        "p-value": p_value1,
        "decision": decision1
    },
    "Islanding Fitness Beaten: Enemies [1,4,6,7] vs Islanding Fitness Beaten: Enemies [1,2,3,4,5,6,7,8]": {
        "t-statistic": t_stat2,
        "p-value": p_value2,
        "decision": decision2
    }
}

# Save the results to a JSON file
with open('results_statistical_tests_boxplot.txt', 'w') as f:
    json.dump(testresults, f)


# Create a table with the computed values
means_data_updated = {
    'Boxplot': ['Islanding Fitness Genotype: Enemies [1,4,6,7]', 'Islanding Fitness Beaten: Enemies [1,4,6,7]', 'Islanding Fitness Genotype: Enemies [1,2,3,4,5,6,7,8]', 'Islanding Fitness Beaten: Enemies [1,2,3,4,5,6,7,8]'],
    'Mean ± 1 Std': [mean_std_boxplot1, mean_std_boxplot2, mean_std_boxplot3, mean_std_boxplot4],
    'Min': [min_boxplot1, min_boxplot2, min_boxplot3, min_boxplot4],
    'Max': [max_boxplot1, max_boxplot2, max_boxplot3, max_boxplot4]
}
df_means_updated = pd.DataFrame(means_data_updated)

# Print the updated table
print(df_means_updated)

# Save the updated table to a CSV file
df_means_updated.to_csv('means_boxplots_updated.csv', index=False)

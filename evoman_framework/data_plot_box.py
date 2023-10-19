from two_sample_ttest_module import two_sample_ttest


from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt
import numpy as np
import os
import json

"""
CHANGE FILENAMES IN `open()` FUNCTIONS
"""
# Read files with champions
with open('data_champion_gen_island.txt', 'r') as f:
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
for algorithm in fs_champions, isl_champions:
    i+=1

    gains_dict[i] = {}

    for team in algorithm.keys():
        
        gains_dict[i][team] = []

        for run in range(10):

            gain = 0
            for n in range(5):

                f, p, e, t = env.play(pcont=algorithm[team][run])
                gain += (p - e) / 5
            
            gains_dict[i][team].append(gain)

# Create the boxplots
gains_teamsmall = [gains['Team [1,4,6,7]'] for gains in gains_dict.values()]
gains_teambig = [gains['Team [1,2,3,4,5,6,7,8]'] for gains in gains_dict.values()]

# Create the labels
labels=['Team [1,4,6,7]', 'Team [1,2,3,4,5,6,7,8]', 'Team [1,4,6,7]', 'Team [1,2,3,4,5,6,7,8]']

# Create the combined boxplot
plt.figure(figsize=(12, 6))
bp = plt.boxplot([gains_teamsmall[0],gains_teambig[0],gains_teamsmall[1],gains_teambig[1]], labels=labels)
# plt.title("Combined Boxplot for all Enemies", y=1.08)  # Adjust the title's y position
plt.ylabel('Gain')
plt.xlabel('Training Team')
plt.xticks(rotation=45)

# Calculate maximum y value to place algorithm labels just above the highest boxplot
ymax = max([item.get_ydata().max() for item in bp['whiskers']])

# Place the additional labels with increased vertical offset
algorithms = ["Genotype", "Results Based"]
for i, algorithm in enumerate(algorithms):
    plt.text(2*i + 1.5, ymax + 2.7 * ymax, algorithm, ha='center', va='center', fontsize=12)  # Increased the vertical offset

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the vertical spacing
plt.show()


# TODO
# TODO Statistical test
# TODO
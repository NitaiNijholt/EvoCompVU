'''
Here we use the 6 best individuals created in data_plot_generate.py to testfight their
respective enemies 5 times and save that data for plotting.

The score of our champion against his enemy is called individual gain
Individual gain = player life - enemy life
'''
from evoman.environment import Environment
from demo_controller import player_controller
import matplotlib.pyplot as plt
import numpy as np
import os
import json


# Read files with champions
with open('data_champion_fitnesssharing.txt', 'r') as f:
    fs_champions = json.load(f)
    # Change enemy number to int and champions to np.arrays
    fs_champions = {int(k): np.array(v) for k, v in fs_champions.items()}
with open('data_champion_fitnesssharing2.txt', 'r') as f:
    isl_champions = json.load(f)
    # Change enemy number to int and champions to np.arrays
    isl_champions = {int(k): np.array(v) for k, v in isl_champions.items()}

# Initialize a dictionary to store the individual gains for each enemy
individual_gains = {enemy: {'fs': [], 'isl': []} for enemy in isl_champions.keys()}    

# Per enemy, play both algorithms 5 times
os.environ["SDL_VIDEODRIVER"] = "dummy"
for enemy in isl_champions.keys():

    env = Environment(experiment_name='test',
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(10), # 10 = n_hidden_neurons
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    # Play 5 times
    for _ in range(5):

        # Let fitness sharing champion play and find his score
        f, p, e, t = env.play(pcont=fs_champions[enemy])
        individual_gain_fs = p - e
        individual_gains[enemy]['fs'].append(individual_gain_fs)

        # Let islanding champion play and find his score
        f, p, e, t = env.play(pcont=isl_champions[enemy])
        individual_gain_isl = p - e
        individual_gains[enemy]['isl'].append(individual_gain_isl)

# Create the boxplots
for enemy, gains in individual_gains.items():
    plt.figure()
    plt.boxplot([gains['fs'], gains['isl']], labels=['Fitness sharing', 'Islanding'])
    plt.title(f"Boxplot for Enemy {enemy}")
    plt.ylabel('Individual Gain')
    plt.xlabel('Algorithm')
    plt.show()
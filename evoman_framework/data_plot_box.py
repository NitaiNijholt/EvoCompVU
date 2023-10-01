from two_sample_ttest_module import two_sample_ttest

'''
Here we use the  best individuals created in data_plot_generate.py to testfight their
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

"""
CHANGE FILENAMES IN `open()` FUNCTIONS
"""
# Read files with champions
with open('data_champion_fitnessharing_115.txt', 'r') as f:
    fs_champions = json.load(f)
    # Change enemy number and run number to int and champions to np.arrays
    fs_champions = {int(k): {int(run): np.array(v) for run, v in runs.items()} for k, runs in fs_champions.items()}
with open('data_champion_islanding_test678.txt', 'r') as f:
    isl_champions = json.load(f)
    # Change enemy number and run number to int and champions to np.arrays
    isl_champions = {int(k): {int(run): np.array(v) for run, v in runs.items()} for k, runs in isl_champions.items()}

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
    
    # Loop through runs
    """
    ATTENTION: amount of runs of the algorithms has been hard coded!
    """
    for run in range(10):  # Assuming 10 runs were performed in the previous code
        # Play 5 times (or 1 or 2??) Hi TA, if we did this, it was for computation time reasons,
        # since the enemy is static this year it really does not matter.
        """
        CHANGE N=5 OF FIGHTS PER CHAMPION IF NEEDED FOR TIME
        """
        N = 1
        for _ in range(N):
            # Let fitness sharing champion of this run play and find his score
            f, p, e, t = env.play(pcont=fs_champions[enemy][run])
            individual_gain_fs = p - e
            individual_gains[enemy]['fs'].append(individual_gain_fs)

            # Let islanding champion of this run play and find his score
            f, p, e, t = env.play(pcont=isl_champions[enemy][run])
            individual_gain_isl = p - e
            individual_gains[enemy]['isl'].append(individual_gain_isl)



results_statistical_tests = []
# Create the boxplots & perform statistical tests
for enemy, gains in individual_gains.items():
    plt.figure()
    plt.boxplot([gains['fs'], gains['isl']], labels=['Fitness sharing', 'Islanding'])
    plt.title(f"Boxplot for Enemy {enemy}")
    results = two_sample_ttest(gains['fs'], gains['isl'], alpha = 0.05)
    result_dict = {'enemy': enemy, 't_stat': results[0], 'p_value': np.round(results[1], 6) , 'decision': results[2]}
    results_statistical_tests.append(result_dict)
    plt.ylabel('Individual Gain')
    plt.xlabel('Algorithm')
    plt.show()

# save the results of the statistical tests
with open('results_statistical_tests.txt', 'w') as f:
    json.dump(results_statistical_tests, f)


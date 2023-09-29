################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
from evoman.environment import Environment
from demo_controller import player_controller

import json
import numpy as np

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[7],
				  playermode="ai",
				  player_controller=player_controller(10),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)

# Read files with champion
with open('data_champion_fitnesssharing.txt', 'r') as f: 
    fs_champions = json.load(f)
    # Change enemy number to int and champions to np.arrays, important as json makes it a list!!
    champ = np.array(fs_champions['7'])

env.play(champ)

"""
HOW TO FIX THE BUG:

The code presumes that the txt file containing the champion is a dictionary {enemy: champion}
However, it is now a nested dictionary {enemy: {run: champion}}
I now realise I could have fixed this in the time that I wrote this gotta go sry
"""
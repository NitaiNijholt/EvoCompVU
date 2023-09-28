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
                  enemies=[8],
				  playermode="ai",
				  player_controller=player_controller(10),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)

# Read files with champion
with open('data_champion_fitnesssharing.txt', 'r') as f: 
    fs_champions = json.load(f)
    # Change enemy number to int and champions to np.arrays
    champ = np.array(fs_champions['8'])

env.play(champ)


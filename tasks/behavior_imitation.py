# Imitate walk_imitation.py

# Import dependences
from typing import Sequence
import numpy as np

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import bowl
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.tasks import escape
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.tasks import reach


# define class
class BehaviorImitation(Walking):
    """Class for task of mice walking and tracking reference"""
    def __init__(self):
        # self.action_space =
        # self.observation_space =
        # self.state = 
        # self.simulate_length =
        pass

    def step(self, action):
        # self.state = 
        # self.simulate_length -= 1

        # if self.state:
        #     reward = 1
        # else:
        #     reward = -1

        # if self.simulate_length <= 0:
        #     done = True
        # else:
        #     done = False

        # self.state += random.randint(-1, 1)

        # info = {}

        # return self.state, reward, done, info
        
        pass

    def render(self):
        # implement visuilization
        pass

    def reset(self):
        # self.state = 
        # self.simulate_length =
        # return self.state
        pass


 


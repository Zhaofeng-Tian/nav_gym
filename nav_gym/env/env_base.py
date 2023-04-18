import yaml
import numpy as np
import sys

import matplotlib.pyplot as plt

# from ir_sim.world import env_plot, mobile_robot, car_robot, obs_circle, obs_polygon
# from ir_sim.env.env_robot import env_robot
# from ir_sim.env.env_car import env_car
# from ir_sim.env.env_obs_cir import env_obs_cir
# from ir_sim.env.env_obs_line import env_obs_line
# from ir_sim.env.env_obs_poly import env_obs_poly
# from ir_sim.env.env_grid import env_grid
from PIL import Image
from pynput import keyboard

class env_base:
    def __init__(self, world_name=None, plot=True, **kwargs):
        if world_name != None:
            world_name !=sys.path[0]+'/'+world_name
        with open(world_name) as file:
            act_list = yaml.load(file, Loader=yaml.FullLoader)
            self
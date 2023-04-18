import math
import os


class Config:
    """
    Configuration of simulation.
    Obstacles and agents maintainence.
    """
    def __init__(self):
        self.world_size_x = 50
        self.world_size_y = 50
        self.map_reso = 0.1
        # Obstacles initalization
        self.n_cubes = [10,30]
        self.n_circles=[10,30]
        self.cube_len = [1, 5]
        self.circle_r = [1,5]
        # Maze
        self.n_walls = [100,200]
        self.wall_width = 0.5
        self.wall_len = [30, 45]
        self.wall_spacing = 5



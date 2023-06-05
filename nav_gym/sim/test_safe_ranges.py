from nav_gym.obj.robot.robot import CarRobot
from nav_gym.obj.robot.robot_param import CarParam
from nav_gym.sim.config import Config
from nav_gym.map.util import load_map
from nav_gym.obj.geometry.util import rot,line_line, line_polygon
from nav_gym.obj.geometry.objects import Polygon
from nav_gym.sim.plot import plot_cars
import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import random
import time

param = CarParam()
car1 = CarRobot(id = 0,param = param)
print(car1.safe_vertices)
polygon = Polygon(car1.safe_vertices)
ends = car1.og_end_points
print("ends" ,ends)
start = (0.,0.)
print("edges",polygon.edges)
edge = polygon.edges[0]
print("single edge: ", edge)
for i in range(len(ends)):
    line = (start, ends[i])
    print(" ************** Iteration ",i," **********************")
    p =line_polygon(line, polygon)
    print(p)
    if p == None:
        print("P none and ends: ",ends[i])

end1 = np.array([2.29610059, 5.5432772 ])
# for edge in polygon.edges:
# print("check edge: ", edge)
print("line_line check: ",line_line((start, end1), np.array([[0.23,0.45],[-1.,0.45]])))


#     edge is:  [[ 0.23  0.45]
#  [-1.    0.45]]  ray is:  ((0.0, 0.0), array([5.5432772 , 2.29610059]))
# intersection is:  None
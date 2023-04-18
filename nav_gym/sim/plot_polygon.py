import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
from nav_gym.obj.robot.robot import CarRobot
from nav_gym.obj.robot.robot_param import CarParam
import numpy as np

def generate_polygon(n_edges_range, center_range, area_range):
    while True:
        # Generate random number of edges, center, and radius
        n_edges = random.randint(*n_edges_range)
        center_x = random.uniform(*center_range[0])
        center_y = random.uniform(*center_range[1])
        center = (center_x, center_y)
        radius = random.uniform(1, min(center))

        # Generate vertices
        angles = sorted(random.uniform(0, 2 * 3.1415926) for _ in range(n_edges))
        vertices = [(center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)) for angle in angles]

        # Calculate area and check if it is within range
        area = 0.5 * sum(x0 * y1 - x1 * y0 for ((x0, y0), (x1, y1)) in zip(vertices, vertices[1:] + [vertices[0]]))
        if area <= area_range[1] and area >= area_range[0]:
            return Polygon(vertices)
        
def plot_polygon(polygon):
    fig, ax = plt.subplots()
    ax.add_patch(polygon)
    ax.set_xlim(-10, 60)
    ax.set_ylim(-10, 60)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# polygon = generate_polygon((3,7), ((0,50),(0,50)), (10,30))
# plot_polygon(polygon)
param = CarParam()
car = CarRobot(1, param, np.array([25.,25.,0.,0.,0.]))
polygon_car = Polygon(car.vertices[:4])
plot_polygon(polygon_car)

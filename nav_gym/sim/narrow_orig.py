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

def narrow_sim():
    """
    Static Polygon, Circles represent static obstacles, will be draw on the map.
    Dynamic Polygons, Circles represent cars and round robots, the geometry of them will be maintained dynamically.
    """
    world_x = 50
    world_y = 50
    world_reso = 0.01
    car_param = CarParam("normal")
    print("car param, ",car_param)
    static_polygons =[]
    static_circles = []
    dynamic_polygons = []
    dynamic_circles = []
    config = Config()
    n_cars = 20
    cars = []
    polygons = []
    center_x = 25.
    center_y = 25.
    r = 10.
    plot = True
    plot_lidar = True
    n_circles = random.randint(config.n_circles[0], config.n_circles[1])
    n_cubes = random.randint(config.n_cubes[0], config.n_cubes[1])
    circles = []



    for i in range(n_cars):
        cars.append(CarRobot(id = i,
                            param= car_param, 
                            initial_state = np.array([center_x +r*cos(i*2*pi/n_cars), center_y+r*sin(i*2*pi/n_cars), i*2*pi/n_cars+pi, 0., 0. ])))
        polygons.append(Polygon(cars[i].vertices[:4]))

    # cars.append(CarRobot(id=1, param=car_param, initial_state=np.array([40., 24.,3.15, 0.0, 0.0])))
    img = load_map("racetrack.png")

    map = 1 - img
    # fig, ax = plt.subplots()
    # ax.imshow(map, origin = 'lower',cmap='gray',extent=[0,world_x,0,world_y])  
    # plt.show()
    num_r = round(world_y/world_reso)
    num_c = round(world_x/world_reso)
    agt_map = np.zeros( (num_r, num_c) )
    ocp_map = map + agt_map

    # ax.imshow(ocp_map, origin = 'lower',cmap='gray',extent=[0,world_x,0,world_y])  
    # plt.show()

    if plot == True:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    """
    Big Loop to update states and observations.
    """
    for i in range(30):
        start_time = time.time()
        polygons = []
        agt_map = np.zeros( (num_r, num_c ) ).astype(np.float32)
        # *************** States Update Loop Starts ***************** #
        for car in cars:
            
            car.update(np.array([1.,0.2]))
            
            polygons.append(Polygon(car.vertices[:4]))
            car.id_fill_body(agt_map)
            # print(agt_map)
            # print((agt_map==0).all())
            # plt.imshow(agt_map, origin = 'lower',cmap='gray',extent=[0,world_x,0,world_y])
            # plt.show()
        # print("where >= 0: ",np.where(agt_map>0.1))
        # *************** States Update Loop Ends *********************
        # plt.imshow(agt_map, origin = 'lower',cmap='gray',extent=[0,world_x,0,world_y])
        # plt.show()
        ocp_map = map + agt_map
        # plt.imshow(agt_map, origin = 'lower',cmap='gray',extent=[0,world_x,0,world_y])
        # plt.show()
        
        # plt.imshow(ocp_map, origin = 'lower',cmap='gray',extent=[0,world_x,0,world_y])
        # plt.draw()
        # plt.pause(0.02)
        # *************** Observation Update Loop Starts **************
        for car in cars:
            temp_polygons = polygons.copy()
            temp_polygons.pop(car.id)
            # car.sensor_update(ocp_map,temp_polygons,circles)
            # car.map_based_sensor_update(ocp_map)
            car.map_id_sensor_update(ocp_map)
        end_time1 = time.time()
        print("*********** time cost w/o plot : ", end_time1-start_time)
        # *************** Observation Update Loop Ends **************

        if plot==True:
            plt.cla()
            ax.imshow(img, origin = 'lower',cmap='gray',extent=[0,50,0,50])
            # ax.imshow(ocp_map, origin = 'lower',cmap='gray',extent=[0,50,0,50])
            if plot_lidar:  
                for car in cars:
                    for end in car.points:
                        start = car.vertices[4].copy()
                        x = [start[0], end[0]]
                        y = [start[1], end[1]]
                        ax.plot(x,y,color='blue')
                # print(" Range observation: ", car.ranges)
            plot_cars(ax, cars)
            plt.draw()
            plt.pause(0.02)
            # plt.show()

        end_time2 = time.time()
        print("*********** time cost with plot : ", end_time2-start_time)
        # plt.show()

narrow_sim()
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

    config = Config()
    n_cars = 1
    cars = []
    polygons = []
    plot = True
    plot_lidar = True
    num_steps = 1




    for i in range(n_cars):
        cars.append(CarRobot(id = i,
                            param= car_param, 
                            initial_state = np.array([27,5,pi/2,0,0])))
        polygons.append(Polygon(cars[i].vertices[:4]))

    print("safe ranges: ",cars[0].safe_ranges)
    print("safe points: ",cars[0].safe_points)
    
    img = load_map("racetrack.png")

    map = 1 - img

    num_r = round(world_y/world_reso)
    num_c = round(world_x/world_reso)
    agt_map = np.zeros( (num_r, num_c) )
    ocp_map = map + agt_map
    for car in cars:
        car.map_id_sensor_update(ocp_map)
        print(" car ranges: ", car.ranges)


    if plot == True:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    """
    Big Loop to update states and observations.
    """
    for i in range(num_steps):
        start_time = time.time()
        polygons = []
        agt_map = np.zeros( (num_r, num_c ) ).astype(np.float32)
        # *************** States Update Loop Starts ***************** #
        for car in cars:
            
            car.update(np.array([1.,0.2]))
            
            polygons.append(Polygon(car.vertices[:4]))
            car.id_fill_body(agt_map)
            
        ocp_map = map + agt_map

        # *************** Observation Update Loop Starts **************
        for car in cars:
            temp_polygons = polygons.copy()
            temp_polygons.pop(car.id)
            car.map_id_sensor_update(ocp_map)
            print(car.ranges)
            print("range - safe range: ",np.array(car.ranges) - np.array(car.safe_ranges))
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
            # plt.draw()
            # plt.pause(0.02)
            plt.show()

        end_time2 = time.time()
        print("*********** time cost with plot : ", end_time2-start_time)
        # plt.show()

narrow_sim()
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
# import gymnasium as gym
import gym
from gym import spaces

"""
I.  Observation:
Lidar     (n, 32)
Goal      (n,  2)
Velocity  (n,  2)
"""

class NarrowSpaceEnv_v0(gym.Env):

    def __init__(self,render_mode = "human"):
        self.world_x = 50
        self.world_y = 50
        self.world_reso = 0.01
        self.carparam = CarParam()
        self.n_cars = 10
        self.cars = []
        self.polygons = []
        self.circles = []
        self.plot_pause_time = 0.02
        if render_mode == "human":
            self.plot = True
        else:
            self.plot = False
        # self.plot = False
        self.plot_lidar = True
        self.max_episode_steps = 1000
        self.num_episodes = 1000
        # self.initial_states = [np.array([1.5,2.5,0.,0.,0.]),np.array([8.5,2.5, pi, 0.,0.])]
        r=11.
        self.initial_states = [np.array([25 +r*cos(i*2*pi/self.n_cars), 25+r*sin(i*2*pi/self.n_cars), i*2*pi/self.n_cars+pi, 0., 0. ]) for i in range(self.n_cars)]
        print("*************** initial states: ", self.initial_states)
        # self.initial_states = [np.array([27,5,pi/2,0,0])]
        # self.global_goals = [np.array([8.0,2.5, pi, 0.,0.]), np.array([2.0,2.5,0.,0.,0.])]
        self.global_goals = self.initial_states[::-1]
        print("*************** goals: ", self.global_goals)
        # self.observation_space = spaces.Dict(
        #     {
        #     "lidar":spaces.Box(0.1, 6.0,shape=(self.n_cars, self.carparam.ray_num,),dtype=np.float32),
        #     "goal":spaces.Box(0.0, 2.0, shape=(self.n_cars,2),dtype=np.float32),
        #     "v": spaces.Box(self.carparam.v_limits[0],self.carparam.v_limits[1],shape=(2,),dtype=np.float32)
        #     }
        # )
        self.observation_space = spaces.Box(low = 0.1, high = 6.0,shape = (34,) , dtype=np.float32)
        self.action_space = spaces.Box(self.carparam.v_limits[1],self.carparam.v_limits[0],shape=(2,),dtype=np.float32)
        self.img = load_map("racetrack.png")
        # map = self.img.copy()
        self.img[np.where(self.img>0.)] = 1.0
        self.map = 1 - self.img
        self.num_r = round(self.world_y/self.world_reso)
        self.num_c = round(self.world_x/self.world_reso)
        self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
        # self.ocp_map = self.map + self.agt_map
        for i in range(self.n_cars):
            self.cars.append(CarRobot(id = i,
                        param= self.carparam, 
                        initial_state = self.initial_states[i],
                        global_goal= self.global_goals[i]))
            self.cars[i].id_fill_body(self.agt_map)
            self.polygons.append(Polygon(self.cars[i].vertices[:4]))
        self.ocp_map = self.map + self.agt_map
        
        for i in range(self.n_cars):
            self.cars[i].state_init(self.initial_states[i])
            self.cars[i].obs_init(self.ocp_map,self.polygons,self.circles)

        if self.plot == True:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")


    def reset(self):
        # clean checklist 1. agent map 2. obstacle list
        self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
        self.polygons = []
        for car in self.cars:
            car.state_init(self.initial_states[car.id])
            car.id_fill_body(self.agt_map)
            self.polygons.append(Polygon(car.vertices[:4]))
        self.ocp_map = self.map + self.agt_map
        for car in self.cars:
            car.obs_init(self.ocp_map,self.polygons,self.circles)
        return self._get_obs()

    def step(self,actions): # action shape [n_agents, action(v, phi)] (n,2) array
        # clean checklist 1. agent map 2. obstacle list
        self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
        self.polygons = []
        # states update loop
        for car in self.cars:
            if car.done:
                print("Use reset function~~~~~~~~~~~~~~~~")
                car.state_init(self.initial_states[car.id])
            else:
                car.state_update(actions[car.id])
            car.id_fill_body(self.agt_map)
            self.polygons.append(Polygon(car.vertices[:4]))
        self.ocp_map = self.map + self.agt_map
        # observation, done, and reward update loop
        for car in self.cars:
            car.obs_update(self.ocp_map,self.polygons,self.circles)
        
        # for car in self.cars:
        #     print("*******************step: ", " *********************")
        #     print("Car ID: ", car.id)
        #     print("lidar obs: ",car.lidar_obs)
        #     print("car state: ",car.state)
        #     print("car local goal: ", car.local_goal)
        #     print("collision: ",car.collision)
        #     print("done: ", car.done )
        #     print("reward: ", car.reward)
        #     print("reward info: ", car.reward_info)
        
        # render



        if self.plot:
            self.render_frame()
        # get data
        obs_ = self._get_obs()
        done = self._get_done()
        reward = self._get_reward()
        truncated = None
        info = {}
        return obs_,  reward, done, truncated, info
        
        

    def _get_reward(self):
        reward_list=[]
        for car in self.cars:
            reward_list.append(car.reward)
        return np.array(reward_list, dtype=np.float32)

    def _get_done(self):
        done_list = []
        for car in self.cars:
            done_list.append(car.done)
        return np.array(done_list,dtype=np.int32)


    def _get_obs(self):
        lidar_obs_list = []
        # goal_obs_list =[]
        # v_obs_list = []
        for car in self.cars:
            obs = car._get_lidar_obs()
            g = car._get_goal_obs()
            obs = np.concatenate((obs,g))
            lidar_obs_list.append(obs)

        return np.array(lidar_obs_list,dtype=np.float32)

    def render_frame(self):
            plt.cla()
            self.ax.imshow(self.img, origin = 'lower',cmap='gray',extent=[0,self.world_x,0,self.world_y])
            # ax.imshow(ocp_map, origin = 'lower',cmap='gray',extent=[0,50,0,50])
            if self.plot_lidar:  
                for car in self.cars:
                    for end in car.points:
                        start = car.vertices[4].copy()
                        x = [start[0], end[0]]
                        y = [start[1], end[1]]
                        self.ax.plot(x,y,color='blue')
                # print(" Range observation: ", car.ranges)
            plot_cars(self.ax, self.cars)
            plt.draw()
            # plt.pause(self.plot_pause_time)
            plt.show()

# class NarrowSpaceEnv_v0(gym.Env):

#     def __init__(self,render_mode = "human"):

#         self.world_x = 50
#         self.world_y = 50
#         self.world_reso = 0.01
#         self.carparam = CarParam()
#         self.n_cars = 1
#         self.cars = []
#         self.polygons = []
#         self.circles = []
#         if render_mode == "human":
#             self.plot = True
#         else:
#             self.plot = False
#         self.plot_pause_time = 2
#         self.plot_lidar = True
#         self.max_episode_steps = 1000
#         self.num_episodes = 1000
#         self.initial_states = [np.array([27,5,pi/2,0,0])]
#         self.observation_space = spaces.Dict(
#             {
#             "lidar":spaces.Box(0.1, 6.0,shape=(self.n_cars, self.carparam.ray_num,),dtype=np.float32),
#             "goal":spaces.Box(0.0, 2.0, shape=(self.n_cars,2),dtype=np.float32),
#             "v": spaces.Box(self.carparam.v_limits[0],self.carparam.v_limits[1],shape=(2,),dtype=np.float32)
#             }
#         )
#         self.action_space = spaces.Box(self.carparam.v_limits[1],self.carparam.v_limits[0],shape=(2,),dtype=np.float32)
#         self.img = load_map("racetrack.png")
#         self.map = 1 - self.img
#         self.num_r = round(self.world_y/self.world_reso)
#         self.num_c = round(self.world_x/self.world_reso)
#         self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
#         # self.ocp_map = self.map + self.agt_map
#         for i in range(self.n_cars):
#             self.cars.append(CarRobot(id = i,
#                         param= self.carparam, 
#                         initial_state = self.initial_states[i]))
#             self.cars[i].id_fill_body(self.agt_map)
#             self.polygons.append(Polygon(self.cars[i].vertices[:4]))
#         self.ocp_map = self.map + self.agt_map
        
#         for i in range(self.n_cars):
#             self.cars[i].state_init(self.initial_states[i])
#             self.cars[i].obs_init(self.ocp_map,self.polygons,self.circles)

#         if self.plot == True:
#             self.fig, self.ax = plt.subplots()
#             self.ax.set_aspect('equal')
#             self.ax.set_xlabel("x [m]")
#             self.ax.set_ylabel("y [m]")


#     def reset(self):
#         # clean checklist 1. agent map 2. obstacle list
#         self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
#         self.polygons = []
#         for car in self.cars:
#             car.state_init(self.initial_states[car.id])
#             car.id_fill_body(self.agt_map)
#             self.polygons.append(Polygon(car.vertices[:4]))
#         self.ocp_map = self.map + self.agt_map
#         for car in self.cars:
#             car.obs_init(self.ocp_map,self.polygons,self.circles)
#         return self._get_obs()

#     def step(self,actions): # action shape [n_agents, action(v, phi)] (n,2) array
#         # clean checklist 1. agent map 2. obstacle list
#         self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
#         self.polygons = []
#         # states update loop
#         for car in self.cars:
#             if car.done:
#                 print("Use reset function~~~~~~~~~~~~~~~~")
#                 car.state_init(self.initial_states[car.id])
#             else:
#                 car.state_update(actions[car.id])
#             car.id_fill_body(self.agt_map)
#             self.polygons.append(Polygon(car.vertices[:4]))
#         self.ocp_map = self.map + self.agt_map
#         # observation, done, and reward update loop
#         for car in self.cars:
#             car.obs_update(self.ocp_map,self.polygons,self.circles)
        
#         for car in self.cars:
#             print("*******************step: ", " *********************")
#             print("Car ID: ", car.id)
#             print("lidar obs: ",car.lidar_obs)
#             print("car state: ",car.state)
#             print("car local goal: ", car.local_goal)
#             print("collision: ",car.collision)
#             print("done: ", car.done )
#             print("reward: ", car.reward)
#             print("reward info: ", car.reward_info)
        
#         # render



#         if self.plot:
#             self.render_frame()
#         # get data
#         obs_ = self._get_obs()
#         done = self._get_done()
#         reward = self._get_reward()
#         info = None
#         return obs_,  reward, done, info
        
        

#     def _get_reward(self):
#         reward_list=[]
#         for car in self.cars:
#             reward_list.append(car.reward)
#         return np.array(reward_list, dtype=np.float32)

#     def _get_done(self):
#         done_list = []
#         for car in self.cars:
#             done_list.append(car.done)
#         return np.array(done_list,dtype=np.int32)


#     def _get_obs(self):
#         lidar_obs_list = []
#         goal_obs_list =[]
#         v_obs_list = []

#         for car in self.cars:
#             lidar_obs_list.append(car._get_lidar_obs())
#             goal_obs_list.append(car._get_goal_obs())
#             v_obs_list.append(car._get_v_obs())
#         return {
#             "lidar": np.array(lidar_obs_list,dtype=np.float32),
#             "goal": np.array(goal_obs_list,dtype=np.float32),
#             "v":np.array(v_obs_list,dtype=np.float32),
#         }

#     def render_frame(self):
#             plt.cla()
#             self.ax.imshow(self.img, origin = 'lower',cmap='gray',extent=[0,50,0,50])
#             # ax.imshow(ocp_map, origin = 'lower',cmap='gray',extent=[0,50,0,50])
#             if self.plot_lidar:  
#                 for car in self.cars:
#                     for end in car.points:
#                         start = car.vertices[4].copy()
#                         x = [start[0], end[0]]
#                         y = [start[1], end[1]]
#                         self.ax.plot(x,y,color='blue')
#                 # print(" Range observation: ", car.ranges)
#             plot_cars(self.ax, self.cars)
#             plt.draw()
#             plt.pause(self.plot_pause_time)
#             # plt.show()

class NarrowSpaceEnv_v1(gym.Env):
    """
    Simple observation:
    Lidar + V: (34,)
    """

    def __init__(self,render_mode = "human"):
        self.world_x = 50
        self.world_y = 50
        self.world_reso = 0.01
        self.carparam = CarParam()
        self.n_cars = 1
        self.cars = []
        self.polygons = []
        self.circles = []
        if render_mode == "human":
            self.plot = True
        else:
            self.plot = False
        self.plot = False
        self.plot_lidar = False
        self.max_episode_steps = 1000
        self.num_episodes = 1000
        self.initial_states = [np.array([25,5,pi/2,0,0])]
        self.observation_space = spaces.Box(low = 0.1, high = 6.0,shape = (34,) , dtype=np.float32)
        # self.observation_space = spaces.Dict(
        #     {
        #     "lidar":spaces.Box(0.1, 6.0,shape=(self.n_cars, self.carparam.ray_num,),dtype=np.float32),
        #     "goal":spaces.Box(0.0, 2.0, shape=(self.n_cars,2),dtype=np.float32),
        #     "v": spaces.Box(self.carparam.v_limits[0],self.carparam.v_limits[1],shape=(2,),dtype=np.float32)
        #     }
        # )

        self.action_space = spaces.Box(self.carparam.v_limits[1],self.carparam.v_limits[0],shape=(2,),dtype=np.float32)
        self.img = load_map("racetrack.png")
        self.map = 1 - self.img
        self.num_r = round(self.world_y/self.world_reso)
        self.num_c = round(self.world_x/self.world_reso)
        self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
        # self.ocp_map = self.map + self.agt_map
        for i in range(self.n_cars):
            self.cars.append(CarRobot(id = i,
                        param= self.carparam, 
                        initial_state = self.initial_states[i]))
            self.cars[i].id_fill_body(self.agt_map)
            self.polygons.append(Polygon(self.cars[i].vertices[:4]))
        self.ocp_map = self.map + self.agt_map
        
        for i in range(self.n_cars):
            self.cars[i].state_init(self.initial_states[i])
            self.cars[i].obs_init(self.ocp_map,self.polygons,self.circles)

        if self.plot == True:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")


    def reset(self):
        # clean checklist 1. agent map 2. obstacle list
        self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
        self.polygons = []
        for car in self.cars:
            car.state_init(self.initial_states[car.id])
            car.id_fill_body(self.agt_map)
            self.polygons.append(Polygon(car.vertices[:4]))
        self.ocp_map = self.map + self.agt_map
        for car in self.cars:
            car.obs_init(self.ocp_map,self.polygons,self.circles)
            obs = self._get_obs()
            # print("In env check obs for reset: ",obs)

        return obs

    def step(self,actions): # action shape [n_agents, action(v, phi)] (n,2) array
        # clean checklist 1. agent map 2. obstacle list
        self.agt_map = np.zeros( (self.num_r, self.num_c) ).astype(np.float32)
        self.polygons = []
        # states update loop
        for car in self.cars:
            if car.done:
                print("Use reset function~~~~~~~~~~~~~~~~")
                car.state_init(self.initial_states[car.id])
            else:
                car.state_update(actions[car.id])
            car.id_fill_body(self.agt_map)
            self.polygons.append(Polygon(car.vertices[:4]))
        self.ocp_map = self.map + self.agt_map
        # observation, done, and reward update loop
        for car in self.cars:
            car.obs_update(self.ocp_map,self.polygons,self.circles)
        
        # for car in self.cars:
            # print("*******************step: ", " *********************")
            # print("Car ID: ", car.id)
            # print("lidar obs: ",car.lidar_obs)
            # print("car state: ",car.state)
            # print("car local goal: ", car.local_goal)
            # print("collision: ",car.collision)
            # print("done: ", car.done )
            # print("reward: ", car.reward)
            # print("reward info: ", car.reward_info)
        
        # render



        if self.plot:
            self.render_frame()
        # get data
        obs_ = self._get_obs()
        done = self._get_done()
        reward = self._get_reward()
        truncated = None
        info = {}
        return obs_,  reward, done, truncated, info
        
        

    def _get_reward(self):
        reward_list=[]
        for car in self.cars:
            reward_list.append(car.reward)
        return np.array(reward_list, dtype=np.float32)

    def _get_done(self):
        done_list = []
        for car in self.cars:
            done_list.append(car.done)
        return np.array(done_list,dtype=np.int32)


    def _get_obs(self):
        lidar_obs_list = []
        # goal_obs_list =[]
        # v_obs_list = []

        for car in self.cars:
            obs = car._get_lidar_obs()
            v = car._get_v_obs()
            obs = np.concatenate((obs,v))
            lidar_obs_list.append(obs)
            # print("obs in get_obs: ", lidar_obs_list)
            # print(" the to array: ",np.array(lidar_obs_list,dtype=np.float32))
        return np.array(lidar_obs_list,dtype=np.float32)


    def render_frame(self):
            plt.cla()
            self.ax.imshow(self.img, origin = 'lower',cmap='gray',extent=[0,50,0,50])
            # ax.imshow(ocp_map, origin = 'lower',cmap='gray',extent=[0,50,0,50])
            if self.plot_lidar:  
                for car in self.cars:
                    for end in car.points:
                        start = car.vertices[4].copy()
                        x = [start[0], end[0]]
                        y = [start[1], end[1]]
                        self.ax.plot(x,y,color='blue')
                # print(" Range observation: ", car.ranges)
            plot_cars(self.ax, self.cars)
            plt.draw()
            plt.pause(0.2)


# env = NarrowSpaceEnv()
# env.reset()
# for i in range(50):
#     obs_, r, done, _ = env.step([np.array([1.0,0.0])])
#     for car in env.cars:
#         print("*******************step: ", i, " *********************")
#         print("Car ID: ", car.id)
#         print("lidar obs: ",car.lidar_obs)
#         print("car state: ",car.state)
#         print("car local goal: ", car.local_goal)
#         print("collision: ",car.collision)
#         print("done: ", car.done )
#         print("reward: ", car.reward)
#     print(" The env obs: ", obs_)
#     print(" env reward: ",r)
#     print(" env done: ", done)
# print(env.observation_space)
# print(env.action_space)
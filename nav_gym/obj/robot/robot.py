import numpy as np
from math import pi,cos,sin,tan, log10, atan2
from nav_gym.obj.geometry.util import rot, topi
from nav_gym.obj.geometry.lidar import *
from nav_gym.obj.robot.robot_param import CarParam
import os
from skimage import io
from skimage.draw import polygon
from math import pi

class CarRobot:
    def __init__(self,id, param,
                 initial_state = np.array([0.,0.,0.,0.,0.,]), 
                 global_goal = [],
                 history_len = 5,dt=0.2):
        self.id = id
        self.value_base = param.value_base # base value of car robots
        self.dv = param.dv # delta value, the id value difference between two id-neighboring robots
        self.id_value = self.calc_id_value() # id value to differentiate robots
        self.world_reso = param.world_reso # world map resolution
        self.state = initial_state.copy() #[x, y, theta, v, phi]
        self.history_len = history_len
        self.history = np.tile(initial_state,(history_len,1)).reshape((history_len,len(initial_state))) # history with set length
        self.shape = param.shape # wheelbase, front, rear suspension, half width, lidar to center(defaut mount on the front)
        self.geo_r = param.geo_r # geometric radius considered as a big disk
        self.disk_r = param.disk_r # smaller circle considered as two or more disks
        self.disk_num = param.disk_num # number of small disks to cover body
        self.og_disk_centers = param.disk_centers
        self.disk_centers = (rot(self.state[2])@self.og_disk_centers.T).T + self.state[:2]
        self.og_vertices = param.vertices # original vertice coords w.r.t. car center at (0,0), and yaw at 0.
        self.safe_vertices = param.safe_vertices
        self.vertices = (rot(self.state[2])@self.og_vertices.T).T + self.state[:2]
        self.v_limits = param.v_limits
        self.a_limits = param.a_limits
        self.dt = dt
        self.action = None
        # Lidar param
        self.fan_range = param.fan_range
        self.min_range = param.min_range
        self.max_range = param.max_range
        self.ray_num = param.ray_num
        # Lidar observations
        self.angles = generate_angles(self.fan_range, self.fan_range, self.ray_num, reso = 0.098, is_num_based = True)
        self.og_end_points = generate_ends(self.angles,self.max_range)
        self.end_points = ends_tf(self.og_end_points, self.state[2],self.vertices[4])
        self.ranges = None
        self.points = None
        self.lidar_obs = None
        self.lidar_obs_history = None
        self.collision = False
        
        self.safe_ranges, self.safe_points = generate_safe_range_points(start=(0.,0.),
                                                      ends = self.og_end_points,
                                                      safe_vertices = self.safe_vertices)
        # Local goal
        self.look_ahead = param.look_ahead_dist
        self.global_goal = global_goal # global goal
        self.local_goal = None # local goal
        self.local_goal_history = None
        self.achieve = False
        self.done = False
        # print("Safe ranges: ", len(self.safe_ranges))
        self.reward = None
        self.achieve_tolerance = param.achieve_tolerance
        self.reward_info = None

    @property
    def A(self,): # state transition model
        x, y, theta, v, phi = self.state
        l = self.shape[0] # wheel_base
        return np.array([cos(theta)*self.dt,sin(theta)*self.dt,self.dt*tan(phi)/l]) 
    
    def _get_obs(self):
        return {"lidar": self.lidar_obs,
                "goal": self.local_goal,
                "v":self.state[3:].astype(np.float32)} # v here denotes [v, phi], i.e., linear speed and steer angle
    
    def _get_lidar_obs(self):
        return self.lidar_obs
    
    def _get_goal_obs(self):
        return self.local_goal
    
    def _get_v_obs(self):
        return self.state[3:].astype(np.float32)
    
    # def reset(self, reset_state,map,polygons, circles):
    #     self.state_init(reset_state)
    #     self.obs_init(map,polygons,circles)




    def state_init(self,state):
        self.state = state.copy()
        self.vertices = (rot(self.state[2])@self.og_vertices.T).T + self.state[:2]
        self.disk_centers = (rot(self.state[2])@self.og_disk_centers.T).T + self.state[:2]
        self.history = np.delete(self.history, 0 , axis=0)
        self.history = np.insert(self.history, len(self.history), self.state, axis=0)
        self.end_points = ends_tf(self.og_end_points, self.state[2],self.vertices[4])


    
    def state_update(self, cmd ):
        # 1. velocity state update 
        self.action = cmd
        self.state[3:] = self.move_base(cmd)
        # 2. x, y, theta pose state update
        self.state[:3] += self.A *self.state[3]
        # print("In sate update before topi: ", self.state)
        self.state[2] = topi(self.state[2])
        # print("after to pi: ", self.state)
        # 3. vertices update
        self.vertices = (rot(self.state[2])@self.og_vertices.T).T + self.state[:2]
        # 4. disk centers update
        self.disk_centers = (rot(self.state[2])@self.og_disk_centers.T).T + self.state[:2]
        # 5. history 
        self.history = np.delete(self.history, 0 , axis=0)
        self.history = np.insert(self.history, len(self.history), self.state, axis=0)
        # 6. ends
        # self.end_points = ends_tf(self.og_end_points, self.state[2],self.state[:2])
        self.end_points = ends_tf(self.og_end_points, self.state[2],self.vertices[4])

    def obs_init(self, map,polygons, circles):
        self.map_id_sensor_update(map)
        self.lidar_obs = np.array(self.ranges)-np.array(self.safe_ranges)
        self.local_goal = self.calc_local_goal()
        self.lidar_obs_history =  np.tile(self.lidar_obs,(self.history_len,1)).reshape((self.history_len,len(self.lidar_obs)))
        self.local_goal_history =  np.tile(self.local_goal,(self.history_len,1)).reshape((self.history_len,len(self.local_goal)))
        self.done = self.check_done()

    def obs_update(self, map,polygons, circles):
        """
        Obs update check: 1. Lidar Y; 2.lidar obs Y; 3. local goal Y 4. obs history Y.
        """
        self.map_id_sensor_update(map)
        self.lidar_obs = np.array(self.ranges)-np.array(self.safe_ranges)
        self.local_goal = self.calc_local_goal()
        self.lidar_obs_history = np.delete(self.lidar_obs_history, 0 , axis=0)
        self.lidar_obs_history = np.insert(self.lidar_obs_history, len(self.lidar_obs_history), self.lidar_obs, axis=0)
        self.local_goal_history = np.delete(self.local_goal_history, 0 , axis=0)
        self.local_goal_history = np.insert(self.local_goal_history, len(self.local_goal_history), self.local_goal, axis=0)
        self.done = self.check_done()
        # self.reward = self.calc_open_space_reward()
        self.reward = self.calc_reward()
    # Auxilary functions
    def move_base(self, cmd):
        v = np.array([0.,0.]) # actually executed v, which is bounded by v and a limits
        if cmd[0] >= self.state[3]: # cmd > v: accelerate demand
            v[0] = min(self.state[3]+self.a_limits[0,0]*self.dt, self.v_limits[0,0], cmd[0])
        elif cmd[0] < self.state[3]: # deaccelerate demand
            v[0] = max(self.state[3]+ self.a_limits[1,0]*self.dt, self.v_limits[1,0], cmd[0])
        if cmd[1] >= self.state[4]: #  turn left demand
            v[1] = min(self.state[4]+self.a_limits[0,1]*self.dt, self.v_limits[0,1], cmd[1])
        elif cmd[1] < self.state[4]: # deaccelerate demand
            v[1] = max(self.state[4]+ self.a_limits[1,1]*self.dt, self.v_limits[1,1], cmd[1])
        return v


    def sensor_update(self, map, polygons, circles):
        """
        Sensor updates after state updates to sychronize agents in the environment
        """
        # 5. Lidar related update
        no_ego_map = map.copy()
        self.remove_body(no_ego_map)
        self.ranges, self.points = generate_range_points(start=(self.vertices[4,0],self.vertices[4,1]),
                                                         ends=self.end_points,
                                                         map=no_ego_map, polygons=polygons, circles=circles, max_range=self.max_range)

    def map_based_sensor_update(self, map):
        no_ego_map = map.copy()
        self.remove_body(no_ego_map)
        self.ranges, self.points = map_based_generate_range_points(start=(self.vertices[4,0],self.vertices[4,1]),
                                                         ends=self.end_points,
                                                         map = no_ego_map)
    def map_id_sensor_update(self,map):
        self.ranges, self.points = map_id_generate_range_points(start=(self.vertices[4,0],self.vertices[4,1]),
                                                         ends=self.end_points, map = map,
                                                         id_value = self.id_value, dv = self.dv)
    def calc_local_goal(self):
        if len(self.global_goal) > 0:
            dx, dy =  self.global_goal[:2] - self.state[:2]
            angle = atan2(dy, dx)
            d_angle = topi(angle - self.state[2])
            dist = sqrt(dx**2 + dy**2)
            return np.array([dist, d_angle],dtype=np.float32)

        else:
            return self.calc_open_space_goal()



    def calc_open_space_goal(self): 
        """
        This is a goal update method based on the longest lidar distance,
        to encourage robot navigate to open space, used when no map available.
        """
        max_range = 0.0
        angle = 0.0
        for i in range(len(self.ranges)):
            if self.angles[i] <= pi/2 or self.angles[i] >= 3*pi/2:
                continue    
            if self.ranges[i] > max_range:
                max_range = self.ranges[i] # the range is from lidar center to the lidar point
                angle = self.angles[i]
        angle = topi(angle)
        if max_range > self.look_ahead:
            max_range = self.look_ahead
        return np.array([max_range,angle],dtype=np.float32)
    
    def calc_reward(self):
        reward = 0.
        if self.collision:
            reward = -50.
        else:
            if self.achieve:
                reward = 50.
            else: 
                if len(self.global_goal) > 0 :
                    # print(" Using goal reward!")
                    reward = self.calc_goal_reward()
                else:
                    # print(" Using open space reward!")
                    reward = self.calc_open_space_reward()
        return reward

    def calc_goal_reward(self):
        alpha = 10.
        previous_goal, current_goal = self.local_goal_history[-2:]
        # print(" previous_goal: ", previous_goal)
        # print("current goal: ", current_goal)
        d_dist, d_angle = previous_goal - current_goal
        # print( "d_dist and d_angle: ", d_dist, " ", d_angle)
        return alpha*(d_dist + d_angle)


    
    def calc_open_space_reward(self):
        """
        This is a shaped reward function without waypoints guidance,
        motivating robot to navigate towards open space while avoiding
        obstacles. The reward functions is well-tuned and valided in
        our previous paper. 
        """
        # print(" ************** calc reward called!")
        reward = 0
        if not self.collision:
            reward1 = 0.; reward2 = 0.; reward3 = 0.; reward4 = 0.
            # *************** Part I Forward ***************
            decay1 = 0.9
            reward1 += self.state[3]*(decay1*self.lidar_obs[0] 
                                    +decay1**2*(self.lidar_obs[1]+self.lidar_obs[self.ray_num - 1])
                                    +decay1**3*(self.lidar_obs[2]+self.lidar_obs[self.ray_num - 2])
                                    +decay1**4*(self.lidar_obs[3]+self.lidar_obs[self.ray_num - 3])
                                    +decay1**5*(self.lidar_obs[4]+self.lidar_obs[self.ray_num - 4]))

            # *************** Part II Obstacle *************
            gaps = self.lidar_obs.copy()
            gaps.sort()
            decay2 = 0.9
            for i in range(12):
                reward2 += decay2 * log10(gaps[i])
                decay2 = decay2*decay2
            # ***************** Part III Middle ****************
            decay3 = 0.9
            index = round(self.ray_num/4)
            reward3 -= decay3*abs(self.lidar_obs[index]-self.lidar_obs[self.ray_num-index])\
                        +decay3**2*abs(self.lidar_obs[index+1]-self.lidar_obs[self.ray_num-(index+1)])\
                        +decay3**2*abs(self.lidar_obs[index-1]-self.lidar_obs[self.ray_num-(index-1)])\
                        +decay3**3*abs(self.lidar_obs[index+2]-self.lidar_obs[self.ray_num-(index+2)])\
                        +decay3**3*abs(self.lidar_obs[index-2]-self.lidar_obs[self.ray_num-(index-2)])\
            # **************** Part IV Time *****************
            reward4 += -1
            # **************** Total reward FOMT ****************
            reward = 0.5*reward1 + reward2 + reward3 + reward4
            self.reward_info = [0.5* reward1, reward2,reward3,reward4]
        else:
            reward = -50
        # print("*********** calced reward: ", reward)
        return reward
            


    
    def check_done(self):
        if self.check_collision() or self.check_achieve():
            self.done = True
        else:
            self.done = False
        return self.done
        
    def check_collision(self):
        if (self.lidar_obs> 0.).all():
            self.collision = False
        else:
            self.collision = True
        return self.collision
    
    def check_achieve(self):
        if dist(self.state[:2],self.global_goal[:2]) <= self.achieve_tolerance:
            self.achieve = True
        else:
            self.achieve = False
        return self.achieve



    def id_fill_body(self,map):
        # dv = 0.0001 # to calc map id value
        r = np.round(self.vertices[:4][:,1]/self.world_reso)
        c = np.round(self.vertices[:4][:,0]/self.world_reso)
        rr, cc = polygon(r, c)
        map[rr,cc] = self.value_base+self.dv*self.id # id=0: 0.9900; id=1:0.9901 how to differentiate? if 0.99+id*dv-dv/2 < value <0.99+id*dv+dv/2

    def fill_body(self, map):
        r = self.vertices[:4][:,1]
        c = self.vertices[:4][:,0]
        rr, cc = polygon(r, c)
        map[rr,cc] = 1

    def remove_body(self, map):
        r = self.vertices[:4][:,1]
        c = self.vertices[:4][:,0]
        rr, cc = polygon(r, c)
        map[rr,cc] = 0
    
    def calc_id_value(self):
        return self.value_base+self.dv*self.id

    def get_scans(self):
        return self.ranges
    def get_states(self):
        return self.state

# image_path = os.path.join(os.getcwd(),'map/racetrack.png')
# # img = mpimg.imread(path)

# img = io.imread(image_path, as_gray=True)/255.0
# param = CarParam()
# robot = CarRobot(id = 0, param = param, initial_state=np.array([10.,10.,0.,0.,0.,]))
# # robot.get_scans()
# print(robot.history)
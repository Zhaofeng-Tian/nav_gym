import numpy as np
from math import sqrt, sin, cos, pi,ceil

class CarParam:
    def __init__(self, type = "normal"):
        if type == "normal": # The params are based on our own robot ZebraT
            # wheelbase, front, rear suspension, half width, lidar to center(defaut mount on the front)
            self.shape = np.array([0.53,0.25,0.25,0.35,0.65])
            # limits 2X2 box ([v1_max, v2_max],
            #                 [v1_min, v2_min]) 
            # where v1 is linear speed, v2 is actually steering angle instead of angular speed in differential robots
            self.v_limits = np.array([[2.,0.6 ],
                                     [-1., -0.6]])
            self.a_limits = np.array([[0.8, 1.0],
                                     [-1., -1.0]])
            # Lidar param
            self.fan_range = np.array([0, 2*pi])
            self.min_range = 0.1
            self.max_range = 6.0
            self.ray_num = 64
            self.angle_reso = 0.098


            """
            The following params are imposed from shape.
            """
            self.geo_r = self.calc_geo_r()                  # Geometry circle radius
            self.disk_r = self.calc_disk_r()                # Collision circle radius
            self.disk_num = self.calc_disk_num()            # Collision circle number
            self.disk_centers = self.calc_disk_centers()    # Centers of collision circles
            # self.vertices contains array([four vertices + lidar center + geometry center])
            self.vertices = self.calc_vertices() # calc vertices and geometry center relative vector to vehicle center, then do tfs

    
    def calc_vertices(self):
        wheel_base = self.shape[0]
        front_sus = self.shape[1]
        rear_sus = self.shape[2]
        half_width = self.shape[3]
        rl = np.array([-rear_sus, half_width]) # rear left vertice
        rr = np.array([-rear_sus, -half_width])
        fl = np.array([wheel_base+front_sus, half_width ])
        fr = np.array([wheel_base+front_sus, -half_width])
        lc = np.array([self.shape[4], 0.]) # Lidar center 
        gc = np.array([(wheel_base+rear_sus+front_sus)/2-rear_sus, 0]) # Geometry center
        return np.array([rl,rr,fr,fl,lc,gc])

    def calc_disk_r(self):
        half_width = self.shape[3]
        return np.sqrt(half_width**2/2)
    
    def calc_geo_r(self):
        wheel_base, front_sus, rear_sus, half_width = self.shape[0:4]
        half_len = (wheel_base+front_sus+rear_sus)/2
        print("Geo Radius: ", sqrt(half_len**2 + half_width**2) )
        return sqrt(half_len**2 + half_width**2) 

    def calc_disk_num(self):
        wheel_base = self.shape[0]
        front_sus = self.shape[1]
        rear_sus = self.shape[2]
        half_width = self.shape[3]
        num = ceil(wheel_base+front_sus+rear_sus)/ (2* half_width )
        return num.astype(np.int32)
    
    def calc_disk_centers(self):
        # First calc front and rear disk
        wheel_base = self.shape[0]
        front_sus = self.shape[1]
        rear_sus = self.shape[2]
        half_width = self.shape[3]
        centers=[]
        rear_center = np.array([half_width, 0.])
        front_center = np.array([wheel_base+rear_sus+front_sus-half_width, 0.])
        centers.append(rear_center)
        dx = (front_center[0] - rear_center[0])/(self.disk_num-1)
        for i in range(self.disk_num-2):
            centers.append(np.array([rear_center[0]+dx*(i+1),0.]))
        centers.append(front_center)
        return np.array(centers)


        
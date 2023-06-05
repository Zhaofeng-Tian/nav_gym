
import numpy as np
from math import pi, cos,sin
from nav_gym.obj.geometry.util import *

def generate_angles(fan_range = np.array([0, 2*pi]), max_range = 6.0, ray_num = 32, reso = 0.098, is_num_based = True, ):
    if is_num_based:
        angles = np.linspace(fan_range[0], fan_range[1], ray_num+1)[:-1] # +1 and [:-1] to remove angle=6.28 that overlaps angle=0.0
    else:
        angles = np.linspace(fan_range[0], fan_range[1], (fan_range[1]-fan_range[0])/reso+1)[:-1]

    print("Got angles, the shape is: ",angles.shape)
    return angles

def generate_ends(angles, max_range = 6.0):
    end_points = []
    for i in range(len(angles)):
        end_points.append(np.array([max_range*cos(angles[i]), max_range*sin(angles[i])]))
    
    return np.array(end_points)

def ends_tf(end_points, theta = 0., trans = np.array([0.,0.])):
    return (rot(theta)@end_points.T).T + trans
    

def generate_range_points(start, ends, map,polygons,circles,max_range = 6.0):
    
    ranges = []
    points = [] # x,y coords of real points
    for end in ends:
        min_range = max_range
        min_point = end
        # 1. line map
        p = line_map((start,end),map, reso=0.01) # line_map does not return None, so its a baseline
        ran = dist(start,p)
        if ran < min_range:
            min_range = ran
            min_point = p
        # 2. line polygons
        # print("Lidar testing==> polygons len: ", len(polygons))
        for polygon in polygons:
            p = line_polygon((start,end), polygon)
            # print("Lidar testing==> intersection : ", p)
            if p ==None:
                continue
            else:
                ran = dist(start,p)
                if ran < min_range:
                    min_range = ran
                    min_point = p
        # 3. line circles
        for circle in circles:
            p = line_circle((start,end), circle)
            if p == None:
                continue
            else:
                ran = dist(start,p)
                if ran < min_range:
                    min_range = ran
                    min_point = p
        ranges.append(min_range)
        points.append(min_point)
    return np.array(ranges,dtype=np.float32), np.array(points,dtype=np.float32)


def generate_safe_range_points(start, ends, safe_vertices,max_range = 6.0):
    
    ranges = []
    points = [] # x,y coords of real points
    polygon = Polygon(safe_vertices)
    # print("polygon vertices: ", polygon.vertices)
    # print("polygon's edges: ", polygon.edges)
    # print("start: ",start)
    # print("ends: ", len(ends))
    for i in range(len(ends)):
        min_range = max_range
        min_point = ends[i]
        

        p = line_polygon((start,ends[i]), polygon)
        # print("Lidar testing==> intersection : ", p)
        if p ==None:
            # print("ends[i] no intersection: ",i)
            # print(ends[i])
            continue
        else:
            ran = dist(start,p)
            if ran < min_range:
                min_range = ran
                min_point = p

        ranges.append(min_range)
        points.append(min_point)
        # print("ranges: ", ranges)
    return np.array(ranges, dtype=np.float32), np.array(points, dtype=np.float32)





def map_based_generate_range_points(start, ends, map,max_range = 6.0, reso=0.01):
    
    ranges = []
    points = [] # x,y coords of real points
    for end in ends:
        min_range = max_range
        min_point = end
        # 1. line map
        p = line_map((start,end),map, reso) # line_map does not return None, so its a baseline
        ran = dist(start,p)
        if ran < min_range:
            min_range = ran
            min_point = p
        ranges.append(min_range)
        points.append(min_point)
    return np.array(ranges, dtype=np.float32), np.array(points, dtype=np.float32)

def map_id_generate_range_points(start, ends, map, id_value,dv, max_range = 6.0, reso=0.01):
    
    ranges = []
    points = [] # x,y coords of real points
    for end in ends:
        min_range = max_range
        min_point = end
        # 1. line map
        p = id_line_map((start,end),map, reso,id_value,dv) # line_map does not return None, so its a baseline
        ran = dist(start,p)
        if ran < min_range:
            min_range = ran
            min_point = p
        ranges.append(min_range)
        points.append(min_point)
    return np.array(ranges, dtype=np.float32), np.array(points, dtype=np.float32)



def test_ends():
    a= generate_angles(ray_num=4)
    e = generate_ends(a)
    print("angles: " ,a)
    print("ends ", e)
    print(e.shape)
    print(e.T)
    theta = pi/2
    new_ends = (np.array([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])@e.T).T + np.array([0.,1.0])
    print(new_ends)
# test_ends()
import numpy as np
from nav_gym.obj.geometry.objects import Polygon, Circle
from math import cos, sin, sqrt

"""
The functions start with "check" return bool.
The functions defined as obj_obj return intersection point coords.
"""
# Check Polygon Circle collisions
def check_poly_circ(polygon, circle):
    # Check if any of the polygon edges intersect the circle
    for edge in polygon.edges:
        v1, v2 = edge
        d = v2 - v1
        f = v1 - circle.center
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - circle.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant >= 0:
            # Collision detected
            return True
    
    # Check if any of the circle edges are inside the polygon
    sampled_points = circle.sample_points(0.1)
    for point in sampled_points:
        if polygon.contains_point(point):
            # Collision detected
            return True
    
    # No collision detected
    return False


def check_poly_poly(poly1, poly2):
    # Check edges of poly1
    for edge in poly1.edges:
        axis = np.array([-edge[1][1]+edge[0][1], edge[1][0]-edge[0][0]]).astype(np.float32)
        axis /= np.linalg.norm(axis)
        poly1_min = np.dot(poly1.vertices, axis)
        poly1_max = np.dot(poly1.vertices, axis)
        poly2_min = np.dot(poly2.vertices, axis)
        poly2_max = np.dot(poly2.vertices, axis)
        if (np.max(poly1_min) < np.min(poly2_max)) or (np.max(poly2_min) < np.min(poly1_max)):
            return False
    return True

vertices = [(0,0), (1,0), (1,1), (0,1)]
vertices2=[(5,0), (6,0), (6,1), (5,1)]

def check_circ_circ(circle1, circle2):
    dist = np.linalg.norm(circle1.center - circle2.center)
    return dist < circle1.radius + circle2.radius



def line_line(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    # calculate slopes and y-intercepts
    m1 = (y2 - y1) / (x2 - x1)
    b1 = y1 - m1 * x1
    
    m2 = (y4 - y3) / (x4 - x3)
    b2 = y3 - m2 * x3
    
    # check if lines are parallel
    if m1 == m2:
        return None
    
    # calculate point of intersection
    x_intersect = (b2 - b1) / (m1 - m2)
    y_intersect = m1 * x_intersect + b1
    
    return np.array([x_intersect, y_intersect])



def line_polygon(ray, polygon):
    closest_intersection = None
    min_distance = np.inf
    
    for i in range(len(polygon.vertices)):
        p1 = polygon.vertices[i]
        p2 = polygon.vertices[(i+1)%len(polygon.vertices)]
        
        edge = (p1, p2)
        intersection = line_line(ray, edge)
        
        if intersection is not None:
            distance = np.linalg.norm(ray[0] - intersection)
            if distance < min_distance:
                closest_intersection = intersection
                min_distance = distance
    
    return closest_intersection



def line_circle(line, circle):
    # calculate the direction vector of the line segment
    circle_center = circle.center; radius = circle.radius
    start = line[0]; end = line[1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # calculate the distance between the circle center and the line
    a = dx**2 + dy**2
    b = 2*dx*(start[0] - circle_center[0]) + 2*dy*(start[1] - circle_center[1])
    c = (start[0] - circle_center[0])**2 + (start[1] - circle_center[1])**2 - radius**2
    
    # calculate the discriminant of the quadratic equation
    discriminant = b**2 - 4*a*c
    
    # if the discriminant is negative, the line does not intersect the circle
    if discriminant < 0:
        return None
    
    # calculate the two possible values of t
    t1 = (-b + sqrt(discriminant)) / (2*a)
    t2 = (-b - sqrt(discriminant)) / (2*a)
    
    # calculate the intersection points
    intersection1 = (start[0] + t1*dx, start[1] + t1*dy)
    intersection2 = (start[0] + t2*dx, start[1] + t2*dy)
    
    # return the intersection points as a list
    return np.array[intersection1, intersection2]    



def line_map(line,map,reso):

    x0, y0 = line[0]
    x1, y1 = line[1]
    x0 = round(x0/reso); y0 = round(y0/reso)
    x1 = round(x1/reso); y1 = round(y1/reso) 
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    # x = []
    # y = []
    while (x0 != x1 or y0 != y1):
        if map[y0,x0]== 0:
            return x0*reso,y0*reso
        # x.append(x0)
        # y.append(y0)
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return x1*reso, y1*reso



def dist(point1,point2):
    # calculate the difference between the x and y coordinates of the two points
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    # calculate the distance using the Pythagorean theorem
    distance = np.sqrt(dx**2 + dy**2)

    return distance

def rot(theta):
    return np.array([[cos(theta), -sin(theta)],[sin(theta), cos(theta)]])

# polygon = Polygon(vertices)
# p2 = Polygon(vertices2)
# circle = Circle((0.5, 0.5), 0.4)
# print(check_poly_circ(polygon, circle)) # True

# circle = Circle((2,2), 0.4)
# circle2 = Circle((5,5),4)
# print(check_poly_circ(polygon, circle)) # False

# print(check_poly_poly(polygon,p2))
# print(check_circ_circ(circle,circle2))

# line1 = ((0, 0), (1, 1))
# line2 = ((1, 1), (7, 8))

# intersect = line_line(line1, line2)
# if intersect is not None:
#     print(f"The lines intersect at ({intersect[0]}, {intersect[1]})")
# else:
#     print("The lines are parallel and do not intersect.")
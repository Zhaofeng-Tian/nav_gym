import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
import skimage
from skimage import io, feature
import time
# import cv2

# Read the image as a NumPy array
image_path = os.path.join(os.getcwd(),'map/racetrack.png')
# img = mpimg.imread(path)

img = io.imread(image_path, as_gray=True)/255.0
edges = feature.canny(img, sigma=3)
print(img.shape)

# Display the original image and the edges using matplotlib
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('Original Image')
# ax[1].imshow(edges, cmap='gray')
# ax[1].set_title('Edges')
# plt.show()

y, x = np.where(edges)
print(x)
print(y)
print(x.shape)
print(y.shape)
# print(np.where(edges))


import numpy as np

def build_ray(start, end,map):
    """
    Builds a line between two points using Bresenham's line algorithm.

    Parameters:
        start (tuple): The starting point of the line in (x, y) format.
        end (tuple): The ending point of the line in (x, y) format.

    Returns:
        x (numpy.ndarray): The x-coordinates of the line pixels.
        y (numpy.ndarray): The y-coordinates of the line pixels.
    """
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    # x = []
    # y = []
    while (x0 != x1 or y0 != y1):
        if map[y0,x0]== 0:
            return x0,y0
        # x.append(x0)
        # y.append(y0)
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy



    return x1, y1




def intersects_edge(p1,p2,edge_x,edge_y):
    # slop and bias
    x1,y1 = p1 # start
    x2,y2 = p2 # end
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1

    iex = np.where((edge_x<=max(x1,x2)) & (edge_x>=min(x1,x2)))
    iey = np.where((edge_y<=max(y1,y2)) & (edge_y>=min(y1,y2)))
    index = np.union1d(iex, iey)
    # print(edge_x[index].shape)
    # print(edge_y[index].shape)
    # print(iex)
    # print(iey)
    xx = edge_x[index]
    yy = edge_y[index]
    yy_ = m*xx+b
    iintersec = np.where(abs(yy-yy_)<=1)
    # print(xx[iintersec].shape)
    # print(yy[iintersec])
    xxi = xx[iintersec]
    yyi = yy[iintersec]
    distance = xxi**2 + yyi**2
    ii = np.argmin(distance)
    return xxi[ii], yyi[ii]


def line_edge_intersection(p1,p2 , edge_x, edge_y):
    # Compute slope and y-intercept of line
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    # Initialize closest intersection point
    closest_point = None
    closest_distance = np.inf
    
    # Check each edge segment for intersection
    for i in range(len(edge_x) - 1):
        # Compute slope and y-intercept of segment
        m2 = (edge_y[i+1] - edge_y[i]) / (edge_x[i+1] - edge_x[i])
        b2 = edge_y[i] - m2 * edge_x[i]
        
        # Check if lines are parallel
        if np.isclose(m, m2):
            continue
        
        # Compute x-coordinate of intersection point
        x_intersect = (b2 - b) / (m - m2)
        
        # Check if intersection point is on line segment
        if x_intersect < min(x1, x2) or x_intersect > max(x1, x2):
            continue
        
        # Compute y-coordinate of intersection point
        y_intersect = m * x_intersect + b
        
        # Compute distance to start point and update closest point if necessary
        distance = np.sqrt((x_intersect - x1)**2 + (y_intersect - y1)**2)
        if distance < closest_distance:
            closest_point = (x_intersect, y_intersect)
            closest_distance = distance
    
    return closest_point

start = (2000, 400)
end = (2900, 600)
print(build_ray(start,end,img))
print(line_edge_intersection(start, end , x, y))

num_loop = 180
start_time = time.time()
for i in range(num_loop):

    intersects_edge(start,end,x,y)
end_time = time.time()
time_cost = end_time-start_time
print("Run {} loops, time coast {} s!  ".format(num_loop, time_cost))
start_time = time.time()

for i in range(num_loop):
    build_ray(start,end,img)
end_time = time.time()
time_cost = end_time-start_time
print("Run {} loops, time coast {} s!  ".format(num_loop, time_cost))

# Display the original image and the edges using matplotlib
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, origin = 'lower',cmap='gray')
ax[0].set_title('Binary Image')

# Plot the edges using x and y arrays
ax[1].imshow(img, origin = 'lower', cmap='gray')
ax[1].plot(x, y, 'r.')
ax[1].plot([start[0],end[0]], [start[1], end[1]])
ax[1].set_title('Edges')
plt.show()
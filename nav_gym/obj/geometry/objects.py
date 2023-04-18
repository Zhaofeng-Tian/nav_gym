import numpy as np

class Polygon:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)
        self.edges = self._get_edges()

    def _get_edges(self):
        num_vertices = len(self.vertices)
        edges = [(self.vertices[i], self.vertices[(i+1)%num_vertices]) for i in range(num_vertices)]
        return np.array(edges)

    def perimeter(self):
        return np.sum(np.linalg.norm(self.edges[:,1] - self.edges[:,0], axis=1))

    def area(self):
        return 0.5 * np.abs(np.dot(self.vertices[:,0], np.roll(self.vertices[:,1],1)) - np.dot(self.vertices[:,1], np.roll(self.vertices[:,0],1)))

    def sample_points(self, resolution):
        sampled_points = []
        for i in range(len(self.vertices)):
            v1 = self.vertices[i]
            v2 = self.vertices[(i+1)%len(self.vertices)]
            edge_length = np.linalg.norm(v2 - v1)
            num_samples = int(np.ceil(edge_length / resolution))
            if num_samples > 1:
                edge_samples = np.linspace(v1, v2, num=num_samples, endpoint=False)
                sampled_points.extend(edge_samples)
        sampled_points.append(self.vertices[0])
        return np.array(sampled_points)
    
    def contains_point(self, point):
        inside = False
        for edge in self.edges:
            if ((edge[0][1] > point[1]) != (edge[1][1] > point[1])) and \
            (point[0] < (edge[1][0] - edge[0][0]) * (point[1] - edge[0][1]) / (edge[1][1] - edge[0][1]) + edge[0][0]):
                inside = not inside
        return inside

class Circle:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def area(self):
        return np.pi * self.radius**2

    def circumference(self):
        return 2 * np.pi * self.radius

    def contains(self, point):
        return np.linalg.norm(self.center - point) <= self.radius

    def sample_points(self, reso):
        theta = np.linspace(0, 2*np.pi, int(2*np.pi/reso), endpoint=False)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        return np.array(list(zip(x,y)))

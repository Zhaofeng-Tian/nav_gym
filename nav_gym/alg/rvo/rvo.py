import numpy as np
from sklearn.neighbors import KDTree
from math import atan2,cos,sin,sqrt
import casadi as ca
class RVO():
    def __init__(self, n_agents):
        self.n_agents = n_agents
    
    def choose_action(self, obs, r_sens = 3):
        goals = obs["goal"].copy()
        centers = obs["center"].copy()
        velocities = obs["velocity"].copy()
        v_vectorized = self.vectorize_v(velocities)
        radii = obs["radius"].copy()
        tree = KDTree(centers)
        print("tree: ", tree)

        print("centers: ", centers)
        for i in range(len(centers)):
            print("agent id: ", i)
            # index = tree.query_radius(centers[i:i+1], r=5)
            dist, index = tree.query(centers[i:i+1], k=3)
            print("NN index: ", index, "  distance: ", dist)
            print("index[1:]: ",index[0][1:])
            # print("centers[index]: ", centers[index[0][1:]])
            VO_all = []
            for ind in index[0][1:]:
                print("ind: ", ind)
                VO = self.calc_VO(centers[i],centers[ind],v_vectorized[i], velocities[ind], radii[i], radii[ind])
                VO_all.append(VO)
            # about the limit, v_limits is defined in linear v and angular w, a_limits is their derivation.
            # consider steering angle:= phi, wheelbase:= L
            # v for robot [-1,2]
            # w for Ackermann-steering robot is constrained by vmax*tan(phi_max)/L
            self.calc_desired_v(VO_all,centers[i], goals[i], v_vectorized[i], v_limits = np.array([[-2,2],[-2,2]]), a_limits=np.array([[-0.5,0.5],[-1,1]]))
    
    def calc_desired_v(self, VO_all, pa, goal, v, v_limits, a_limits, dt):
        goal_c = pa - goal # goal w.r.t. current position
        vdes = goal_c * v_limits[0,1]/self.dist(goal_c)
        self.build_VO_problem(VO_all, pa, goal, v, v_limits, a_limits, dt)
        for VO in VO_all:
            # VO: pvo_px, pvo_py x_lleg, y_lleg, x_rleg, y_rleg
            v[0]-VO[0]
            VO[1]

    def build_VO_problem(self, VO_all, pa, goal, v, v_limits, a_limits, dt):
        goal_c = pa - goal # goal w.r.t. current position
        vdes = goal_c * v_limits[0,1]/self.dist(goal_c)
        vmax = v_limits[0,1]
        opti = ca.Opti();
        vcx,vcy = v; vdesx,vdesy = vdes
        dvx = opti.variable();
        dvy = opti.variable();


        opti.minimize(  (vx-vdesx)**2+(vy-vdesy)**2   );
        # dv subject to acceleration limits
        opti.subject_to(  );
        opti.subject_to(  vx**2+vy**2 <=vmax**2 );

        opti.solver('ipopt');

        sol = opti.solve();

        sol.value(x)
        sol.value(y)
              
    def calc_VO(self, pa, pb, va, vb, ra, rb):
        pc = pb - pa
        print("comparative position: ", pc)
        vc = va - vb
        print("comparative velocity: ", vc)
        rc = ra+rb
        pvo = pa+vb # position of voVO_all = []
        theta = atan2(pc[1],pc[0])
        pc_dist = self.dist(pc)
        leg_len = sqrt(pc_dist**2-rc**2)
        phi = atan2(rc,leg_len) # central line angle
        theta_left = theta+phi
        theta_right = theta-phi
        x_lleg = cos(theta_left)
        y_lleg = sin(theta_left)
        x_rleg = cos(theta_right)
        y_rleg = sin(theta_right)

        return np.array([pvo[0],pvo[1], x_lleg, y_lleg, x_rleg, y_rleg])

    def calc_RVO(self, pa, pb, va, vb, ra, rb):
        pc = pb - pa
        print("comparative position: ", pc)
        vc = va - vb
        print("comparative velocity: ", vc)
        rc = ra+rb
        pvo = pa+(va+vb)/2 # position of voVO_all = []
        theta = atan2(pc[1],pc[0])
        pc_dist = self.dist(pc)
        leg_len = sqrt(pc_dist**2-rc**2)
        phi = atan2(rc,leg_len)
        theta_left = theta+phi
        theta_right = theta-phi
        x_lleg = cos(theta_left)
        y_lleg = sin(theta_left)
        x_rleg = cos(theta_right)
        y_rleg = sin(theta_right)

        return np.array([pvo[0],pvo[1], x_lleg, y_lleg, x_rleg, y_rleg])       

    def vectorize_v(self,velocities):
        vectorized_v = []
        for v in velocities:
            vectorized_v.append(np.array([cos(v[0])*v[1], sin(v[0])*v[1]]))
        return vectorized_v

    def dist(self, vector):
        return sqrt(vector[0]**2+vector[1]**2)


agent = RVO(10)

obs = {'lidar': [[6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.7278361, 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        2.7278361, 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.7278361, 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        2.7278361, 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.7278361, 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        2.7278361, 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.7278361, 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        2.7278361, 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.5      , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.7278361, 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        2.7278361, 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.7278361, 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        2.7278361, 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.7278361, 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        2.7278361, 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ],
       [6.       , 6.       , 6.       , 6.       , 6.       , 4.822379 ,
        6.       , 2.430086 , 2.7278361, 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        6.       , 6.       , 6.       , 6.       , 6.       , 6.       ,
        2.7278361, 2.430086 , 6.       , 4.822379 , 6.       , 6.       ,
        6.       , 6.       ]], 'goal': [[10.,  0.],
       [10.,  0.],
       [10.,  0.],
       [10.,  0.],
       [10.,  0.],
       [10.,  0.],
       [10.,  0.],
       [10.,  0.],
       [10.,  0.],
       [10.,  0.]], 'center': [np.array([12.235,  7.5  ]), np.array([11.33069547, 10.28316317]), np.array([ 8.96319547, 12.0032526 ]), np.array([ 6.03680453, 12.0032526 ]), np.array([ 3.66930453, 10.28316317]), np.array([2.765, 7.5  ]), np.array([3.66930453, 4.71683683]), np.array([6.03680453, 2.9967474 ]), np.array([8.96319547, 2.9967474 ]), np.array([11.33069547,  4.71683683])], 'velocity': [np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ]), np.array([3.14159265, 0.        ])], 'radius': [0.6226756780218736, 0.6226756780218736, 0.6226756780218736, 0.6226756780218736, 0.6226756780218736, 0.6226756780218736, 0.6226756780218736, 0.6226756780218736, 0.6226756780218736, 0.6226756780218736]}
print(obs)

agent.choose_action(obs)
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

    def build_VO_problem( self, VO_all, state, goal, v_limits= np.array([[-2,2],[-2,2]]), a_limits=np.array([[-0.5,0.5],[-1.5,1.5]]), dt=0.2):
        vmax = v_limits[0,1] # Linear speed max
        vmin = v_limits[0,0]
        amax = a_limits[0,1]  # linear acceleration limit
        bmax = a_limits[0,0]  # brake decceleration limit
        wmax = v_limits[1,1]  # rotate speed limit
        wmin = v_limits[1,0]
        wamax = a_limits[1,1]  # rotation acceleration limit
        wrmax = a_limits[1,0]  # reverse rotation acceleration limit
        
        px,py, theta_cur, v_cur, w_cur = state
        goal_c =  goal - state[0:2] # goal w.r.t. current position
        vdes = goal_c * vmax/self.dist(goal_c)
        v_des,theta_des = self.factorize_vector(vdes) # desired v
        dt = dt 
        
        # build optimization problem
        opti = ca.Opti();    
        v = opti.variable();
        w = opti.variable();
        # 0. objective
        opti.minimize(  (v-v_des)**2+(theta_cur+w*dt-theta_des)**2 )

        # 1. linear speed value constraint
        opti.subject_to( v <= vmax)
        opti.subject_to( v >= vmin)

        # 2. dv subject to acceleration limits
        opti.subject_to( v-v_cur <= amax * dt  ) 
        opti.subject_to( v-v_cur >= bmax * dt )     

        # # 3. angular speed value constraint
        opti.subject_to( w <= wmax)
        opti.subject_to( w >= wmin)
        
        # 4. angular acceleration 
        opti.subject_to( w-w_cur <= wamax*dt)
        opti.subject_to( w-w_cur >= wrmax*dt)

        for VO in VO_all:
            # VO = VO_all[0]
            # VO  :=   px,  py  ,x_left_leg, y_left_leg, x_r_leg, y_r_leg
            pxvo = VO[0]
            pyvo = VO[1]
            x_lleg = VO[2]
            y_lleg = VO[3]
            x_rleg = VO[4]
            y_rleg = VO[5]

            # # 4. subject to VO left leg   (vt − vp) × vl < 0 & (vt − vp) × vr > 0
            opti.subject_to( (v*np.cos(theta_cur+w*dt)-pxvo)*y_lleg-(v*np.sin(theta_cur+w*dt)-pyvo)*x_lleg < 0 )

            # # 5. subject to VO right leg   (vt − vp) × vl < 0 & (vt − vp) × vr > 0
            opti.subject_to( (v*np.cos(theta_cur+w*dt)-pxvo)*y_rleg-(v*np.sin(theta_cur+w*dt)-pyvo)*x_rleg < 0 )
            print(" Iteration for VO constraints")
        opti.solver('ipopt');
        No_solution = False
        try:
            sol = opti.solve()
        except RuntimeError as error:
            print("No solution in original problem")
            print("deug value: ",opti.debug.value(v))
            No_solution = True

        if No_solution:
            v_computed = opti.debug.value(v) 
            w_computed = opti.debug.value(w)
        else: 
            v_computed = sol.value(v) 
            w_computed = sol.value(w) 
        return np.array([v_computed,w_computed])
              
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

    def dist(vector):
        return sqrt(vector[0]**2+vector[1]**2)

    def factorize_vector(vector):
        return np.array([self.dist(vector),atan2(vector[1],vector[0])])


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
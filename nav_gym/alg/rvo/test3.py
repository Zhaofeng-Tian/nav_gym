import casadi as ca
from math import cos,sin,atan2,sqrt,pi
import numpy as np


def dist(vector):
    return sqrt(vector[0]**2+vector[1]**2)

def factorize_vector(vector):
    return np.array([dist(vector),atan2(vector[1],vector[0])])

def build_VO_problem( VO_all, state, goal, v_limits= np.array([[-2,2],[-2,2]]), a_limits=np.array([[-0.5,0.5],[-1.5,1.5]]), dt=0.2):
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
    vdes = goal_c * vmax/dist(goal_c)
    v_des,theta_des = factorize_vector(vdes) # desired v
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


    print("Computed linear speed: ",v_computed)
    print("Computed rotation speed: ",w_computed)
    print("desired v: ", vdes)

VO_all = [ np.array([0.5,0.5,cos(150/180*pi), sin(150/180*pi), cos(120/180*pi), sin(120/180*pi)]), np.array([0.9,0.9,cos(150/180*pi), sin(150/180*pi), cos(120/180*pi), sin(120/180*pi)]) ]
state = np.array([0.,0.,0. , 0.3 , 0.0])
goal =np.array([4.,8.])



build_VO_problem(VO_all, state, goal)
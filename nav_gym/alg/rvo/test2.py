import casadi as ca
from math import cos,sin,atan2,sqrt,pi
import numpy as np


def dist(vector):
    return sqrt(vector[0]**2+vector[1]**2)

def build_VO_problem( VO_all, pa, goal, v, v_limits= np.array([[-2,2],[-2,2]]), a_limits=np.array([[-0.5,0.5],[-1,1]]), dt=0.2):
    VO = VO_all[0]
    # VO  :=   px,  py  ,x_left_leg, y_left_leg, x_r_leg, y_r_leg
    pxvo = VO[0]
    pyvo = VO[1]
    x_lleg = VO[2]
    y_lleg = VO[3]
    x_rleg = VO[4]
    y_rleg = VO[5]
    goal_c =  goal -pa # goal w.r.t. current position
    vdes = goal_c * v_limits[0,1]/dist(goal_c)
    vmax = 2. # 
    amax = 0.5 # linear acceleration limit
    wmax = 2. # rotate speed limit
    

    opti = ca.Opti();
    px, py = pa # agent current position
    vcx,vcy = v # current velocity
    vdesx,vdesy = vdes # desired v
    len_v = sqrt(vx**2+vy**2)
    len_cv = sqrt(vcx**2+vcy**2)
    
    vx = opti.variable();
    vy = opti.variable();
    opti.minimize(  (vx-vdesx)**2+(vy-vdesy)**2   )

    # 1. linear speed value constraint
    opti.subject_to( vx**2+vy**2 <=vmax**2 )

    # 2. dv subject to acceleration limits
    opti.subject_to( (vx**2+vy**2) - (vcx**2+vcy**2) <=amax*dt  )   

    # # 3. angular speed value constraint
    # opti.subject_to( np.arctan2(vy, vx)-np.arctan2(vcy,vcx) >= -wmax*dt) 
    # opti.subject_to( np.arctan2(vy, vx)-np.arctan2(vcy,vcx) <= wmax*dt) 

    # # 4. subject to VO left leg   (vt − vp) × vl < 0 & (vt − vp) × vr > 0
    # opti.subject_to( (vx-pxvo)*y_lleg-(vy-pyvo)*x_lleg < 0 )

    # # 5. subject to VO right leg   (vt − vp) × vl < 0 & (vt − vp) × vr > 0
    # opti.subject_to( (vx-pxvo)*y_rleg-(vy-pyvo)*x_rleg < 0 )

    opti.solver('ipopt');

    sol = opti.solve();

    sol.value(vx)
    print("vx: ",sol.value(vx))
    sol.value(vy)
    print("vy: ",sol.value(vy))
    print("desired v: ", vdes)
VO_all = [ np.array([0.5,0.5,cos(150/180*pi), sin(150/180*pi), cos(120/180*pi), sin(120/180*pi)]) ]
pa = np.array([0.,0.])
goal =np.array([4.,8.])
v = np.array([0.,0.])

build_VO_problem(VO_all, pa, goal, v)
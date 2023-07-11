import casadi as ca
from math import cos,sin,atan2

vx = ca.SX.sym('vx')
vy = ca.SX.sym('vy')

xll = ca.SX.sym('xll'); yll = ca.SX.sym('yll')
xrl = ca.SX.sym('xrl'); yrl = ca.SX.sym('yrl')
px = ca.SX.sym('px'); py = ca.SX.sym('py')

f = vx**2 + vy**2

qp = {'x':ca.vertcat(vx,vy), 'f':f}
S = ca.qpsol('S','qpoases', qp)
print(S)
r = S(lbg = 0)
x_opt = r['x']
print('x_opt:',x_opt)
print(x_opt.shape)
print(x_opt)

# opti = ca.Opti()
# x = opti.variable(); y = opti.variable()
# opti.minimize(x**2+y**2)
# opti.solver('ipopt')
# sol = opti.solve()
# print(sol.value(x))

# x = SX.sym('x'); y = SX.sym('y')
# qp = {'x':vertcat(x,y), 'f':x**2+y**2, 'g':x+y-10}
# S = qpsol('S', 'qpoases', qp)
# print(S)
# 
# 
# v_limits = np.array([[-2,2],[-2,2]]), a_limits=np.array([[-0.5,0.5],[-1,1]]))

def build_VO_problem(self, VO_all, pa, goal, v, v_limits= np.array([[-2,2],[-2,2]]), a_limits=np.array([[-0.5,0.5],[-1,1]]), dt=0.2):
    goal_c = pa - goal # goal w.r.t. current position
    vdes = goal_c * v_limits[0,1]/self.dist(goal_c)
    vmax = v_limits[0,1]
    opti = ca.Opti();
    vcx,vcy = v; # current velocity
    vdesx,vdesy = vdes # desired v
    dvx = opti.variable();
    dvy = opti.variable();


    # opti.minimize(  (vx-vdesx)**2+(vy-vdesy)**2   );
    # dv subject to acceleration limits
    # opti.subject_to( atan2(vx, vy)-atan2(vcx,vcy)>= );
    # linear speed value constraint
    opti.subject_to( vx**2+vy**2 <=vmax**2 );
    # angular speed value constraint
    opti.subject_to()

    opti.solver('ipopt');

    sol = opti.solve();

    sol.value(vx)
    sol.value(vy)
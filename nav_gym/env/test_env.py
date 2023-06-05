# import gymnasium as gym
# from gym.envs.registration import register
from nav_gym.env.register import Make_Env
# from nav_gym.env.narrow_env import NarrowSpaceEnv_v0
#import nav_gym
import numpy as np  
actions = np.zeros((10,2))
env = Make_Env('Narrow-v0')
o1 = env.reset()
for i in range(100):
    print(i)
    # env.step(np.array([[1.0,0.0],[0.3,0.0]]))
    env.step(actions)
    env.render_frame()
print(o1)
# env = gym.make('gym_examples:gym_examples/GridWorld-v0)
# env = gym.make('Narrow-v0')
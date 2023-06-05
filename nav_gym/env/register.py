# import gymnasium as gym
import gym
from gym.envs.registration import register

# register(
#     id='Narrow-v0',
#     entry_point='nav_gym.env:NarrowSpaceEnv_v0',
#     max_episode_steps=1000,
# )





def Register_Env(task_env, max_episode_steps = 10000):
    """ Check whether the input task_env in the task list """
    task_list = ['Narrow-v0', 'Narrow-v1', 'Corridor-v0']
    if task_env not in task_list:
        return False
    if task_env == 'Narrow-v0':
        register(
            id = task_env,
            entry_point = 'nav_gym.env.narrow_env:NarrowSpaceEnv_v0',
            max_episode_steps = max_episode_steps,)
        from nav_gym.env.narrow_env import NarrowSpaceEnv_v0
        return True

    if task_env == 'Narrow-v1':
        register(
            id = task_env,
            entry_point = 'nav_gym.env.narrow_env:NarrowSpaceEnv_v1',
            max_episode_steps = max_episode_steps,)
        from nav_gym.env.narrow_env import NarrowSpaceEnv_v1
        return True

    if task_env == 'Corridor-v0':
        register(
            id = task_env,
            entry_point = 'nav_gym.env.corridor_env:CorridorEnv_v0',
            max_episode_steps = max_episode_steps,)
        from nav_gym.env.corridor_env import CorridorEnv_v0
        return True




def Make_Env(env_name):
    """ If in list, gym make the env """
    env_in_list = Register_Env(task_env=env_name, max_episode_steps= 10000)
    print("Making Env!")
    if env_in_list:
        print("Registered Task Env Successfully!")
        env = gym.make(env_name)
    else:
        env = None
        raise Exception("Task Env Not Found, Try Another!!")
    return env
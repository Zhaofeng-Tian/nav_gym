import gym
import numpy as np
from nav_gym.alg.facmac.agent import Agent
from utils import plot_learning_curve
from nav_gym.env.register import Make_Env
import torch as T

T.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    T.autograd.set_detect_anomaly(True)
    env = Make_Env('Corridor-v0')
    agent = Agent(alpha=0.0001, beta=0.0002, 
                    obs_dim=34, tau=0.001,
                    batch_size=64, fc1_dims=512, fc2_dims=512, 
                    n_actions=env.action_space.shape[0],
                    n_agents= env.n_cars, model_path = 'C:\\Users\\61602\\Desktop\\Coding\\models\\corridor_model')
    n_games = 1000
    epsilon = 0.7

    if_print = True
    best_score = env.reward_range[0]
    score_history = []
    step_ctr = 0
    for i in range(n_games):
        epsilon -= (0.7-0.1)/n_games
        print("****** Episode ", str(i), " ********")
        observation = env.reset()
        print(observation)
        terminated = False
        score = 0
        agent.noise.reset()
        while not terminated:
            action = agent.choose_action(observation, epsilon)
            
            observation_, reward, done, truncated,info = env.step(action)
            step_ctr += 1
            for flag in done:
                if flag == True:
                    terminated = True
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            for r in reward:
                score += r
            observation = observation_
            if if_print:
                print(" **** Episode: ",str(i), " ** Step: ", step_ctr)
                print("1. action: ", action)
                print("2. reward: ",reward)
                print("2. obs_: ", observation)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    # x = [i+1 for i in range(n_games)]
    # plot_learning_curve(x, score_history, figure_file)





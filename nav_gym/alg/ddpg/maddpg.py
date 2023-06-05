import gym
import numpy as np
from ddpg_torch import Agent2, Agent_VDN,Agent_VDN2,Agent_QMix, Agent_LagMix
from utils import plot_learning_curve
from nav_gym.env.register import Make_Env
import os

# use VDN2 
if __name__ == '__main__':
    env = Make_Env('Corridor-v0')
    agent = Agent2(alpha=0.0001, beta=0.0002, 
                    input_dims=34, tau=0.001,
                    batch_size=64, fc1_dims=512, fc2_dims=512, 
                    n_actions=env.action_space.shape[0],
                    n_agents= env.n_cars, model_path = 'C:\\Users\\61602\\Desktop\\Coding\\models\\corridor_model')
    run_name = "DDPG5"
    save_path = 'C:\\Users\\61602\\Desktop\\Coding\\nav_gym\\history\\'+run_name+'.txt'
    n_games = 1000
    epsilon = 0.7

    if_print = True
    best_score = env.reward_range[0]
    score_history = []
    cost_history = []
    success_history = []
    step_ctr = 0
    for i in range(n_games):
        epsilon -= (0.7-0.1)/n_games
        print("****** " + run_name + ": Episode ", str(i), " ********")
        observation = env.reset()
        print(observation)
        terminated = False
        score = 0.
        cost = 0.
        success = 0
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
                if r >= 40.0:
                    success = 1
            observation = observation_
            if if_print:
                print(" **** Episode: ",str(i), " ** Step: ", step_ctr)
                # print("1. action: ", action)
                print("2. reward: ",reward)
                # print("2. obs_: ", observation)
                # print("3. done: ", done)
            rc = -np.min(observation,axis=-1, keepdims=True)
            print("rc ", rc)
            for c in rc:
                cost += c[0]
        cost_history.append(cost)
        score_history.append(score)
        success_history.append(success)
        avg_score = np.mean(score_history[-100:])


        if os.path.isfile(save_path):
            os.remove(save_path)
        with open(save_path, "a") as f:
            for i in range(len(score_history)):
                f.write('{:}   {:}   {:} \n'. format(score_history[i],cost_history[i],success_history[i]))

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    # x = [i+1 for i in range(n_games)]
    # plot_learning_curve(x, score_history, figure_file)

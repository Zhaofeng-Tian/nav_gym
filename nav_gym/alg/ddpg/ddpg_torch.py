import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork, QMixer
from noise import OUActionNoise
from buffer import ReplayBuffer, Buffer, MABuffer
import random

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, n_agents, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, model_path = 'C:\\Users\\61602\\Desktop\\Coding\\corridor_model'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.model_path = model_path

        self.memory = Buffer(max_size, input_dims, n_actions, n_agents)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor', chkpt_dir= self.model_path)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic',chkpt_dir= self.model_path)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor',chkpt_dir= self.model_path)

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic',chkpt_dir= self.model_path)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, epsilon):
        if random.uniform(0,1) < epsilon:
            actions = []
            for i in range(len(observation)):
                a1 = random.uniform(-0.6,0.6)
                a2 = random.uniform(-0.6,0.6)
                actions.append(np.array([a1,a2]))
            print("Random Exploration!!!")

            return np.array(actions)
        self.actor.eval()
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # print("state:" ,state.shape)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        # print("Choosing Action: ",action)
        action = np.clip(action, np.array([-0.6, -0.6]), np.array([0.6, 0.6]))
        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        print(" Learning!~~~~ mem counter: ", self.memory.mem_cntr)
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        print(" Update network Parameters!")
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)





class Agent2():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, n_agents, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, model_path = 'C:\\Users\\61602\\Desktop\\Coding\\corridor_model'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.model_path = model_path
        self.obs_dim = input_dims
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.memory = MABuffer(max_size, input_dims, n_actions, n_agents)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor', chkpt_dir= self.model_path)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic',chkpt_dir= self.model_path)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor',chkpt_dir= self.model_path)

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic',chkpt_dir= self.model_path)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, epsilon):
        if random.uniform(0,1) < epsilon:
            actions = []
            for i in range(len(observation)):
                a1 = random.uniform(-0.6,0.6)
                a2 = random.uniform(-0.6,0.6)
                actions.append(np.array([a1,a2]))
            print("Random Exploration!!!")

            return np.array(actions)
        self.actor.eval()
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # print("state:" ,state.shape)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        # print("Choosing Action: ",action)
        action = np.clip(action, np.array([-0.6, -0.6]), np.array([0.6, 0.6]))
        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        print(" Learning!~~~~ mem counter: ", self.memory.mem_cntr)
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = np.reshape(states, (-1, self.obs_dim))
        states_ = np.reshape(states_, (-1, self.obs_dim))
        actions = np.reshape(actions, (-1, self.n_actions))
        rewards = np.reshape(rewards, (-1))
        done = np.reshape(done, (-1))
        #print("state shape: ", states.shape)
        #print(done)
        #assert done == 1, '!'
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done,dtype=bool).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        # print("target actions shape :", target_actions.shape)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        # print("critic_ shape :", critic_value_.shape)
        critic_value = self.critic.forward(states, actions)
        # print("critic shape :", critic_value.shape)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)
        # print("critic_ shape :", critic_value_.shape)

        target = rewards + self.gamma*critic_value_
        # print("target shape :", target.shape)
        target = target.view(self.batch_size* self.n_agents, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        print(" Update network Parameters!")
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)


class Agent_VDN():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, n_agents, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, model_path = 'C:\\Users\\61602\\Desktop\\Coding\\corridor_model'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.model_path = model_path
        self.obs_dim = input_dims
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.memory = MABuffer(max_size, input_dims, n_actions, n_agents)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor', chkpt_dir= self.model_path)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic',chkpt_dir= self.model_path)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor',chkpt_dir= self.model_path)

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic',chkpt_dir= self.model_path)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, epsilon):
        if random.uniform(0,1) < epsilon:
            actions = []
            for i in range(len(observation)):
                a1 = random.uniform(-0.6,0.6)
                a2 = random.uniform(-0.6,0.6)
                actions.append(np.array([a1,a2]))
            print("Random Exploration!!!")

            return np.array(actions)
        self.actor.eval()
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # print("state:" ,state.shape)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        # print("Choosing Action: ",action)
        action = np.clip(action, np.array([-0.6, -0.6]), np.array([0.6, 0.6]))
        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        print(" Learning!~~~~ mem counter: ", self.memory.mem_cntr)
        if self.memory.mem_cntr < self.batch_size:
            return

        obs, actions, rewards, obs_, done = \
                self.memory.sample_buffer(self.batch_size)


        # obs = np.reshape(obs, (-1, self.obs_dim))
        # obs_ = np.reshape(obs_, (-1, self.obs_dim))
        # actions = np.reshape(actions, (-1, self.n_actions))
        # rewards = np.reshape(rewards, (-1))
        # done = np.reshape(done, (-1))
        #print("state shape: ", obs.shape)
        #print(done)
        #assert done == 1, '!'
        a_obs = np.reshape(obs, (-1, self.obs_dim))
        a_obs_ = np.reshape(obs_, (-1, self.obs_dim))
        s = np.reshape(obs, (self.batch_size, self.n_agents* self.obs_dim))
        s_ = np.reshape(obs_, (self.batch_size, self.n_agents* self.obs_dim))
        a_actions = np.reshape(actions, (-1, self.n_actions))
        a_r = np.reshape(rewards,(self.batch_size*self.n_agents))
        a_done = np.reshape(done, (-1))


        a_obs = T.tensor(a_obs, dtype=T.float).to(self.actor.device)
        a_obs_ = T.tensor(a_obs_, dtype=T.float).to(self.actor.device)
        s = T.tensor(s, dtype=T.float).to(self.actor.device)   
        s_ = T.tensor(s_, dtype=T.float).to(self.actor.device)
        a_actions = T.tensor(a_actions, dtype=T.float).to(self.actor.device) 
        a_r = T.tensor(a_r,dtype=bool).to(self.actor.device)  
        a_done = T.tensor(a_done,dtype=bool).to(self.actor.device)     

        obs = T.tensor(obs, dtype=T.float).to(self.actor.device)
        obs_ = T.tensor(obs_, dtype=T.float).to(self.actor.device)       #(b,n,o)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device) #(b,n,a)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device) #(b,n,1)
        done = T.tensor(done,dtype=bool).to(self.actor.device)  # （b,n,1)



        target_actions = self.target_actor.forward(a_obs_) # (b*n, o) -> (b*n, a)
        print("target actions shape :", target_actions.shape) 
        critic_value_ = self.target_critic.forward(a_obs_, target_actions) # (b*n, o+a) -> (b*n, 1)
        print("critic_ shape :", critic_value_.shape)
        critic_value = self.critic.forward(a_obs, a_actions) # (b*n, o+a) -> (b*n, 1)
        print("critic shape :", critic_value.shape)

        critic_value_[a_done] = 0.0
        critic_value = critic_value.view(self.batch_size, self.n_agents)
        critic_value_ = critic_value_.view(self.batch_size, self.n_agents)
        # print(critic_value_)\
        mixed_critic = T.sum(critic_value, dim =1)
        mixed_critic_ = T.sum(critic_value_, dim= 1) # (b, 1)
        # print(mixed_critic_)
        # print("mixed critic_ shape: ", mixed_critic_.shape)
        assert mixed_critic.shape == (self.batch_size,), " Stop for view"
        assert mixed_critic_.shape == (self.batch_size,), " Stop for view"

        # print(rewards)
        sumed_r = T.sum(rewards, dim = 1)
        # print(sumed_r)
        # print("sumed_r shape: ", sumed_r.shape)
        assert sumed_r.shape == (self.batch_size,), " Stop for view"

        # critic_value_ = critic_value_.view(-1)
        # print("critic_ shape :", critic_value_.shape)

        target = sumed_r + self.gamma * mixed_critic_
        # target = rewards + self.gamma*critic_value_
        # target = a_r + self.gamma*critic_value_
        print(target)
        print("target shape :", target.shape)
        assert sumed_r.shape == (self.batch_size,), " Stop for view"
        # target = target.view(self.batch_size* self.n_agents, 1)


        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, mixed_critic)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        print(self.critic.forward(obs, self.actor.forward(obs)))
        actor_loss = -self.critic.forward(obs, self.actor.forward(obs))
        print("actor loss", actor_loss)
        actor_loss = T.sum(actor_loss, dim=1)
        print("actor loss", actor_loss)
        print("actor loss shape ", actor_loss.shape)
        

        actor_loss = T.mean(actor_loss)
        print("actor_loss: ", actor_loss.shape)

        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        print(" Update network Parameters!")
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)

class Agent_VDN2():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, n_agents, gamma=0.95,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, model_path = 'C:\\Users\\61602\\Desktop\\Coding\\corridor_model'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.model_path = model_path
        self.obs_dim = input_dims
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.memory = MABuffer(max_size, input_dims, n_actions, n_agents)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor', chkpt_dir= self.model_path)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic',chkpt_dir= self.model_path)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor',chkpt_dir= self.model_path)

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic',chkpt_dir= self.model_path)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, epsilon):
        if random.uniform(0,1) < epsilon:
            actions = []
            for i in range(len(observation)):
                a1 = random.uniform(-0.6,0.6)
                a2 = random.uniform(-0.6,0.6)
                actions.append(np.array([a1,a2]))
            print("Random Exploration!!!")

            return np.array(actions)
        self.actor.eval()
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # print("state:" ,state.shape)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        # print("Choosing Action: ",action)
        action = np.clip(action, np.array([-0.6, -0.6]), np.array([0.6, 0.6]))
        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        print(" Learning!~~~~ mem counter: ", self.memory.mem_cntr)
        if self.memory.mem_cntr < self.batch_size:
            return

        obs, actions, rewards, obs_, done = \
                self.memory.sample_buffer(self.batch_size)
        print(" Actor's device: ", self.actor.device)


        # obs = np.reshape(obs, (-1, self.obs_dim))
        # obs_ = np.reshape(obs_, (-1, self.obs_dim))
        # actions = np.reshape(actions, (-1, self.n_actions))
        # rewards = np.reshape(rewards, (-1))
        # done = np.reshape(done, (-1))
        #print("state shape: ", obs.shape)
        #print(done)
        #assert done == 1, '!'
        # a_obs = np.reshape(obs, (-1, self.obs_dim))
        # a_obs_ = np.reshape(obs_, (-1, self.obs_dim))
        # s = np.reshape(obs, (self.batch_size, self.n_agents* self.obs_dim))
        # s_ = np.reshape(obs_, (self.batch_size, self.n_agents* self.obs_dim))
        # a_actions = np.reshape(actions, (-1, self.n_actions))
        # a_r = np.reshape(rewards,(self.batch_size*self.n_agents))
        # a_done = np.reshape(done, (-1))


        # a_obs = T.tensor(a_obs, dtype=T.float).to(self.actor.device)
        # a_obs_ = T.tensor(a_obs_, dtype=T.float).to(self.actor.device)
        # s = T.tensor(s, dtype=T.float).to(self.actor.device)   
        # s_ = T.tensor(s_, dtype=T.float).to(self.actor.device)
        # a_actions = T.tensor(a_actions, dtype=T.float).to(self.actor.device) 
        # a_r = T.tensor(a_r,dtype=bool).to(self.actor.device)  
        # a_done = T.tensor(a_done,dtype=bool).to(self.actor.device)     

        obs = T.tensor(obs, dtype=T.float).to(self.actor.device)
        obs_ = T.tensor(obs_, dtype=T.float).to(self.actor.device)       #(b,n,o)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device) #(b,n,a)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device) #(b,n,1)
        done = T.tensor(done,dtype=bool).to(self.actor.device)  # （b,n)



        target_actions = self.target_actor.forward(obs_) # (b,n, o) -> (b,n, a)
        # print("target actions shape :", target_actions.shape) 
        critic_value_ = self.target_critic.forward(obs_, target_actions) # (b,n, o+a) -> (b,n, 1)
        # print("critic_ shape :", critic_value_.shape)
        critic_value = self.critic.forward(obs, actions) # (b,n, o+a) -> (b, n, 1)
        # print("critic_ ",critic_value_)
        # print("critic shape :", critic_value.shape)
        # print("done: ", done)
        # critic_value_[done] = 0.0
        # print("critic_value_", critic_value_)
        # if self.memory.mem_cntr >= 300:
        #     assert 5==2, 'stop to view done'
        # critic_value = critic_value.view(self.batch_size, self.n_agents)
        # critic_value_ = critic_value_.view(self.batch_size, self.n_agents)
        # print(critic_value_)\
        mixed_critic = T.sum(critic_value, dim =1)
        mixed_critic_ = T.sum(critic_value_, dim= 1) # (b, 1)
        mixed_rewards = T.sum(rewards, dim = 1)
        # print(mixed_critic)
        # print(mixed_critic_)
        # print("mixed critic_ shape: ", mixed_critic_.shape)
        # print("rewards: ", mixed_rewards)
        # print("reward shape: ", mixed_rewards.shape)
        mixed_critic = mixed_critic.view(self.batch_size)
        mixed_critic_ = mixed_critic_.view(self.batch_size)
        # print("mixed_critc", mixed_critic)
        # print("mixed_critic_" ,mixed_critic_)
        target = mixed_rewards+ self.gamma * mixed_critic_
        # print("target",target)
        # print("target shape :", target.shape)
        # assert target.shape == (self.batch_size,), " Stop for view"
        # target = target.view(self.batch_size* self.n_agents, 1)


        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, mixed_critic)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(obs, self.actor.forward(obs))
        actor_loss = T.sum(actor_loss, dim=1)      
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        print(" Update network Parameters!")
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)



class Agent_QMix():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, n_agents, gamma=0.95,
                 max_size=200000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, model_path = 'C:\\Users\\61602\\Desktop\\Coding\\corridor_model'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.model_path = model_path
        self.obs_dim = input_dims
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_shape = self.obs_dim*self.n_agents
        self.mix_dim = 256
        self.q_dim = 1

        self.memory = MABuffer(max_size, input_dims, n_actions, n_agents)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor', chkpt_dir= self.model_path)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic',chkpt_dir= self.model_path)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor',chkpt_dir= self.model_path)

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic',chkpt_dir= self.model_path)
        self.qmixer = QMixer(n_agents= self.n_agents, state_shape=self.state_shape,
                             mixing_embed_dim= self.mix_dim, q_embed_dim= self.q_dim,
                             name = 'qmixer', chkpt_dir= self.model_path)
        
        self.target_qmixer = QMixer(n_agents= self.n_agents, state_shape=self.state_shape,
                             mixing_embed_dim= self.mix_dim, q_embed_dim= self.q_dim,
                             name = 'target_qmixer', chkpt_dir= self.model_path)
        
        self.critic_params = list(self.critic.parameters())+ list(self.qmixer.parameters())
        self.critic_optimizer = T.optim.Adam(self.critic_params, lr=self.alpha)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, epsilon):
        if random.uniform(0,1) < epsilon:
            actions = []
            for i in range(len(observation)):
                a1 = random.uniform(-0.6,0.6)
                a2 = random.uniform(-0.6,0.6)
                actions.append(np.array([a1,a2]))
            print("Random Exploration!!!")

            return np.array(actions)
        self.actor.eval()
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # print("state:" ,state.shape)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        # print("Choosing Action: ",action)
        action = np.clip(action, np.array([-0.6, -0.6]), np.array([0.6, 0.6]))
        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        print(" Learning!~~~~ mem counter: ", self.memory.mem_cntr)
        if self.memory.mem_cntr < self.batch_size:
            return

        obs, actions, rewards, obs_, done = \
                self.memory.sample_buffer(self.batch_size)   

        obs = T.tensor(obs, dtype=T.float).to(self.actor.device)
        obs_ = T.tensor(obs_, dtype=T.float).to(self.actor.device)       #(b,n,o)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device) #(b,n,a)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device) #(b,n,1)
        done = T.tensor(done,dtype=bool).to(self.actor.device)  # （b,n)

        state = obs.view(self.batch_size, self.n_agents * self.obs_dim)
        state_ = obs_.view(self.batch_size, self.n_agents * self.obs_dim)

        target_actions = self.target_actor.forward(obs_) # (b,n, o) -> (b,n, a)
        critic_value_ = self.target_critic.forward(obs_, target_actions) # (b,n, o+a) -> (b,n, 1)
        critic_value = self.critic.forward(obs, actions) # (b,n, o+a) -> (b, n, 1)
        # mixed_critic = T.sum(critic_value, dim =1)
        # mixed_critic_ = T.sum(critic_value_, dim= 1) # (b, 1)
        mixed_critic = self.qmixer(critic_value, state)
        mixed_critic_ = self.target_qmixer(critic_value_, state_)
        # print("mixed_critic: ", mixed_critic)
        # print("mixed_critic_ shape: ", mixed_critic.shape)
        # assert mixed_critic_.shape == (self.batch_size, 1), "stop to view qmix shape"
        mixed_rewards = T.sum(rewards, dim = 1)
        mixed_critic = mixed_critic.view(self.batch_size)
        mixed_critic_ = mixed_critic_.view(self.batch_size)


        # print("mixed crtic: ", mixed_critic)
        # print("mixed rewards: ", mixed_rewards)
        
        assert mixed_critic_.shape == (self.batch_size, ), "stop to view qmix shape"
        target = mixed_rewards+ self.gamma * mixed_critic_
        # print("target: ", target)

        # qmix_p = dict(self.qmixer.named_parameters())
        # critic_p = dict(self.critic.named_parameters())
        # print("*******************************")
        # print("qmix_p:,",qmix_p)
        # print("critic_p : ", critic_p)

        critic_loss = F.mse_loss(target, mixed_critic)

        # self.critic.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # critic_loss = F.mse_loss(target, mixed_critic)
        critic_loss.backward()
        # self.critic.optimizer.step()
        self.critic_optimizer.step()

        # qmix_p_ = dict(self.qmixer.named_parameters()) 
        # critic_p_ = dict(self.critic.named_parameters())
        # print("*******************************")
        # print("qmix_p_," ,qmix_p_)
        # print("critic_p_ : ", critic_p)
        # assert 1==2, " view parameter if update"

        self.actor.optimizer.zero_grad()
        # actor_loss = -self.critic.forward(obs, self.actor.forward(obs))
        # actor_loss = T.sum(actor_loss, dim=1) 
        qs = self.critic.forward(obs, self.actor.forward(obs))
        # print("qs: ", qs)
        actor_loss = -self.qmixer(qs, state)  
        # print("actor loss: ", actor_loss)   
        actor_loss = T.mean(actor_loss)
        # print("actor mean loss: ",actor_loss)
        # assert 1==2, " check to view"
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        print(" Update network Parameters!")
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        qmixer_params = self.qmixer.named_parameters()
        target_qmixer_params = self.target_qmixer.named_parameters()
        qmixer_state_dict = dict(qmixer_params)
        target_qmixer_state_dict = dict(target_qmixer_params)
        # print("qmixer params: ", self.qmixer.named_parameters())
        # print("qmixer params dict: ", dict(self.qmixer.named_parameters()))
        # print("qmixer_params: " ,qmixer_params)
        # print("acotr state dict: ", dict(actor_params))
        # print("qmixer_state_dict: ", qmixer_state_dict)
        # assert 1==2,"!!!!!!!!!!!!!!!"

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        for name in qmixer_state_dict:
            qmixer_state_dict[name] = tau*qmixer_state_dict[name].clone() + \
                                 (1-tau)*target_qmixer_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        self.target_qmixer.load_state_dict(qmixer_state_dict)


class Agent_LagMix():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, n_agents, gamma=0.95,
                 max_size=200000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, model_path = 'C:\\Users\\61602\\Desktop\\Coding\\corridor_model'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.model_path = model_path
        self.obs_dim = input_dims
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_shape = self.obs_dim*self.n_agents
        self.mix_dim = 256
        self.q_dim = 1
        self.lag_lambda = 0.5

        self.memory = MABuffer(max_size, input_dims, n_actions, n_agents)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor', chkpt_dir= self.model_path)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic',chkpt_dir= self.model_path)
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic',chkpt_dir= self.model_path)

        self.cost = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic',chkpt_dir= self.model_path)
        
        self.target_cost = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic',chkpt_dir= self.model_path)

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor',chkpt_dir= self.model_path)


        self.qmixer = QMixer(n_agents= self.n_agents, state_shape=self.state_shape,
                             mixing_embed_dim= self.mix_dim, q_embed_dim= self.q_dim,
                             name = 'qmixer', chkpt_dir= self.model_path)
        
        self.target_qmixer = QMixer(n_agents= self.n_agents, state_shape=self.state_shape,
                             mixing_embed_dim= self.mix_dim, q_embed_dim= self.q_dim,
                             name = 'target_qmixer', chkpt_dir= self.model_path)
        
        self.critic_params = list(self.critic.parameters()) + list(self.cost.parameters()) +  list(self.qmixer.parameters())
        self.critic_optimizer = T.optim.Adam(self.critic_params, lr=self.alpha)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, epsilon):
        if random.uniform(0,1) < epsilon:
            actions = []
            for i in range(len(observation)):
                a1 = random.uniform(-0.6,0.6)
                a2 = random.uniform(-0.6,0.6)
                actions.append(np.array([a1,a2]))
            print("Random Exploration!!!")

            return np.array(actions)
        self.actor.eval()
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # print("state:" ,state.shape)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        # print("Choosing Action: ",action)
        action = np.clip(action, np.array([-0.6, -0.6]), np.array([0.6, 0.6]))
        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        print(" Learning!~~~~ mem counter: ", self.memory.mem_cntr)
        if self.memory.mem_cntr < self.batch_size:
            return

        obs, actions, rewards, obs_, done = \
                self.memory.sample_buffer(self.batch_size)   

        # d_obs = np.min(obs, axis=-1, keepdims=True)
        # d_obs_ = np.min(obs_, axis=-1, keepdims=True)

        cost = -np.min(obs, axis=-1, keepdims=True)
        cost_ = -np.min(obs_, axis=-1, keepdims=True)
        # print("d_obs", cost)
        # assert 1==2, " stop to view d_obs"

        obs = T.tensor(obs, dtype=T.float).to(self.actor.device)
        obs_ = T.tensor(obs_, dtype=T.float).to(self.actor.device)       #(b,n,o)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device) #(b,n,a)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device) #(b,n,1)
        done = T.tensor(done,dtype=bool).to(self.actor.device)  # （b,n)
        cost = T.tensor(cost, dtype=T.float).to(self.actor.device) # (b,n,1)
        cost_ = T.tensor(cost_, dtype=T.float).to(self.actor.device) # (b,n,1)

        state = obs.view(self.batch_size, self.n_agents * self.obs_dim)
        state_ = obs_.view(self.batch_size, self.n_agents * self.obs_dim)


        target_actions = self.target_actor.forward(obs_) # (b,n, o) -> (b,n, a)
        critic_value_ = self.target_critic.forward(obs_, target_actions) # (b,n, o+a) -> (b,n, 1)
        critic_value = self.critic.forward(obs, actions) # (b,n, o+a) -> (b, n, 1)

        cost_value_ = self.target_critic.forward(obs_, target_actions) # (b,n, o+a) -> (b,n, 1)
        cost_value = self.critic.forward(obs, actions) # (b,n, o+a) -> (b, n, 1)
        # print("Cirtic: ", critic_value)
        # print(" Cost: ", cost_value)
        # print("lamda* cost: ", self.lag_lambda * cost_value)
        # print("critci+ lambda* cost: ", critic_value + self.lag_lambda*cost_value)
        # assert 1==2, " stop to view d_obs"
        # mixed_critic = T.sum(critic_value, dim =1)
        # mixed_critic_ = T.sum(critic_value_, dim= 1) # (b, 1)
        obj_value = critic_value - self.lag_lambda*cost_value
        obj_value_ = critic_value_ - self.lag_lambda*cost_value_
        mixed_critic = self.qmixer(obj_value, state)
        mixed_critic_ = self.target_qmixer(obj_value_, state_)
        # print("mixed_critic: ", mixed_critic)
        # print("mixed_critic_ shape: ", mixed_critic.shape)
        # assert mixed_critic_.shape == (self.batch_size, 1), "stop to view qmix shape"
        # print("rewards shape",rewards.shape)
        # print("cost",cost_.shape)
        cost_ = cost_.view((self.batch_size, self.n_agents))
        obj_rewards = rewards - self.lag_lambda * cost_
        mixed_rewards = T.sum(obj_rewards, dim = 1)
        mixed_critic = mixed_critic.view(self.batch_size)
        mixed_critic_ = mixed_critic_.view(self.batch_size)


        # print("mixed crtic: ", mixed_critic)
        # print("mixed rewards: ", mixed_rewards)
        
        assert mixed_critic_.shape == (self.batch_size, ), "stop to view qmix shape"
        target = mixed_rewards+ self.gamma * mixed_critic_
        # print("target: ", target)

        # qmix_p = dict(self.qmixer.named_parameters())
        # critic_p = dict(self.critic.named_parameters())
        # print("*******************************")
        # print("qmix_p:,",qmix_p)
        # print("critic_p : ", critic_p)

        critic_loss = F.mse_loss(target, mixed_critic)

        # self.critic.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # critic_loss = F.mse_loss(target, mixed_critic)
        critic_loss.backward()
        # self.critic.optimizer.step()
        self.critic_optimizer.step()

        # qmix_p_ = dict(self.qmixer.named_parameters()) 
        # critic_p_ = dict(self.critic.named_parameters())
        # print("*******************************")
        # print("qmix_p_," ,qmix_p_)
        # print("critic_p_ : ", critic_p)
        # assert 1==2, " view parameter if update"

        self.actor.optimizer.zero_grad()
        # actor_loss = -self.critic.forward(obs, self.actor.forward(obs))
        # actor_loss = T.sum(actor_loss, dim=1) 
        qs = self.critic.forward(obs, self.actor.forward(obs))
        # print("qs: ", qs)
        actor_loss = -self.qmixer(qs, state)  
        # print("actor loss: ", actor_loss)   
        actor_loss = T.mean(actor_loss)
        # print("actor mean loss: ",actor_loss)
        # assert 1==2, " check to view"
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        print(" Update network Parameters!")
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        qmixer_params = self.qmixer.named_parameters()
        target_qmixer_params = self.target_qmixer.named_parameters()
        qmixer_state_dict = dict(qmixer_params)
        target_qmixer_state_dict = dict(target_qmixer_params)
        # print("qmixer params: ", self.qmixer.named_parameters())
        # print("qmixer params dict: ", dict(self.qmixer.named_parameters()))
        # print("qmixer_params: " ,qmixer_params)
        # print("acotr state dict: ", dict(actor_params))
        # print("qmixer_state_dict: ", qmixer_state_dict)
        # assert 1==2,"!!!!!!!!!!!!!!!"

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        for name in qmixer_state_dict:
            qmixer_state_dict[name] = tau*qmixer_state_dict[name].clone() + \
                                 (1-tau)*target_qmixer_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        self.target_qmixer.load_state_dict(qmixer_state_dict)
import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork, QMixer, CriticNetwork_Concat
from noise import OUActionNoise
from buffer import MABuffer
import random

T.autograd.set_detect_anomaly(True)
class Agent():
    def __init__(self, alpha, beta, obs_dim, tau, n_actions, n_agents, gamma=0.95,
                 max_size=1000000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, 
                 mix_dim = 512,
                 q_dim = 1,                 
                 model_path = 'C:\\Users\\61602\\Desktop\\Coding\\corridor_model'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.model_path = model_path
        self.obs_dim = obs_dim
        self.n_actions = n_actions # 2 actions for robot tasks
        self.n_agents = n_agents
        self.state_shape = self.obs_dim * self.n_agents
        self.mix_dim = mix_dim
        self.q_dim = q_dim # dim 1 for continuous tasks

        self.memory = MABuffer(max_size, obs_dim, n_actions, n_agents)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, obs_dim, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor', chkpt_dir= self.model_path)
        # self.critic = CriticNetwork(beta, obs_dim, fc1_dims, fc2_dims,
        #                         n_actions=n_actions, name='critic',chkpt_dir= self.model_path)
        self.critic = CriticNetwork_Concat(beta, obs_dim, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic',chkpt_dir= self.model_path)

        self.target_actor = ActorNetwork(alpha, obs_dim, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor',chkpt_dir= self.model_path)

        self.target_critic = CriticNetwork_Concat(beta, obs_dim, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic',chkpt_dir= self.model_path)
        
        # self.qmixer = QMixer(n_agents= self.n_agents, state_shape=self.state_shape,
        #                      mixing_embed_dim= self.mix_dim, q_embed_dim= self.q_dim,
        #                      name = 'qmixer', chkpt_dir= self.model_path)
        
        # self.target_qmixer = QMixer(n_agents= self.n_agents, state_shape=self.state_shape,
        #                      mixing_embed_dim= self.mix_dim, q_embed_dim= self.q_dim,
        #                      name = 'target_qmixer', chkpt_dir= self.model_path)



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
        else:
            return self.sample_actions(observation)
        
    def sample_actions(self, observation):
        self.actor.eval()
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        print("11111111111111111111 state:" ,state.shape)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()
        action = mu_prime.cpu().detach().numpy()
        print("Choosing Action: ",action)
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
        obs_ = T.tensor(obs_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        assert obs.shape == (self.batch_size, self.n_agents, self.obs_dim), " Obs dim worng when building state !"
        # state = obs.view(self.batch_size, self.n_agents * self.obs_dim).to(self.actor.device)
        # state_ = obs_.view(self.batch_size, self.n_agents * self.obs_dim).to(self.actor.device)
        obs_c = obs.clone()
        obs_c = obs_c.view(self.batch_size*self.n_agents, self.obs_dim).to(self.actor.device)
        obs_c_ = obs_.clone()
        obs_c_ = obs_c_.view(self.batch_size*self.n_agents, self.obs_dim).to(self.actor.device)
        actions_c = actions.clone()
        actions_c = actions.view(self.batch_size*self.n_agents, self.n_actions).to(self.actor.device)
        # obs shape (b, n, o)  batch_size, n_agents, obs_dim
        # actor taking input: o
        # state shape: concat all obs to a flatten vector
        # = [o1, o2, ..., on], with a shape (b, n*o)
        """
        1. Sample target actions (b, n, a)
        """
        
        target_actions = self.sample_actions(obs_c_)
        t_actions = T.tensor(target_actions.astype(np.float32)).to(self.actor.device)
        # target_actions = target_actions.view(self.batch_size, self.n_agents, self.n_actions)
        assert target_actions.shape == (self.batch_size * self.n_agents, self.n_actions), " Wrong action dim !"
        """
        2. Compute agents Qs (b, n, q)
        """
        
        agents_qs = self.critic.forward(obs_c,
                                       actions_c)
        agents_qs = agents_qs.view(self.batch_size, self.n_agents, self.q_dim)
        # print("~~~~~~~~~~~~~~~agemts qs shape: ", agents_qs)
        assert agents_qs.shape == (self.batch_size, self.n_agents, self.q_dim), " Wrong Q dim !"

        """
        3. Compute target agents Qs (b, n, q)
        """
        target_agents_qs = self.target_critic.forward(obs_c_,
                                       t_actions)
        target_agents_qs = target_agents_qs.view(self.batch_size, self.n_agents, self.q_dim)
        assert target_agents_qs.shape == (self.batch_size, self.n_agents, self.q_dim), " Wrong Q dim !"

        """
        4. Mixed Q (b, mq) batch_size, mixed Q dim = 1 here
        """
        # mixed_qs =  self.qmixer.forward(agents_qs, state)
        mixed_qs = T.sum(agents_qs, dim=1, keepdim=True)
        # mixed_qs = mixed_qs.view(self.batch_size, -1)
        # assert mixed_qs.shape == (self.batch_size,  self.q_dim), " Wrong mixed Q dim !"

        """
        5. Target mixed Q (b, mq) 
        """
        # target_mixed_qs =  self.target_qmixer.forward(target_agents_qs, state_)
        target_mixed_qs = T.sum(target_agents_qs, dim=1, keepdim=True)
        # target_mixed_qs = target_mixed_qs.view(self.batch_size, self.q_dim) 
        # print((self.batch_size,  self.q_dim))
        print("~~~~~~~~~~~~~~~~~~~mixed Q dims: ", target_mixed_qs.shape)
        assert target_mixed_qs.shape == (self.batch_size, 1, self.q_dim), " Wrong mixed Q dim !"


        """
        6.  Optimize critic
        """
        # critic_loss = F.mse_loss(target_mixed_qs, mixed_qs)
        critic_loss = F.mse_loss(target_agents_qs, agents_qs)
        self.critic.optimizer.zero_grad()
        # critic_loss = F.mse_loss(target_mixed_qs, mixed_qs)
        # critic_grad_norm = T.nn.utils.clip_grad_norm_(list(self.critic.parameters()), 0.5)
        # critic_loss.backward(retain_graph=True)
        # critic_loss.backward(retain_graph=True)
        critic_loss.backward()
        self.critic.optimizer.step()


        """
        7. Actor training
        """
        sample_actions = self.sample_actions(obs_c)
        # sample_actions = self.actor.forward(obs_c)
        t_actions = T.tensor(sample_actions.astype(np.float32)).to(self.actor.device)
        # sample_actions = sample_actions.view(self.batch_size, self.n_agents, self.n_actions)
        qs = self.critic.forward(obs_c,
                                       t_actions)
        qs = agents_qs.view(self.batch_size, self.n_agents, self.q_dim)
        m_qs =  T.sum(qs, dim=1, keepdim=True)
        # m_qs = m_qs.view(self.batch_size, self.q_dim) 




        """
        8. Optimize Actor
        """
        # actor_loss = T.mean(-m_qs)
        actor_loss = T.mean(qs)
        self.actor.optimizer.zero_grad()

        # actor_loss.backward(retain_graph=True)
        actor_loss.backward()
        # agent_grad_norm = T.nn.utils.clip_grad_norm_(list(self.actor.parameters()), 0.5)
        self.actor.optimizer.step()


        """
        9. Update network parameters
        """
        self.update_network_parameters()



        # self.cr
        
        # target_actions = self.target_actor.forward(obs_)
        # critic_value_ = self.target_critic.forward(obs_, target_actions)
        # critic_value = self.critic.forward(obs, actions)

        # critic_value_[done] = 0.0
        # critic_value_ = critic_value_.view(-1)

        # target = rewards + self.gamma*critic_value_
        # target = target.view(self.batch_size, 1)

        # self.critic.optimizer.zero_grad()
        # critic_loss = F.mse_loss(target, critic_value)
        # critic_loss.backward()
        # self.critic.optimizer.step()

        # self.actor.optimizer.zero_grad()
        # actor_loss = -self.critic.forward(obs, self.actor.forward(obs))
        # actor_loss = T.mean(actor_loss)
        # actor_loss.backward()
        # self.actor.optimizer.step()

        self.update_network_parameters()



    def update_network_parameters(self, tau=None):
        print(" Update network Parameters!")
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        # qmixer_params = self.qmixer.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        # target_qmixer_params = self.target_qmixer.named_parameters()
    

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        # qmixer_state_dict = dict(qmixer_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)
        # target_qmixer_state_dict = dict(target_qmixer_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()
        
        # for name in qmixer_state_dict:
        #     qmixer_state_dict[name] = tau*qmixer_state_dict[name].clone() + \
        #                          (1-tau)*target_qmixer_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        # self.target_qmixer.load_state_dict(qmixer_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)






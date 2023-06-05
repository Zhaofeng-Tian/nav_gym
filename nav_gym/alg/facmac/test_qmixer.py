from networks import QMixer
import numpy as np
import torch as T  
import torch.nn as nn

n_agents = 2
obs_shape = (2,)
state_shape= (4,)
q_embed_dim= 1
bs = 3


qmixer = QMixer(n_agents= n_agents, state_shape= state_shape, mixing_embed_dim = 256, q_embed_dim= q_embed_dim)

# a = np.array([[3,2],[4,5]])
# print(a)
# print(T.tensor(a))

states = T.rand(bs, *state_shape).to(qmixer.device)
print(states)
agent_qs = T.rand(bs, n_agents, q_embed_dim).to(qmixer.device)
print(agent_qs)
agent_qs = agent_qs.view(-1,1,n_agents*q_embed_dim)
print(agent_qs)
# qmixer.
y = qmixer(agent_qs, states)
print(" The output is  ^ o ^: ", y)
print(" y shape ", y.shape)

# # assert y.shape == (1,1), "dim worng lo"

# a = T.rand(3,2,4)

# print(a)
# actions = []
# for i in range(3):
#     actions.append(a[i])
# print(actions)
# # print(a[0])
# # print(a[0].shape)
# stacked_actions = T.stack(actions, dim = 0)
# print("stacked")
# print(stacked_actions)

# print("tensor~~~~~~~~")

a = T.rand(2,2,4)
print("a" ,a)
b = np.array([[[2,2],[2,2],[2,2],[2,2]],[[2,2],[2,2],[2,2],[2,2]]],dtype=np.float32)
b = T.tensor(b)
print("b",b)
c = T.bmm(a, b)
print("c ", c)

d = T.mm(a.view(-1, 4), b[0])
print("d" ,d)

print("d view back")
print (d.view(2,2,2))
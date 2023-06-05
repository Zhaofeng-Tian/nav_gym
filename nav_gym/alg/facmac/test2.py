import torch
import torch.nn as nn

# Define a simple neural network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Create an instance of the network
model = MyNetwork()

# Iterate over the named parameters
for name, param in model.named_parameters():
    print(name, param.shape)
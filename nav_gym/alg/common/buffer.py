import numpy as np

class Buffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class Buffer2():
    def __init__(self, max_size, o1_shape, o2_shape, n_actions, stack_number):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.o1_memory = np.zeros((self.mem_size, *o1_shape))
        self.new_o1_memory = np.zeros((self.mem_size, *o1_shape))
        self.o2_memory = np.zeros((self.mem_size, *o2_shape))
        self.new_o2_memory = np.zeros((self.mem_size, *o2_shape))

        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.stack_number = stack_number

    def store_transition(self, o1, o2, action, reward, o1_, o2_, done):
        index = self.mem_cntr % self.mem_size
        if index + self.stack_number > self.mem_size:
            self.o1_memory[index:] = o1[:self.mem_size-index]
            self.o2_memory[index:] = o2[:self.mem_size-index]
            self.action_memory[index:] = action[:self.mem_size-index]
            self.reward_memory[index:] = reward[:self.mem_size-index]
            self.new_o1_memory[index:] = o1_[:self.mem_size-index]
            self.new_o2_memory[index:] = o2_[:self.mem_size-index]
            self.terminal_memory[index] = done[:self.mem_size-index]
            self.mem_cntr += self.mem_size-index
        else:
            self.o1_memory[index:index+self.stack_number] = o1
            self.o2_memory[index:index+self.stack_number] = o2
            self.action_memory[index:index+self.stack_number] = action
            self.reward_memory[index:index+self.stack_number] = reward
            self.new_o1_memory[index:index+self.stack_number] = o1_
            self.new_o2_memory[index:index+self.stack_number] = o2_
            self.terminal_memory[index:index+self.stack_number] = done

            self.mem_cntr += self.stack_number

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        o1 = self.o1_memory[batch]
        o2 = self.o2_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        o1_ = self.new_o1_memory[batch]
        o2_ = self.new_o2_memory[batch]
        dones = self.terminal_memory[batch]

        return o1,o2, actions, rewards, o1_, o2_,dones
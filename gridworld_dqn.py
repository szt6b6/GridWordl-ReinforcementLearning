"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from gridlWorldEnv_dpn import GridWorldEnv

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(25, 256)
        self.out = nn.Linear(256, N_ACTIONS)

    def forward(self, x):
        # x = torch.nn.functional.one_hot(x, num_classes=5).view(x.shape[0], 5 * N_STATES).float()
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 3))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r], s_, done))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, N_STATES+2:-1])
        done_ = torch.FloatTensor(b_memory[:, -1:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * (1-done_)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train(dqn):
    for i_episode in range(EPOCHS):
        s = env.reset()
        ep_r = 0
        while True:
            if(np.random.rand(1) < EPSILON):
                a = dqn.choose_action(s)
            else:
                a = env.action_space.sample()

            # take action
            s_, r, done, info = env.step(a)

            dqn.store_transition(s, a, r, s_, done)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                        '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            s = s_

def choose_action(x, net):
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    # input only one sample
    if np.random.uniform() < EPSILON:   # greedy
        actions_value = net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
    else:   # random
        action = np.random.randint(0, N_ACTIONS)
        action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
    return action


def test():
    env.render_mode = "human"
    print('\ntesting...')
    net = torch.load("net_eval.pth")

    s = env.reset()
    for i in range(50):
        a_r = 0
        while True:
            env.render()
            a = choose_action(s, net)

            # take action
            s, r, done, info = env.step(a)
            a_r += r
            if(done):
                print("第 %d 次游戏, 累计奖励 %.2f" % (i, a_r))
                s = env.reset()
                break
    env.close()

# Hyper Parameters
BATCH_SIZE = 256
LR = 0.0005                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 10000
env = GridWorldEnv(size=5)
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = 25
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
EPOCHS = 8000

dqn = DQN()
train(dqn)
torch.save(dqn.target_net, "net.pth")
torch.save(dqn.eval_net, "net_eval.pth")
test()
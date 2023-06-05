import numpy as np
from gridlWorldEnv_qlearning import GridWorldEnv

epsilon = 0.9       
gamma = 0.95         
alpha = 0.01

env = GridWorldEnv(size=5) 
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 4 actions
N_STATES = 5 * 5 * 5 * 5  # 对应坐标x1*125+y1*25+x2*5+y2序号

Q = np.zeros([N_STATES, N_ACTIONS]) 

for epoch in range(1,4000):
    s = env.reset()[0]
    while True:
        # env.render()
        s_num = s[0] * 125 + s[1] * 25 + s[2] * 5 + s[3]
        if(np.random.rand(1) < epsilon):
            action = np.argmax(Q[s_num])
        else:
            action = env.action_space.sample()
        s_new, r, done, truncted, info = env.step(action)

        s_new_num = s_new[0] * 125 + s_new[1] * 25 + s_new[2] * 5 + s_new[3]
        Q[s_num, action] += alpha * (r + gamma * np.max(Q[s_new_num]) - Q[s_num, action])

        s = s_new
        if(done):
            break

# human render test
env.render_mode = "human"
for i in range(100):
    s= env.reset()[0]
    while True: 
        env.render()
        s_num = s[0] * 125 + s[1] * 25 + s[2] * 5 + s[3]
        action = np.argmax(Q[s_num])
        s_, r, done, chuncked, info = env.step(action) #take step using selected action
        if(done):
            break
            
        s = s_
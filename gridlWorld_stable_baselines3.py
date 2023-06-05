# 核心方法
# step, reset, render, 

# 属性
# action_space, observation_space, reward_range

# wrapper
# 允许子类重写step和reset等方法， observationWrapper, rewardWrapper, actionWrapper



import gym
from gym import spaces
import pygame
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class GridWorldEnv(gym.Env):
    metadata = {"render_modes" : ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super(GridWorldEnv).__init__()

        self.size = size # gird size
        self.window_size = 512 # game panel size
        self.current_step = 0 # current step
        self.max_step = 100

        #observation
        self.observation_space = spaces.Box(low=0, high=2, shape=(1, self.size, self.size), dtype=np.int32)
        # action
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0 : np.array([1, 0]),
            1 : np.array([0, 1]),
            2 : np.array([-1, 0]),
            3 : np.array([0, -1])
        }
                                                 
        assert render_mode is None or render_mode == render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        # reward
        self.window = None
        self.clock = None


    def _get_obs(self):
        map = np.zeros(shape=(1, self.size, self.size), dtype=np.int32)
        map[0][self._agent_location[0]][self._agent_location[1]] = 128
        map[0][self._target_location[0]][self._target_location[1]] = 255
        return map
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    
    def reset(self):
        self._agent_location = np.random.randint(0, self.size-1, size=2)
        self._target_location = np.random.randint(0, self.size-1, size=2)
        # self._target_location = np.array([2, 2], dtype=np.int)
        
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = np.random.randint(0, self.size-1, size=2)
        
        observation = self._get_obs()

        return observation
    

    def step(self, action):
        self.current_step += 1

        if self.current_step > self.max_step:
            self.current_step = 0
            return self._get_obs(), -10, True, self._get_info()

        direction = self._action_to_direction[action]

        prev_dis = np.linalg.norm(self._agent_location - self._target_location)

        self._agent_location = self._agent_location + direction
        
        self._agent_location = np.clip(self._agent_location, 0, self.size-1)
        step_to_closer = np.linalg.norm(self._agent_location - self._target_location) < prev_dis
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        reward = 0.1 if step_to_closer else -0.2

        if terminated:
            self.current_step = 0
            reward = 1

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward,  terminated, info
                                              
    def render(self, mode="human"):
        return self._render_frame()
        
    def _render_frame(self):
        if(self.window is None and self.render_mode == "human"):
            pygame.init()
            pygame.display.init()

            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if(self.clock is None and self.render_mode == "human"):
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))

        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.size) # 每个格子大小

        pygame.draw.rect(
            canvas,
            (255,0,0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size)
            )
        )# draw target

        pygame.draw.circle(
            canvas,
            (0,0,255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3
        )# draw agent

        for x in range(self.size+1):
            pygame.draw.line(
                canvas,
                0,
                (0, x * pix_square_size),
                (self.window_size, x * pix_square_size),
                width=3
            )

            pygame.draw.line(
                canvas,
                0,
                (x * pix_square_size, 0),
                (x * pix_square_size, self.window_size),
                width=3
            ) # vertivcal and horizontal line between grids

        if(self.render_mode == "human"):
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

        else: # rgb_array render
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2)
            )
    
    def close(self):
        if(self.window is not None):
            pygame.display.quit()
            pygame.quit()
        return super().close()


if __name__ == "__main__":
    size = 5
    env = GridWorldEnv("human", size)

    # 把环境向量化，如果有多个环境写成列表传入DummyVecEnv中，可以用一个线程来执行多个环境，提高训练效率
    env = DummyVecEnv([lambda : env])
    # 定义一个DQN模型，设置其中的各个参数
    model = DQN(
        "MlpPolicy",                                # MlpPolicy定义策略网络为MLP网络
        env=env, 
        learning_rate=5e-4,
        batch_size=256,
        buffer_size=10000,
        learning_starts=0,
        target_update_interval=100,
        policy_kwargs={"net_arch" : [256, 256]},     # 这里代表隐藏层为2层256个节点数的网络
        verbose=1,                                   # verbose=1代表打印训练信息，如果是0为不打印，2为打印调试信息
        tensorboard_log="./tensorboard/gridworld/"  # 训练数据保存目录，可以用tensorboard查看
    )
    # model = PPO("MlpPolicy", env, verbose=1)
    # model = PPO.load("./model/gridworld_PPO.pkl")
    model = DQN.load("./model/gridworld_DQN.pkl")
    # 开始训练
    # model.learn(total_timesteps=1e5)
    # 策略评估
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, render=True)
    #env.close()
    print("mean_reward:",mean_reward,"std_reward:",std_reward)
    # 保存模型到相应的目录
    # model.save("./model/gridworld_DQN.pkl")

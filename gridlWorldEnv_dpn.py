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


class GridWorldEnv(gym.Env):
    metadata = {"render_modes" : ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super(GridWorldEnv).__init__()

        self.size = size # gird size
        self.window_size = 512 # game panel size
        self.current_step = 0 # current step
        self.max_step = 100

        #observation
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=np.int32)
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
        map = np.zeros(shape=(self.size, self.size), dtype=np.int32)
        map[self._agent_location[0]][self._agent_location[1]] = 128
        map[self._target_location[0]][self._target_location[1]] = 255
        map = map.reshape(self.size * self.size)
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

    s = env.reset()
    done = False

    while not done:
        env.render()

        action = env.action_space.sample()

        s, r, done, info = env.step(action)
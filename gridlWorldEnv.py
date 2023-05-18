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
    metadata = {"render_modes" : ["human", "rgn_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super(GridWorldEnv).__init__()

        self.size = size # gird size
        self.window_size = 512 # game panel size
        

        #observation
        self.observation_space = spaces.Dict(
            {
                "agent" : spaces.Box(0, size-1, shape=(2,), dtype=int), 
                "target" : spaces.Box(0, size-1, shape=(2,), dtype=int)
            }
        )

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
        return np.hstack((self._agent_location, self._target_location))
    
    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    
    
    def reset(self, seed=None, options=None):
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        # self._target_location = np.array([2, 2], dtype=np.int)
        
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        observation = self._get_obs()
        info = self._get_info()

        if(self.render_mode == "human"):
            self._render_frame()

        return observation, info
    

    def step(self, action):
        direction = self._action_to_direction[action]

        step_to_closer = np.sum(np.power(self._agent_location - self._target_location, 2)) >= np.sum(np.power(self._agent_location + direction - self._target_location, 2))

        self._agent_location = self._agent_location + direction
        outofmap = False
        if(self._agent_location[0] < 0 or self._agent_location[0] >= self.size or 
           self._agent_location[1] < 0 or self._agent_location[1] >= self.size):
            outofmap = True
        self._agent_location = np.clip(self._agent_location, 0, self.size-1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        reward = 0 if step_to_closer else -1
        if outofmap:
            reward -= 10
        if terminated:
            reward += 10

        observation = self._get_obs()
        info = self._get_info()
        
        if(self.render_mode == "human"):
            self._render_frame()
        
        return observation, reward, terminated, False, info
                                              
    def render(self):
        if self.render_mode == "rgb_array":
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
    env.reset(10)
    for _ in range(1000):
        action = env.np_random.integers(0, 4)
        observation, reward, terminated, truncked, info = env.step(action)

        if(terminated):
            env.reset(10)
        
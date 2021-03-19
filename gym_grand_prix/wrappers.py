import gym
from gym import spaces


class GrandPrixWrapper(gym.RewardWrapper):
    def __init__(self, env, steps_per_episode=2000, fine=0., nrays=5):
        self.steps_per_episode = steps_per_episode
        self.fine = fine
        self.scale = (1, 1, *(0.4,)*nrays)
        self.possible_actions = ((0, 0), (1, .75), (-1, .75), (0, .75), (0, -.75))
        self.steps = 0
        self.env = env
        self.action_space = spaces.Discrete(len(self.possible_actions))
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(self.possible_actions[action])
        self.steps += 1
        if self.steps == self.steps_per_episode:
            self.steps = 0
            done = True
        if 'collision' in info and info['collision']:
            reward -= self.fine
        return observation * self.scale, reward, done, info

    def reward(self, reward):
        return reward

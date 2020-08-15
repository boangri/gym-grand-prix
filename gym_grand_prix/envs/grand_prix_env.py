import gym
import numpy as np
import pygame
import random

from cmath import rect, pi, phase

from gym import error, spaces, utils
from gym.utils import seeding

from gym_grand_prix.envs.cars.track import generate_map
from gym_grand_prix.envs.cars.world import SimpleCarWorld
from gym_grand_prix.envs.cars.agent import SimpleCarAgent
from gym_grand_prix.envs.cars.physics import SimplePhysics


class GrandPrixEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.nrays = 5
        self.action_space = spaces.Box(low=np.array([-1., -.75]),
                                       high=np.array([1., .75]),
                                       dtype=np.float32)  # steer, gas, brake
        self.observation_space = spaces.Box(low=np.array([0., -1., 0., 0., 0., 0., 0.]),
                                            high=np.array([100., 1., 20., 20., 20., 20., 20.]), dtype=np.float32)
        # self.seed()
        seed = 3
        np.random.seed(seed)
        random.seed(seed)
        m = generate_map(8, 5, 3, 3)
        self.world = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, window=True, timedelta=0.2)
        self.reset()
        if self.world.visual:
            self.scale = self.world._prepare_visualization()
        # self.world.run(steps=2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        return self.world.step(action[0], action[1])

    def reset(self):
        self.world.set_agents(agent_class=SimpleCarAgent)

    def render(self, mode='human', close=False):
        if self.world.visual:
            self.world.visualize(self.scale)
            if self.world._update_display() == pygame.QUIT:
                self.world.done = True

    def close(self):
        pass

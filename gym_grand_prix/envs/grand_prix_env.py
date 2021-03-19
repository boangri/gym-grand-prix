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
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    def __init__(self):
        self.nrays = 5
        self.seed = 3
        self.steps = 0  # means infinity
        self.display = False
        self.action_space = spaces.Box(low=np.array([-1., -.75]),
                                       high=np.array([1., .75]),
                                       dtype=np.float32)  # steer, gas, brake
        self.observation_space = spaces.Box(low=np.array([0., -1., 0., 0., 0., 0., 0.]),
                                            high=np.array([100., 1., 20., 20., 20., 20., 20.]), dtype=np.float64)
        self.world = None
        self.scale = None
        self.setOptions({})

    def setOptions(self, options):
        if 'nrays' in options:
            self.nrays = options['nrays']
        if 'seed' in options:
            self.seed = options['seed']
        if 'steps' in options:
            self.steps = options['steps']
        if 'display' in options:
            self.display = True
        np.random.seed(self.seed)
        random.seed(self.seed)
        m = generate_map(8, 5, 3, 3)
        self.world = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, window=self.display, timedelta=0.2)
        self.world.nrays = self.nrays
        self.world.steps = self.steps
        self.world.set_agents(agent_class=SimpleCarAgent)
        if self.display:
            self.world.visual = True
            self.scale = self.world._prepare_visualization()
        else:
            self.world.visual = False

    def step(self, action):
        return self.world.step(action[0], action[1])

    def reset(self, options=None):
        if options is not None:
            self.setOptions(options)
        # self.world.set_agents([SimpleCarAgent()])
        a = self.world.agents[0]
        vision = self.world.vision_for(a)
        return np.array(vision)

    def render(self, mode='human', close=False):
        if self.world.visual:
            self.world.visualize(self.scale)
            if self.world._update_display() == pygame.QUIT:
                self.world.done = True
            a = pygame.surfarray.array3d(pygame.display.get_surface())
            return np.transpose(a, (1, 0, 2))

    def close(self):
        # print("closing display window")
        self.world.quit()
        pygame.quit()

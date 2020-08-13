import gym
import numpy as np
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
        print("creating world...")
        self.nrays = 5
        self.action_space = spaces.Box(np.array([-1., 0, 1.]),
                                       np.array([-.75, 0, .75]),
                                       dtype=np.float64)  # steer, gas/brake

        self.observation_space = spaces.Box(low=-1., high=20., shape=(2 + self.nrays,), dtype=np.float32)
        m = generate_map(8, 5, 3, 3)
        self.world = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, window=True, timedelta=0.2)
        self.seed()
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
        print("reset env")
        pos = (self.world.map[0][0] + self.world.map[0][1]) / 2
        vel = 0
        heading = rect(-0.3, 1)
        print(pos, heading)
        pass

    def render(self, mode='human', close=False):
        print("render the picture")
        pass

    def close(self):
        print("closing the window")
        pass

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class GrandPrixEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print("creating world...")
        pass

    def step(self, action):
        print("making a step")
        pass

    def reset(self):
        print("reset env")
        pass

    def render(self, mode='human', close=False):
        print("render the picture")
        pass

    def close(self):
        print("closing the window")
        pass

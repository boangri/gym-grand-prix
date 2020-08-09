from gym.envs.registration import register

register(
    id='GrandPrix-v0',
    entry_point='gym_grand_prix.envs:GrandPrixEnv',
)

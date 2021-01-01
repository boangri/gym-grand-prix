from gym.envs.registration import register

print("GrandPrix-v0 version 0.1.2 1.1.2021")
register(
    id='GrandPrix-v0',
    entry_point='gym_grand_prix.envs:GrandPrixEnv',
)

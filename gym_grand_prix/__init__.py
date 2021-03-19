from gym.envs.registration import register

print("GrandPrix-v0 version 0.2.2 19.3.2021")
register(
    id='GrandPrix-v0',
    entry_point='gym_grand_prix.envs:GrandPrixEnv',
)

from gym.envs.registration import register

print("GrandPrix-v0 version 0.0.3")
register(
    id='GrandPrix-v0',
    entry_point='gym_grand_prix.envs:GrandPrixEnv',
)

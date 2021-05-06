from gym.envs.registration import register

register(
    id='SpringMass-v0',
    entry_point='gym_SpringMass.envs:SpringMassDiscrete',
)

register(
    id='SpringMassCont-v0',
    entry_point='gym_SpringMass.envs:SpringMassContinuous',
)

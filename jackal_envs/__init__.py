from gym.envs.registration import register
from jackal_envs.gazebo_simulation import GazeboSimulation
from jackal_envs.navigation_stack import NavigationStack
from jackal_envs import jackal_env_wrapper

register(
    id='jackal_discrete-v0',
    entry_point='jackal_envs.jackal_env_discrete:JackalEnvDiscrete',
)

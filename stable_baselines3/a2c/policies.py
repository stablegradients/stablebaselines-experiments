# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy, ActorCriticTransformerPolicy
from stable_baselines3.common.torch_layers import Transformer

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
TransformerPolicy = ActorCriticTransformerPolicy

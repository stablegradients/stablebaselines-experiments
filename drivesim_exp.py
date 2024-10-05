import time
import os
import torch
import gymnasium as gym
import argparse
from typing import Any, Dict
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from torchdriveenv.gym_env import HomogeneousWrapper
from torchdriveenv.env_utils import load_default_train_data, load_default_validation_data
from torchdriveenv.gym_env import GymEnv
from common import BaselineAlgorithm, load_rl_training_config

# Load training and validation data for the environment
training_data = load_default_train_data()
validation_data = load_default_validation_data()

# Set API key for the environment's simulator
os.environ["IAI_API_KEY"] = "ae81W7v2Dt8PFdU3aTZE76uajvdOCnf06oZBPVeU"

class EvalNTimestepsCallback(BaseCallback):
    """
    A callback for evaluating the RL model at specific intervals during training.

    Parameters:
    -----------
    eval_env: The environment used for evaluation.
    n_steps: Number of timesteps between two evaluations.
    eval_n_episodes: Number of episodes to evaluate at each step.
    deterministic: Whether to use deterministic actions during evaluation.
    log_tab: The log category for recording the results.
    """
    def __init__(self, eval_env, n_steps: int, eval_n_episodes: int, deterministic=False, log_tab="eval"):
        super().__init__()
        self.log_tab = log_tab
        self.n_steps = n_steps
        self.eval_n_episodes = eval_n_episodes
        self.deterministic = deterministic
        self.last_time_trigger = 0
        self.eval_env = eval_env

    def _calc_metrics(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback function to calculate metrics after each step.

        Parameters:
        -----------
        locals_: Local variables at the current step.
        globals_: Global variables at the current step.
        """
        info = locals_["info"]
        if "psi_smoothness" not in info:
            return
        
        # Collect metrics for each episode
        self.psi_smoothness_for_single_episode.append(info["psi_smoothness"])
        self.speed_smoothness_for_single_episode.append(info["speed_smoothness"])

        # Check for specific conditions like offroad, collision, etc.
        if (info["offroad"] > 0 or info["collision"] > 0 or 
            info["traffic_light_violation"] > 0 or info["is_success"]):
            
            self.episode_num += 1

            if info["offroad"] > 0:
                self.offroad_num += 1
            if info["collision"] > 0:
                self.collision_num += 1
            if info["traffic_light_violation"] > 0:
                self.traffic_light_violation_num += 1
            if info["is_success"]:
                self.success_num += 1

            self.reached_waypoint_nums.append(info["reached_waypoint_num"])

            # Calculate episode-level smoothness
            if len(self.psi_smoothness_for_single_episode) > 0:
                self.psi_smoothness.append(
                    sum(self.psi_smoothness_for_single_episode) / len(self.psi_smoothness_for_single_episode)
                )
            if len(self.speed_smoothness_for_single_episode) > 0:
                self.speed_smoothness.append(
                    sum(self.speed_smoothness_for_single_episode) / len(self.speed_smoothness_for_single_episode)
                )

    def _evaluate(self) -> bool:
        """
        Perform evaluation on the environment and record the metrics.
        """
        # Reset all metrics
        self.episode_num = 0
        self.offroad_num = 0
        self.collision_num = 0
        self.traffic_light_violation_num = 0
        self.success_num = 0
        self.reached_waypoint_nums = []
        self.psi_smoothness = []
        self.speed_smoothness = []

        mean_episode_reward = 0
        mean_episode_length = 0

        # Evaluate for the specified number of episodes
        for i in range(self.eval_n_episodes):
            self.psi_smoothness_for_single_episode = []
            self.speed_smoothness_for_single_episode = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=1,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                callback=self._calc_metrics,
            )

            # Accumulate rewards and lengths
            mean_episode_reward += sum(episode_rewards) / len(episode_rewards)
            mean_episode_length += sum(episode_lengths) / len(episode_lengths)

        mean_episode_reward /= self.eval_n_episodes
        mean_episode_length /= self.eval_n_episodes

        # Log evaluation results
        self.logger.record(f"{self.log_tab}/mean_episode_reward", mean_episode_reward)
        self.logger.record(f"{self.log_tab}/mean_episode_length", mean_episode_length)
        self.logger.record(f"{self.log_tab}/offroad_rate", self.offroad_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/collision_rate", self.collision_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/traffic_light_violation_rate", self.traffic_light_violation_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/success_percentage", self.success_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/reached_waypoint_num", sum(self.reached_waypoint_nums) / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/psi_smoothness", sum(self.psi_smoothness) / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/speed_smoothness", sum(self.speed_smoothness) / self.eval_n_episodes)

    def _on_training_start(self) -> None:
        """
        Perform evaluation at the start of training.
        """
        self._evaluate()

    def _on_step(self) -> bool:
        """
        Called at each step during training to check if evaluation should be triggered.
        """
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            self._evaluate()
        return True

def make_env_(env_config):
    """
    Create and configure a training environment.

    Parameters:
    -----------
    env_config: Configuration for the environment.

    Returns:
    --------
    env: The created gym environment.
    """
    env = gym.make('torchdriveenv-v0', args={'cfg': env_config, 'data': training_data})
    return env

def make_val_env_(env_config):
    """
    Create and configure a validation environment.

    Parameters:
    -----------
    env_config: Configuration for the environment.

    Returns:
    --------
    env: The created gym environment.
    """
    env = gym.make('torchdriveenv-v0', args={'cfg': env_config, 'data': validation_data})
    return env

if __name__ == '__main__':
    # Command-line argument parser
    parser = argparse.ArgumentParser(prog='tde_examples', description='Execute benchmarks for TDE')
    parser.add_argument("--config_file", type=str, default="env_configs/multi_agent/sac_training.yml")  
    args = parser.parse_args()

    # Load RL training configuration
    rl_training_config = load_rl_training_config(args.config_file)

    # Merge configuration parameters for the environment and simulator
    config = {k: v for (k, v) in vars(rl_training_config).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))}
    config.update({'env-' + k: v for (k, v) in vars(rl_training_config.env).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))})
    config.update({'tds-' + k: v for (k, v) in vars(rl_training_config.env.simulator).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))})

    experiment_name = f"{rl_training_config.algorithm}_{int(time.time())}"

    # Create training environment
    env = make_env_(rl_training_config.env)
    
    # Initialize model (A2C in this case)
    model = A2C("TransformerPolicy", env, verbose=1, n_steps=int(256 / rl_training_config.parallel_env_num), gae_lambda=0.95, ent_coef=0.01)

    # Initialize evaluation callbacks for validation and training environments
    eval_val_env = make_env_(rl_training_config.env)
    eval_val_callback = EvalNTimestepsCallback(eval_val_env, n_steps=rl_training_config.eval_val_callback['n_steps'], 
                                               eval_n_episodes=rl_training_config.eval_val_callback['eval_n_episodes'], 
                                               deterministic=rl_training_config.eval_val_callback['deterministic'], log_tab="eval_val")
    
    eval_train_env = make_env_(rl_training_config.env)
    eval_train_callback = EvalNTimestepsCallback(eval_train_env, n_steps=rl_training_config.eval_train_callback['n_steps'], 
                                                 eval_n_episodes=rl_training_config.eval_train_callback['eval_n_episodes'], 
                                                 deterministic=rl_training_config.eval_train_callback['deterministic'], log_tab="eval_train")

    # Run training loop
    vec_env = model.get_env()
    obs = vec_env.reset()

    # Example loop for obtaining relative positions and actions during training
    for i in range(1000):
        relative_positions = vec_env.envs[0].simulator.get_all_agents_relative(exclude_self=True)
        if len(relative_positions.shape) == 2:
            relative_positions = relative_positions.unsqueeze(0)
        if len(relative_positions.shape) == 4:
            relative_positions = relative_positions.squeeze(0)

        # Predict the action and take a step in the environment
        action, _state = model.predict(relative_positions.cuda(), deterministic=True)
        action = action.squeeze(0)
        obs, reward, done, info = vec_env.step(action)
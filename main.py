from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from cleaning_room import CleaningRoomEnv
import numpy as np
import torch
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Calculate the flat observation space size
        dirt_map_size = np.prod(env.observation_space['dirt_map'].shape)
        agent_state_size = np.prod(env.observation_space['agent_state'].shape)
        total_size = dirt_map_size + agent_state_size
        
        # Define the new flattened observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_size,), 
            dtype=np.float32
        )

    def observation(self, obs):
        # Flatten the observation dictionary into a single array
        dirt_map_flat = obs['dirt_map'].flatten()
        agent_state = obs['agent_state']
        return np.concatenate([dirt_map_flat, agent_state])

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=50000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            # Render the agent in the environment at the specified interval
            self.training_env.envs[0].render()
        return True

def make_env(grid_size=10, rank=0):
    def _init():
        env = CleaningRoomEnv(grid_size=grid_size)
        env = FlattenObservationWrapper(env)
        env = Monitor(env)
        return env
    return _init

def train_agent():
    # Environment parameters
    grid_size = 30
    num_envs = 4  # Number of parallel environments
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(grid_size=grid_size) for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Define model parameters
    model_params = {
        "policy": "MlpPolicy",
        "learning_rate": 5e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.1,
        "policy_kwargs": dict(
            net_arch=dict(
                pi=[512, 512],  # Actor network
                vf=[512, 512]   # Critic network
            ),
            activation_fn=torch.nn.LeakyReLU
        ),
        "device": "cpu"
    }
    
    # Setup TensorBoard logging
    log_dir = "./logs/tensorboard"

    # Create the model
    # model = PPO(env=env, tensorboard_log=log_dir, verbose=1, **model_params)
    model = PPO.load("logs/best_model/best_model.zip", env=env, tensorboard_log=log_dir, verbose=1, **model_params)
    
    # Set up callbacks
    eval_env = DummyVecEnv([make_env(grid_size=grid_size)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/checkpoints/",
        name_prefix="cleaning_model"
    )
    
    # render_callback = RenderCallback(render_freq=50000)


    # Train the agent
    total_timesteps = 3_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save("cleaning_agent_final")
    env.save("vec_normalize.pkl")

def evaluate_agent(model_path="cleaning_agent_final", vec_normalize_path="vec_normalize.pkl"):
    # Create environment
    env = DummyVecEnv([make_env(grid_size=30)])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False  # Don't update normalization statistics during evaluation
    env.norm_reward = False
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Run evaluation episodes
    n_eval_episodes = 10
    episode_rewards = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # Optional: render every few steps
            if episode == 0:  # Render first episode
                env.envs[0].render()
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1} reward: {episode_reward}")
    
    print(f"\nAverage reward over {n_eval_episodes} episodes: {np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    # Train the agent
    train_agent()
    
    # Evaluate the trained agent
    evaluate_agent()
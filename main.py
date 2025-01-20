from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from cleaning_room import CleaningRoomEnv
import numpy as np
import torch
from gymnasium import spaces
import gymnasium as gym

class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        dirt_map_size = np.prod(env.observation_space['dirt_map'].shape)
        agent_state_size = np.prod(env.observation_space['agent_state'].shape)
        total_size = dirt_map_size + agent_state_size
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_size,), 
            dtype=np.float32
        )

    def observation(self, obs):
        dirt_map_flat = obs['dirt_map'].flatten()
        agent_state = obs['agent_state']
        return np.concatenate([dirt_map_flat, agent_state]).astype(np.float32)

class MonitorStdCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.std_values = []

    def _on_step(self) -> bool:
        if hasattr(self.model.policy, 'action_dist'):
            std = self.model.policy.action_dist.distribution.stddev.mean().item()
            self.std_values.append(std)
            if self.n_calls % 10000 == 0:
                self.logger.record('train/action_std', std)
        return True

class AdaptiveLearningRateCallback(BaseCallback):
    def __init__(self, learning_rate_schedule, verbose=0):
        super().__init__(verbose)
        self.learning_rate_schedule = learning_rate_schedule

    def _init_callback(self) -> None:
        # No need to store any additional attributes
        pass
        
    def _on_step(self) -> bool:
        try:
            # Get progress from the local training progress
            if not hasattr(self, "training_env") or not hasattr(self, "n_calls") or not hasattr(self, "num_timesteps"):
                self.logger.warn("Required attributes not found, skipping learning rate update")
                return True
            
            # Calculate progress remaining (1.0 -> 0.0)
            if hasattr(self.model, "num_timesteps") and hasattr(self.model, "_total_timesteps"):
                progress_remaining = 1.0 - float(self.model.num_timesteps) / float(self.model._total_timesteps)
                new_lr = self.learning_rate_schedule(progress_remaining)
                
                # Update the learning rate
                if hasattr(self.model, "learning_rate"):
                    self.model.learning_rate = new_lr
                    if self.verbose >= 2:
                        self.logger.record("train/learning_rate", new_lr)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in learning rate callback: {str(e)}")
            return True

def make_env(grid_size=10, rank=0):
    def _init():
        env = CleaningRoomEnv(grid_size=grid_size)
        env = FlattenObservationWrapper(env)
        env = Monitor(env)
        return env
    return _init

def linear_schedule(initial_value: float, final_value: float) -> callable:
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func

def train_agent(continue_training: bool = False):
    # Environment setup
    grid_size = 10
    num_envs = 8  # Increased number of parallel environments
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(grid_size=grid_size) for _ in range(num_envs)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    # Learning rate schedule
    lr_schedule = linear_schedule(3e-4, 1e-4)
    
    # Define model parameters
    model_params = {
        "policy": "MlpPolicy",
        "learning_rate": lr_schedule,  # Dynamic learning rate
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": 0.2,
        "ent_coef": 0.005,
        "vf_coef": 0.75,
        "max_grad_norm": 0.5,
        "policy_kwargs": dict(
            net_arch=dict(
                pi=[256, 256, 128],
                vf=[256, 256, 128]
            ),
            activation_fn=torch.nn.ReLU,
            log_std_init=-2.0,
            ortho_init=True
        ),
        "device": "cpu" if not torch.cuda.is_available() else "cuda"
    }
    
    # Setup logging
    log_dir = "./logs/tensorboard"
    
    # Create or load model
    if continue_training:
        model = PPO.load(
            "logs/best_model/best_model.zip",
            env=env,
            tensorboard_log=log_dir,
            verbose=1,
            **model_params
        )
    else:
        model = PPO(
            env=env,
            tensorboard_log=log_dir,
            verbose=1,
            **model_params
        )
    
    # Setup evaluation environment
    eval_env = DummyVecEnv([make_env(grid_size=grid_size)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/checkpoints/",
        name_prefix="cleaning_model"
    )
    
    std_callback = MonitorStdCallback()
    lr_callback = AdaptiveLearningRateCallback(lr_schedule)
    
    # Train the agent
    total_timesteps = 3_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, std_callback, lr_callback],
        progress_bar=True,
        log_interval=1000
    )
    
    # Save final model and environment
    model.save("cleaning_agent_final")
    env.save("vec_normalize.pkl")
    
    return model, env

def evaluate_agent(
    model_path: str = "cleaning_agent_final",
    vec_normalize_path: str = "vec_normalize.pkl",
    n_eval_episodes: int = 10,
    render: bool = True
):
    env = DummyVecEnv([make_env(grid_size=10)])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(model_path)
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if render and episode == 0:
                env.envs[0].render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1} reward: {episode_reward:.2f}, length: {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation over {n_eval_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.2f}")
    
    return mean_reward, std_reward, mean_length

if __name__ == "__main__":
    # Train the agent
    model, env = train_agent(continue_training=False)
    
    # Evaluate the trained agent
    evaluate_agent(render=True)
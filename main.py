from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback

# Import your custom environment (assuming it's now Gymnasium-compatible)
from cleaning_room import EnhancedCleaningRoomEnv

# --- 1. Create Environments ---
# Instantiate the training environment
env = EnhancedCleaningRoomEnv(grid_size=64, max_steps=1000, initial_dirt_pattern='rug_cleaning')
# Wrap the training environment with Monitor
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, filename=os.path.join(log_dir, "training_monitor.csv"), info_keywords=("dirt_left",))
env = DummyVecEnv([lambda: env]) # Vectorize for training

# Create a separate evaluation environment
eval_env = EnhancedCleaningRoomEnv(grid_size=64, max_steps=1000, initial_dirt_pattern='rug_cleaning')
# Wrap the evaluation environment with Monitor
eval_log_dir = "eval_logs"
os.makedirs(eval_log_dir, exist_ok=True)
eval_env = Monitor(eval_env, filename=os.path.join(eval_log_dir, "eval_monitor.csv"), info_keywords=("dirt_left",))
eval_env = DummyVecEnv([lambda: eval_env]) # Vectorize evaluation env


# --- 2. Instantiate the Algorithm with TensorBoard Logging ---
# Define TensorBoard log directory
tensorboard_log_dir = os.path.join('tensorboard_logs', 'ppo_cleaning_room_gymnasium') # Updated TensorBoard log dir name
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Use PPO with MultiInputPolicy, enable TensorBoard logging
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)


# --- 3. Set up Callbacks ---
# --- Checkpoint Callback ---
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=os.path.join('training', 'checkpoints'),
    name_prefix='ppo_cleaning_agent_checkpoint'
)

# --- Evaluation Callback ---
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join('training', 'best_model'),
    log_path=os.path.join('training', 'eval_logs'),
    eval_freq=1000,
    deterministic=True,
    render=False
)

# --- Progress Bar Callback ---
progress_bar_callback = ProgressBarCallback()

# Combine callbacks into a list
callbacks = [checkpoint_callback, eval_callback, progress_bar_callback]


# --- 4. Train the Agent with Callbacks ---
total_timesteps = 2000
model.learn(total_timesteps=total_timesteps, callback=callbacks)


# --- 5. Save the Final Trained Model ---
final_model_save_path = os.path.join('training', 'final_model', 'ppo_cleaning_agent_final')
os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True)
model.save(final_model_save_path)
print(f"Final trained model saved to: {final_model_save_path}")

# --- 6. Evaluation Episodes (using the wrapped evaluation environment) ---
loaded_model = PPO.load(final_model_save_path, env=eval_env)

num_episodes = 3
for episode in range(num_episodes):
    # Gymnasium reset returns observation and info dictionary
    obs, info = eval_env.reset() # **[Gymnasium Update: reset() returns tuple]**
    done = False # Initialize done for each episode (not strictly needed in this loop structure but good practice)
    episode_reward = 0
    step_count = 0
    print(f"--- Episode {episode+1} ---")
    while not done: # 'done' is used for loop control, but 'terminated' is the Gymnasium flag
        action, _ = loaded_model.predict(obs, deterministic=True)
        # Gymnasium step returns observation, reward, terminated, truncated, info
        obs, rewards, terminated, truncated, infos = eval_env.step(action) # **[Gymnasium Update: step() returns terminated, truncated]**
        reward = rewards[0]
        terminated_env = terminated[0] # Get terminated flag for the first (and only) env in VecEnv
        truncated_env = truncated[0] # Get truncated flag
        info = infos[0]

        episode_reward += reward
        step_count += 1

        eval_env.envs[0].render() # Optional rendering

        # Gymnasium uses 'terminated' and 'truncated' instead of 'done'
        if terminated_env or truncated_env: # **[Gymnasium Update: Check terminated OR truncated]**
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step_count}, Terminated = {terminated_env}, Truncated = {truncated_env}") # **[Gymnasium Update: Print terminated and truncated]**
            done = True # Set 'done' to exit the episode loop
            break
eval_env.close()

print("Evaluation finished.")
print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
print(f"Training Monitor logs saved to: {os.path.join(log_dir, 'training_monitor.csv')}")
print(f"Evaluation Monitor logs saved to: {os.path.join(eval_log_dir, 'eval_monitor.csv')}")
print("Checkpoints and best model are saved in the 'training' directory.")
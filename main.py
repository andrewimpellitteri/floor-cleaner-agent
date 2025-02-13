from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    ProgressBarCallback,
)

# from gymnasium.wrappers import FlattenObservation
import numpy as np

# Import your custom environment
from cleaning_room import EnhancedCleaningRoomEnv
from feature_extraction import CustomCleaningFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize

# --- Configuration ---
TRAINING_LOG_DIR = "training_logs"
EVAL_LOG_DIR = "eval_logs"
TENSORBOARD_LOG_DIR = os.path.join("tensorboard_logs", "ppo_cleaning_room_gymnasium")
CHECKPOINT_SAVE_PATH = os.path.join("training", "checkpoints")
BEST_MODEL_SAVE_PATH = os.path.join("training", "best_model")
FINAL_MODEL_SAVE_PATH = os.path.join(
    "training", "final_model", "ppo_cleaning_agent_final"
)
EVAL_FREQ = 1000
CHECKPOINT_FREQ = 5000
TOTAL_TIMESTEPS = 20_000  # Example value, adjust as needed
GRID_SIZE = 64
MAX_STEPS = 1000
INITIAL_DIRT_PATTERN = "rug_cleaning"
MODEL_POLICY = "MultiInputPolicy"
EVAL_EPISODES = 3
CONTINUE_TRAINING = (
    False  # New configuration option to enable/disable continue training
)

def create_eval_env():
    """
    Creates and returns the evaluation environment.
    """
    os.makedirs(EVAL_LOG_DIR, exist_ok=True)

    def make_eval_env():
        base_eval_env = EnhancedCleaningRoomEnv(
            grid_size=GRID_SIZE,
            max_steps=MAX_STEPS,
            initial_dirt_pattern=INITIAL_DIRT_PATTERN,
        )
        monitor_eval_env = Monitor(
            base_eval_env,
            filename=os.path.join(EVAL_LOG_DIR, "eval_monitor.csv"),
            info_keywords=("dirt_left",),
        )
        return monitor_eval_env

    eval_env_vectorized = DummyVecEnv([make_eval_env])
    # Add VecNormalize wrapper with same parameters as training env
    eval_env_vectorized = VecNormalize(
        eval_env_vectorized,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        training=False  # Important: set training=False for evaluation
    )
    eval_env_render = eval_env_vectorized.envs[0].env

    print("Evaluation environment created.")
    return eval_env_vectorized, eval_env_render, EVAL_LOG_DIR




def train_model(
    eval_env_vectorized, continue_training=CONTINUE_TRAINING
):  # Added continue_training parameter with default value from config
    """
    Trains the PPO model and sets up evaluation and checkpoint callbacks,
    with the option to continue training from a checkpoint.

    Args:
        eval_env_vectorized: Vectorized evaluation environment for EvalCallback.
        continue_training (bool): If True, tries to load the latest checkpoint to continue training.

    Returns:
        str: Path to the saved final trained model.
    """
    # 1. Create Training Environment
    os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
    train_env = EnhancedCleaningRoomEnv(
        grid_size=GRID_SIZE,
        max_steps=MAX_STEPS,
        initial_dirt_pattern=INITIAL_DIRT_PATTERN,
    )
    train_env = train_env
    train_env = Monitor(
        train_env,
        filename=os.path.join(TRAINING_LOG_DIR, "training_monitor.csv"),
        info_keywords=("dirt_left",),
    )
    train_env_vectorized = DummyVecEnv([lambda: train_env])

    train_env_vectorized = VecNormalize(
        train_env_vectorized,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # 2. Instantiate or Load Algorithm & TensorBoard Logging
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    latest_checkpoint_path = None  # Initialize to None

    policy_kwargs = dict(
        features_extractor_class=CustomCleaningFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    if continue_training:
        import glob

        checkpoint_files = glob.glob(
            os.path.join(CHECKPOINT_SAVE_PATH, "ppo_cleaning_agent_checkpoint_*.zip")
        )
        if checkpoint_files:  # Check if any checkpoint files exist
            latest_checkpoint_path = max(
                checkpoint_files, key=os.path.getctime
            )  # Find the latest checkpoint file based on creation time
            print(
                f"Checkpoint files found. Loading model from: {latest_checkpoint_path}"
            )
            model = PPO.load(
                latest_checkpoint_path,
                env=train_env_vectorized,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                gae_lambda=0.95,
                clip_range=0.2,
                policy_kwargs=policy_kwargs,
                tensorboard_log=TENSORBOARD_LOG_DIR,
                verbose=1,
            )  # Load with tensorboard log dir
            print("Model loaded from checkpoint, continuing training.")
        else:
            print("No checkpoint found. Starting training from scratch.")
            model = PPO(
                MODEL_POLICY,
                train_env_vectorized,
                learning_rate=3e-4,
                n_steps=2048,  # Increase batch size
                batch_size=64,  # Add mini-batch training
                n_epochs=10,  # Multiple PPO optimization epochs
                gamma=0.99,  # Adjust if needed based on your reward structure
                ent_coef=0.01,
                gae_lambda=0.95,
                clip_range=0.2,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log=TENSORBOARD_LOG_DIR,
                verbose=1,
            )  # Create new model
    else:  # Start from scratch if continue_training is False
        print("Starting training from scratch (continue_training=False).")
        model = PPO(
            MODEL_POLICY,
            train_env_vectorized,
            learning_rate=3e-4,
            n_steps=2048,  # Increase batch size
            batch_size=64,  # Add mini-batch training
            n_epochs=10,  # Multiple PPO optimization epochs
            gamma=0.99,  # Adjust if needed based on your reward structure
            ent_coef=0.01,
            gae_lambda=0.95,
            clip_range=0.2,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            verbose=1,
        )  # Create new model

    print(f"PPO model initialized with policy: {MODEL_POLICY}")

    # 3. Set up Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_SAVE_PATH,
        name_prefix="ppo_cleaning_agent_checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env_vectorized,
        best_model_save_path=BEST_MODEL_SAVE_PATH,
        log_path=os.path.join(
            TRAINING_LOG_DIR, EVAL_LOG_DIR
        ),  # Corrected log_path to be within training logs
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )
    progress_bar_callback = ProgressBarCallback()
    callbacks = [checkpoint_callback, eval_callback, progress_bar_callback]
    print("Callbacks set up for checkpointing, evaluation, and progress tracking.")

    # 4. Train Agent
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=not continue_training,
    )  # Important: reset_num_timesteps
    print("Training finished.")

    # 5. Save Final Model
    os.makedirs(os.path.dirname(FINAL_MODEL_SAVE_PATH), exist_ok=True)
    model.save(FINAL_MODEL_SAVE_PATH)
    print(f"Final trained model saved to: {FINAL_MODEL_SAVE_PATH}")

    print(f"TensorBoard logs saved to: {TENSORBOARD_LOG_DIR}")
    print(
        f"Training Monitor logs saved to: {os.path.join(TRAINING_LOG_DIR, 'training_monitor.csv')}"
    )
    print(
        f"Checkpoints and best model are saved in: {os.path.dirname(CHECKPOINT_SAVE_PATH)}"
    )

    return FINAL_MODEL_SAVE_PATH


def evaluate_model(
    final_model_save_path, eval_env_vectorized, eval_env_render, eval_log_dir
):
    """
    Evaluates the trained PPO model over a number of episodes and renders one episode.

    Args:
        final_model_save_path: Path to the saved final model.
        eval_env_vectorized: Vectorized evaluation environment.
        eval_env_render: Base evaluation environment for rendering.
        eval_log_dir: Directory for evaluation logs.
    """
    print(f"Loading model from: {final_model_save_path}")
    loaded_model = PPO.load(final_model_save_path, env=eval_env_vectorized)
    print("Model loaded successfully.")

    for episode in range(EVAL_EPISODES):
        obs = eval_env_vectorized.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle vectorized reset tuple output if needed

        terminated = False
        truncated = False
        episode_reward = 0
        step_count = 0

        print(f"--- Episode {episode + 1} Evaluation ---")

        while not terminated and not truncated:
            action, _ = loaded_model.predict(obs, deterministic=True)
            action = np.nan_to_num(
                action, nan=0.0, posinf=1.0, neginf=-1.0
            )  # Sanitize action
            if len(action.shape) == 1:
                action = action.reshape(
                    1, -1
                )  # Ensure correct shape for vectorized env

            step_result = eval_env_vectorized.step(action)
            if len(step_result) == 5:
                obs_step, rewards, terminated_vec, truncated_vec, infos = step_result
            else:
                obs_step, rewards, terminated_vec, infos = step_result
                truncated_vec = terminated_vec  # Assume truncated is same as terminated if not returned

            # Unpack vectorized results
            obs = obs_step[0] if isinstance(obs_step, (list, np.ndarray)) else obs_step
            reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
            terminated = (
                terminated_vec[0]
                if isinstance(terminated_vec, (list, np.ndarray))
                else terminated_vec
            )
            truncated = (
                truncated_vec[0]
                if isinstance(truncated_vec, (list, np.ndarray))
                else truncated_vec
            )

            episode_reward += reward
            step_count += 1

            eval_env_render.render()  # Render base environment

            if terminated or truncated:
                info = (
                    infos[0]
                    if isinstance(infos, (list, np.ndarray)) and len(infos) > 0
                    else {}
                )
                print(
                    f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, Steps = {step_count}, Terminated = {terminated}, Truncated = {truncated}, Info: {info}"
                )
                break

    # Close the environments after all episodes are done.
    eval_env_vectorized.close()
    eval_env_render.close()

    print(
        f"Evaluation Monitor logs saved to: {os.path.join(eval_log_dir, 'eval_monitor.csv')}"
    )
    print("Evaluation finished.")


if __name__ == "__main__":
    RUN_TRAINING = True  # Control flag to enable/disable training
    RUN_EVALUATION = True  # Control flag to enable/disable evaluation

    RUN_BEST_CHECKPOINT = False

    # Create evaluation environment (needed for EvalCallback during training and for evaluation runs)
    eval_env_vectorized, eval_env_render, eval_log_dir = create_eval_env()

    if RUN_TRAINING:
        final_model_save_path = train_model(
            eval_env_vectorized, continue_training=CONTINUE_TRAINING
        )  # Pass continue_training flag

    if RUN_EVALUATION:
        if RUN_BEST_CHECKPOINT:
            BEST_MODEL_SAVE_PATH = os.path.join(BEST_MODEL_SAVE_PATH, "best_model.zip")
            evaluate_model(
                BEST_MODEL_SAVE_PATH, eval_env_vectorized, eval_env_render, eval_log_dir
            )

        else:
            evaluate_model(
                FINAL_MODEL_SAVE_PATH,
                eval_env_vectorized,
                eval_env_render,
                eval_log_dir,
            )

    print(
        f"Evaluation Monitor logs saved to: {os.path.join(eval_log_dir, 'eval_monitor.csv')}"
    )
    print("Script execution finished.")

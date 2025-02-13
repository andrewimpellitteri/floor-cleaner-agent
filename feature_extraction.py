import gymnasium
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCleaningFeaturesExtractor(BaseFeaturesExtractor):
    """
    Improved custom feature extractor for EnhancedCleaningRoomEnv.
    Deeper and wider CNN for dirt_map, more complex MLP, Batch Normalization.
    """
    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # --- Deeper and Wider CNN for Dirt Map Processing with Batch Normalization ---
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # Compute CNN output dimension
        with th.no_grad():
            sample_dirt = th.as_tensor(observation_space.spaces['dirt_map'].sample()).unsqueeze(0).unsqueeze(0).float()
            n_flatten = self.cnn(sample_dirt).shape[1]

        # --- More Complex MLP for Agent State and Relative Drain Position with Batch Normalization ---
        mlp_input_dim = (np.prod(observation_space.spaces['agent_state'].shape) +
                         np.prod(observation_space.spaces['relative_drain_position'].shape))
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # --- Combined Final Layer ---
        self.final_layer = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        dirt_map = observations['dirt_map']
        agent_state = observations['agent_state']
        rel_drain_pos = observations['relative_drain_position']

        dirt_map = dirt_map.unsqueeze(1) # Add channel dimension
        dirt_features = self.cnn(dirt_map)

        state_features = self.mlp(th.cat([agent_state, rel_drain_pos], dim=1))

        combined_features = th.cat([dirt_features, state_features], dim=1)
        return self.final_layer(combined_features)

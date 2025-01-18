import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Output different ranges for movement and angle
        out = self.fc3(x)
        # Constrain movement actions to [-1, 1] and angle to [-π, π]
        out[:, :2] = torch.tanh(out[:, :2])  # movement
        out[:, 2:] = torch.tanh(out[:, 2:]) * np.pi  # angle
        return out

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        
        # Initialize actor network
        self.actor = ActorNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=0.002)
        
        # Exploration parameters
        self.noise_std = 1.5
        self.noise_decay = 0.997
        self.noise_min = 0.1
        
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.actor(state.unsqueeze(0)).squeeze(0)
            
            # Add exploration noise
            noise = torch.randn_like(action) * self.noise_std
            action = action + noise
            
            # Clip actions to valid ranges
            action[:2] = torch.clamp(action[:2], -1, 1)  # movement
            action[2] = torch.clamp(action[2], -np.pi, np.pi)  # angle
            
            return action.cpu().numpy()
    
    def train(self, state, action, reward, next_state, done):
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor([float(done)]).to(self.device)
        
        # Simple policy gradient update
        predicted_action = self.actor(state.unsqueeze(0)).squeeze(0)
        loss = -torch.mean(reward * torch.sum((predicted_action - action) ** 2))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration noise
        self.noise_std = max(self.noise_std * self.noise_decay, self.noise_min)
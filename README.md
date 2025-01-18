# Cleaning Room RL Agent

This project builds a Reinforcement Learning (RL) agent to find an optimal floor cleaning method using a power washer in a simulated environment. The agent can move, rotate, and spray the washer to clean dirt, with the reward being the inverse of dirt concentration on the floor.

## Features

- **Power Washer Simulation**: Agent can move and rotate to spray dirt.
- **RL Training**: Trains an agent using Proximal Policy Optimization (PPO).
- **Custom Environment**: Built with `gymnasium` for easy integration with RL algorithms.

## Dependencies

Install required packages:

```bash
pip install stable-baselines3 torch numpy gymnasium matplotlib
```
## Setup and Usage
1. **Training the Agent**
Run the train_agent() function to train the PPO agent:

python train_agent.py
This trains the agent with 4 parallel environments for 1,000,000 timesteps, saving the model as cleaning_agent_final.

2. **Evaluating the Agent**
After training, evaluate the agent's performance:

```bash
python evaluate_agent.py
```

This runs 10 evaluation episodes and prints the average reward.

Feel free to fork and submit pull requests for improvements.

## License
This project is licensed under the MIT License.


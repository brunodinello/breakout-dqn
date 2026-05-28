import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from abstract_agent import Agent
from replay_memory import Transition


class DQNAgent(Agent):
    """Vanilla DQN (Mnih et al., 2015) with a single Q-network."""

    def __init__(self, env, model, obs_processing_func, memory_buffer_size, batch_size,
                 learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps,
                 episode_block, device):
        super().__init__(env, obs_processing_func, memory_buffer_size, batch_size,
                         learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps,
                         episode_block, device)
        self.policy_net = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def select_action(self, state, current_steps, train=True):
        epsilon = self.compute_epsilon(current_steps)

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state)
            q_max = torch.max(q_values).item()

        if train and np.random.rand() <= epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.greedy_action(q_values)
        return action, q_max

    def greedy_action(self, q_values):
        return torch.argmax(q_values, dim=1).item()

    def update_weights(self):
        if len(self.memory) < self.batch_size:
            return

        self.optimizer.zero_grad()
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states_tensor = torch.stack(batch.state).to(self.device)
        actions_tensor = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_tensor = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones_tensor = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_tensor = torch.stack(batch.next_state).to(self.device)

        q_s_a = self.policy_net(states_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            max_q_next = self.policy_net(next_states_tensor).max(dim=1, keepdim=True)[0] * (1 - dones_tensor)
            target = rewards_tensor + self.gamma * max_q_next

        loss = self.criterion(q_s_a, target)
        loss.backward()
        self.optimizer.step()

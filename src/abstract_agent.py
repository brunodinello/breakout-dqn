import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import wandb
from tqdm import tqdm

from replay_memory import ReplayMemory


class Agent(ABC):
    """Base class for DQN-family agents.

    Holds the replay buffer, ε-greedy schedule, and the training/evaluation loops.
    Subclasses implement select_action() and update_weights().
    """

    def __init__(self, gym_env, obs_processing_func, memory_buffer_size, batch_size,
                 learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps,
                 episode_block, device, save_freq=1000, checkpoint_dir='checkpoints'):
        self.device = device
        self.env = gym_env
        self.state_processing_function = obs_processing_func
        self.memory = ReplayMemory(memory_buffer_size)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal_steps = epsilon_anneal_steps

        self.episode_block = episode_block
        self.total_steps = 0

        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, number_episodes=50_000, max_steps_episode=10_000, max_steps=1_000_000):
        rewards = []
        average_q_per_episode = []
        metrics = {"reward": 0.0, "epsilon": self.epsilon_i, "steps": 0}
        pbar = tqdm(range(number_episodes), desc="Training", unit="episode")

        for ep in pbar:
            if self.total_steps > max_steps:
                break

            state, _ = self.env.reset()
            state_phi = self.state_processing_function(state)
            current_episode_reward = 0.0
            current_episode_q_values = 0.0
            current_episode_steps = 0
            done = False

            for _ in range(max_steps_episode):
                action, q_max = self.select_action(state_phi, self.total_steps, train=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state_phi = self.state_processing_function(next_state)

                current_episode_reward += reward
                current_episode_q_values += q_max
                self.total_steps += 1
                current_episode_steps += 1

                self.memory.add(state_phi, action, reward, done, next_state_phi)
                self.update_weights()

                state = next_state
                state_phi = next_state_phi

                if done:
                    break

            rewards.append(current_episode_reward)
            if current_episode_steps > 0:
                avg_q_for_episode = current_episode_q_values / current_episode_steps
                average_q_per_episode.append(avg_q_for_episode)
            else:
                avg_q_for_episode = 0.0
                average_q_per_episode.append(0.0)

            metrics["reward"] = np.mean(rewards[-self.episode_block:])
            metrics["epsilon"] = self.compute_epsilon(self.total_steps)
            metrics["steps"] = self.total_steps
            pbar.set_postfix(metrics)

            if wandb.run is not None:
                wandb.log({
                    "reward": current_episode_reward,
                    "average_q": avg_q_for_episode,
                    "epsilon": self.compute_epsilon(self.total_steps),
                    "steps_in_episode": current_episode_steps,
                }, step=ep)

            if (ep + 1) % self.save_freq == 0:
                self.save_parameters(ep + 1, backup=True)

        self.save_parameters(ep + 1, backup=False)
        return rewards, average_q_per_episode

    def compute_epsilon(self, steps_so_far):
        """Linear ε-decay from epsilon_i to epsilon_f over epsilon_anneal_steps."""
        if steps_so_far < self.epsilon_anneal_steps:
            return self.epsilon_i - (self.epsilon_i - self.epsilon_f) * (steps_so_far / self.epsilon_anneal_steps)
        return self.epsilon_f

    def save_parameters(self, episode, backup=True):
        model_name = self.__class__.__name__.lower().replace('agent', '')
        fname = f"{model_name}_ep{episode}.dat" if backup else f"{model_name}_final.dat"
        path = os.path.join(self.checkpoint_dir, fname)
        torch.save(self.policy_net.state_dict(), path)

    def play(self, env=None, episodes=1):
        """Evaluation mode: run episodes without updating the network."""
        env = env or self.env
        rewards = []
        for ep in range(episodes):
            state, _ = env.reset()
            state_phi = self.state_processing_function(state)
            done = False
            ep_reward = 0
            while not done:
                action = self.select_action(state_phi, 0, train=False)
                if isinstance(action, tuple):
                    action = action[0]
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state_phi = self.state_processing_function(next_state)
                ep_reward += reward
            rewards.append(ep_reward)
            print(f"Episode {ep + 1} — reward: {ep_reward}")
        return rewards

    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        """Select an action. ε-greedy if train=True, greedy otherwise."""

    @abstractmethod
    def update_weights(self):
        """Sample a minibatch from replay memory and take one gradient step."""

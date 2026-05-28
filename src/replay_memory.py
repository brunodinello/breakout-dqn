import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))


class ReplayMemory:
    """Fixed-size circular buffer for storing past transitions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, done, next_state):
        transition = Transition(state, action, reward, done, next_state)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > len(self):
            raise ValueError(
                f"batch_size={batch_size} larger than stored transitions ({len(self)})"
            )
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = []
        self.position = 0

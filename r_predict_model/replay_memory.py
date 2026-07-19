"""Replay memory utilities for the generic MBPO training template."""

from operator import itemgetter
import random

import numpy as np


class ReplayMemory:
    """
    Fixed-capacity circular replay buffer.

    The buffer stores transitions as ``(state, action, reward, next_state,
    done)`` tuples.  After the buffer reaches ``capacity``, new transitions
    overwrite the oldest entries by advancing ``position`` modulo capacity.
    """

    def __init__(self, capacity):
        """Create an empty replay buffer with a maximum number of transitions."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Insert one transition and advance the circular write pointer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        """
        Insert a batch of transitions, wrapping around capacity if necessary.

        Args:
            batch: Sequence of transition tuples with the same layout accepted
                by ``push``.
        """
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            tail_len = len(self.buffer) - self.position
            self.buffer[self.position : len(self.buffer)] = batch[:tail_len]
            self.buffer[: len(batch) - tail_len] = batch[tail_len:]
            self.position = len(batch) - tail_len

    def sample(self, batch_size):
        """Randomly sample up to ``batch_size`` transitions without replacement."""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        # zip(*batch) transposes transition tuples into grouped fields; stack
        # then converts each field group into a NumPy batch array.
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size):
        """Sample ``batch_size`` transitions with replacement."""
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        """Return the raw internal transition list."""
        return self.buffer

    def __len__(self):
        """Return the number of currently stored transitions."""
        return len(self.buffer)

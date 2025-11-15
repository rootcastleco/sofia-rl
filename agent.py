import random
import logging
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("rootcastle.sofia_rl.agent")
logger.setLevel(logging.INFO)


class QNetwork(nn.Module):
    """Simple MLP Q-network. Replaceable for more complex policies."""

    def __init__(self, state_size: int, action_size: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_size = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU(inplace=True))
            in_size = h
        layers.append(nn.Linear(in_size, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Simple ring-buffer replay memory with uniform sampling."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.memory: Deque = deque(maxlen=self.capacity)
        random.seed(seed)
        np.random.seed(seed)

    def push(self, experience: Tuple):
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class SofiaRLAgent:
    """DQN agent with Double DQN target update, epsilon-greedy exploration."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr: float = 5e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = None,
        seed: int = 0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        # deterministic seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # networks
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # replay
        self.memory = ReplayBuffer(buffer_size, seed=seed)

        # exploration
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        logger.info(f"SofiaRLAgent initialized on device: {self.device}")

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            logger.debug(f"Exploring: random action {action}")
            return action

        state_tensor = (
            torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.qnetwork_local(state_tensor)
            action = int(q_values.argmax(dim=1).item())
        logger.debug(f"Exploiting: action {action}")
        return action

    def step(self, state, action, reward, next_state, done):
        """Store experience and trigger learning when enough samples exist."""
        self.memory.push((state, action, reward, next_state, done))

        # only learn when we have enough experiences
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences: List[Tuple]):
        """
        Double DQN learning:
         - local network selects next action (argmax)
         - target network evaluates Q-value of that action
        """
        states, actions, rewards, next_states, dones = zip(*experiences)

        # convert to tensors
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # current Q values
        current_q = self.qnetwork_local(states).gather(1, actions)

        # Double DQN:
        # local network chooses best action for next states
        next_actions = self.qnetwork_local(next_states).argmax(dim=1).unsqueeze(1)
        # target network evaluates the chosen actions
        q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()

        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        loss = nn.MSELoss()(current_q, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        logger.debug(f"Learned batch. Loss: {loss.item():.6f}")

    @staticmethod
    def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logger.debug(f"Epsilon decayed to {self.epsilon:.6f}")

    def save_checkpoint(self, filepath: str):
        torch.save(
            {
                "state_dict": self.qnetwork_local.state_dict(),
                "epsilon": self.epsilon,
                "memory_size": len(self.memory),
            },
            filepath,
        )
        logger.info(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint["state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        logger.info(f"Model loaded from {filepath}. Epsilon restored: {self.epsilon:.6f}")

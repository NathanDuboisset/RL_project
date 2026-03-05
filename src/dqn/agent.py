from abc import abstractmethod, ABC
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .models import BlockBlastCNNNet1P

class BaseAgent(ABC):
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @abstractmethod
    def select_action(self, state, epsilon):
        pass

    @abstractmethod
    def update_model(self):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

class DDQNAgent1P(BaseAgent):
    def __init__(self, action_size=192, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = BlockBlastCNNNet1P(output_size=action_size).to(self.device)
        self.target_net = BlockBlastCNNNet1P(output_size=action_size).to(self.device)
        self.update_target_model() # identical weights at the start
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size) # Replay Buffer
        # self.loss_fn = nn.SmoothL1Loss() # Huber Loss
        self.loss_fn = nn.MSELoss() # Mean Squared Error Loss

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, epsilon):
        """Epsilon-Greedy Action Selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            board = torch.FloatTensor(state['board']).unsqueeze(0).to(self.device)
            pieces = torch.FloatTensor(state['pieces']).unsqueeze(0).to(self.device)
            used = torch.FloatTensor(state['pieces_used']).unsqueeze(0).to(self.device)
            combo = torch.FloatTensor(state['combo']).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(board, pieces, used, combo)
            return torch.argmax(q_values).item()

    def update_model(self):
        """Batch training with Replay Buffer."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        
        # preparing batches for each component of the state
        states = {key: torch.stack([torch.FloatTensor(s[0][key]) for s in batch]).to(self.device) 
                  for key in ['board', 'pieces', 'pieces_used', 'combo']}
        actions = torch.LongTensor([s[1] for s in batch]).to(self.device).view(-1, 1)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device).view(-1, 1)
        next_states = {key: torch.stack([torch.FloatTensor(s[3][key]) for s in batch]).to(self.device) 
                       for key in ['board', 'pieces', 'pieces_used', 'combo']}
        dones = torch.FloatTensor([s[4] for s in batch]).to(self.device).view(-1, 1)

        # computing current Q values and target Q values
        current_q = self.policy_net(states['board'], states['pieces'], 
                                    states['pieces_used'], states['combo']).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_net(next_states['board'], next_states['pieces'], 
                                     next_states['pieces_used'], next_states['combo']).max(1)[0].view(-1, 1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # loss and optimization step
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_model(self, path):
        """Save architecture and weights"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Load a saved model from path"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class DuelingDDQNAgent(DDQNAgent):
    pass

class PrioritizedReplayDDQNAgent(DDQNAgent):
    pass

class NoiseNetDDQNAgent(DDQNAgent):
    pass

class CategoricalDDQNAgent(DDQNAgent):
    pass

class MultiStepDDQNAgent(DDQNAgent):
    pass

class RainbowDDQNAgent(DDQNAgent):
    pass
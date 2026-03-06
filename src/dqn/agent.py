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
    def __init__(self, action_size=64, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64, device = None):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        if device == None :
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

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
        """Epsilon-Greedy Action Selection with valid move masking"""
        valid_mask = state['valid_placements'].flatten()
        valid_actions = np.flatnonzero(valid_mask)
        
        if len(valid_actions) == 0:
            # Fallback if no valid actions exist (should be handled by done flag first generally)
            return random.randint(0, self.action_size - 1)

        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            board = torch.FloatTensor(state['board']).unsqueeze(0).to(self.device)
            piece = torch.FloatTensor(state['piece']).unsqueeze(0).to(self.device)
            valid = torch.FloatTensor(state['valid_placements']).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(board, piece, valid).squeeze(0)
            
            # Mask invalid actions
            valid_tensor_mask = torch.BoolTensor(valid_mask).to(self.device)
            q_values[~valid_tensor_mask] = -1e9
            
            return torch.argmax(q_values).item()

    def update_model(self):
        """Batch training with Replay Buffer."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        
        # preparing batches for each component of the state
        states = {key: torch.stack([torch.FloatTensor(s[0][key]) for s in batch]).to(self.device) 
                  for key in ['board', 'piece', 'valid_placements']}
        actions = torch.LongTensor([s[1] for s in batch]).to(self.device).view(-1, 1)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device).view(-1, 1)
        next_states = {key: torch.stack([torch.FloatTensor(s[3][key]) for s in batch]).to(self.device) 
                       for key in ['board', 'piece', 'valid_placements']}
        dones = torch.FloatTensor([s[4] for s in batch]).to(self.device).view(-1, 1)

        # computing current Q values and target Q values
        current_q = self.policy_net(states['board'], states['piece'], 
                                    states['valid_placements']).gather(1, actions)
        
        with torch.no_grad():
            next_q_all = self.target_net(next_states['board'], next_states['piece'], 
                                         next_states['valid_placements'])
            
            # mask invalid actions in next state
            next_valid_mask = next_states['valid_placements'].view(self.batch_size, -1).bool()
            next_q_all[~next_valid_mask] = -1e9
            
            next_q = next_q_all.max(1)[0].view(-1, 1)
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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, error, sample):
        idx = self.write + self.capacity - 1
        self.data[self.write] = sample
        self.update(idx, error)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n < self.capacity:
            self.n += 1

    def update(self, idx, error):
        p = self._get_priority(error)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def _get_priority(self, error):
        return (np.abs(error) + 1e-5) ** self.alpha

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.get(s)
            
            # handle empty cells
            if data == 0 or data is None: 
                # Retry sampling randomly
                dataIdx = random.randint(0, self.n - 1)
                idx = dataIdx + self.capacity - 1
                p = self.tree[idx]
                data = self.data[dataIdx]
                
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.total()
        is_weight = np.power(self.n * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def __len__(self):
        return self.n

class RainbowAgent1P(BaseAgent):
    def __init__(self, action_size=64, lr=1e-4, gamma=0.99, buffer_size=10000, 
                 batch_size=64, device=None, num_atoms=51, v_min=-10.0, v_max=10.0, 
                 n_step=3, alpha=0.5, beta=0.4, beta_increment_per_sampling=0.001):
        
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_step = n_step
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Distributional RL support calculation
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        from .models import RainbowCNNNet1P # Local import just to be sure
        
        self.policy_net = RainbowCNNNet1P(action_size=action_size, num_atoms=num_atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.target_net = RainbowCNNNet1P(action_size=action_size, num_atoms=num_atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # PER Buffer
        self.memory = PrioritizedReplayBuffer(capacity=buffer_size, alpha=self.alpha)
        
        # Multi-step
        self.n_step_buffer = deque(maxlen=self.n_step)

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """Modified store_transition that uses N-step returns and PER"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Calculate N-step return
        reward_n_step, next_state_n_step, done_n_step = self._get_n_step_info()
        state_0, action_0 = self.n_step_buffer[0][:2]
        
        transition = (state_0, action_0, reward_n_step, next_state_n_step, done_n_step)
        
        # Compute max priority for new transitions to ensure they are sampled at least once
        max_priority = np.max(self.memory.tree[-self.memory.capacity:]) if self.memory.n > 0 else 1.0
        self.memory.add(max_priority, transition)
        
    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def select_action(self, state, epsilon=0.0):
        # Epsilon is ignored in Rainbow because Noisy Nets handle exploration
        valid_mask = state['valid_placements'].flatten()
        valid_actions = np.flatnonzero(valid_mask)
        
        if len(valid_actions) == 0:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            board = torch.FloatTensor(state['board']).unsqueeze(0).to(self.device)
            piece = torch.FloatTensor(state['piece']).unsqueeze(0).to(self.device)
            valid = torch.FloatTensor(state['valid_placements']).unsqueeze(0).to(self.device)
            
            # Forward pass returning distribution probabilities
            probs = self.policy_net(board, piece, valid) # shape (1, num_actions, num_atoms)
            
            # Compute expected Q-values: sum(p_i * z_i)
            # shape of probs: (1, 64, 51), shape of support: (51)
            q_values = (probs * self.support).sum(dim=2).squeeze(0) # shape (64,)
            
            valid_tensor_mask = torch.BoolTensor(valid_mask).to(self.device)
            q_values[~valid_tensor_mask] = -1e9
            
            action = torch.argmax(q_values).item()
            return action

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return None

        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        # Sample from PER
        batch, idxs, is_weights = self.memory.sample(self.batch_size, beta=self.beta)
        if len(batch[0]) == 1: # if we somehow sampled a 0 initialization from numpy
            return None
            
        states = {key: torch.stack([torch.FloatTensor(s[0][key]) for s in batch]).to(self.device) 
                  for key in ['board', 'piece', 'valid_placements']}
        actions = torch.LongTensor([s[1] for s in batch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        next_states = {key: torch.stack([torch.FloatTensor(s[3][key]) for s in batch]).to(self.device) 
                       for key in ['board', 'piece', 'valid_placements']}
        dones = torch.FloatTensor([s[4] for s in batch]).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        # Resample noise for noisy layers before computing loss
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Categorical DQN Loss Calculation (Cross Entropy)
        with torch.no_grad():
            # Double DQN action selection for next state
            next_action_probs = self.policy_net(next_states['board'], next_states['piece'], next_states['valid_placements'])
            next_valid_mask = next_states['valid_placements'].view(self.batch_size, -1).bool()
            
            # Expected Q-values
            next_q_values = (next_action_probs * self.support).sum(dim=2)
            next_q_values[~next_valid_mask] = -1e9
            
            # Action selected by online network
            next_actions = next_q_values.argmax(dim=1)
            
            # Get target distribution
            next_dist = self.target_net(next_states['board'], next_states['piece'], next_states['valid_placements'])
            next_dist = next_dist[range(self.batch_size), next_actions] # shape (batch_size, num_atoms)

            # Compute projection of the target distribution
            Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)
            
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handling edge case where l == u
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            # Project probabilities
            m = torch.zeros(self.batch_size, self.num_atoms, device=self.device)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.num_atoms), self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).to(self.device)
            
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        # Current distribution
        dist = self.policy_net(states['board'], states['piece'], states['valid_placements'])
        dist = dist[range(self.batch_size), actions]

        # KL divergence loss
        epsilon = 1e-8 # Prevent nan
        loss = -(m * torch.log(dist + epsilon)).sum(dim=1)
        
        # Update PER priorities
        td_errors = loss.detach().cpu().numpy()
        for i in range(self.batch_size):
            self.memory.update(idxs[i], td_errors[i])

        # Multiply by Importance Sampling Weights
        weighted_loss = (loss * is_weights).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return weighted_loss.item()

    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
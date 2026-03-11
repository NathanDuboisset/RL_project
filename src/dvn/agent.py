import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.dqn.agent import BaseAgent
from .models import BlockBlastValueNet1Pmultikernel
from abc import abstractmethod, ABC


class DVNAgent1P(BaseAgent):
    def __init__(self, action_size=64, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64, punish_for_invalid=-500.0, device = None):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.punish_for_invalid = punish_for_invalid
        if device == None :
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.policy_net = BlockBlastValueNet1Pmultikernel().to(self.device)
        self.target_net = BlockBlastValueNet1Pmultikernel().to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)
        self.memory = deque(maxlen=buffer_size)
        self.loss_fn = nn.SmoothL1Loss()
        self.grid_size = 8
        self.base_points = 10

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _from_action_to_coordinates(self, action):
        row = action // self.grid_size
        col = action % self.grid_size
        return row, col

    def _from_coordinates_to_action(self, row, col):
        return row * self.grid_size + col


    def select_action(self, state, epsilon):
        """
        Sélectionne l'action maximisant l'équation de Bellman prospective:
        a* = argmax_a [ r(s,a) + gamma * V(S_after(s,a)) ]
        """
        valid_mask = state['valid_placements'].flatten()
        valid_actions = np.flatnonzero(valid_mask)
        
        if len(valid_actions) == 0:
            return random.randint(0, self.action_size - 1)

        if random.random() < epsilon:
            return random.choice(valid_actions)
            
        afterstates = []
        rewards = []
        
        for a in valid_actions:
            r, c = self._from_action_to_coordinates(a)
            hyp_next_board, hyp_reward = state['placements_result'][0][r, c], state['placements_result'][1][r, c]
            rewards.append(hyp_reward)
            afterstates.append(hyp_next_board)

        boards_tensor = torch.FloatTensor(np.array(afterstates)).to(self.device)
        
        with torch.no_grad():
            v_values = self.policy_net(boards_tensor).squeeze(-1).cpu().numpy()
            
        q_estimates = np.array(rewards) + self.gamma * v_values
        best_idx = np.argmax(q_estimates)
        
        return valid_actions[best_idx]

    def update_model(self):
        """
        Minimise l'erreur de différence temporelle (TD Error) par MSE.
        L(theta) = E [ (Target - V_theta(S_after_t))^2 ]
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        
        current_afterstates = []
        for s in batch:
            state, action = s[0], s[1]
            r, c = self._from_action_to_coordinates(action)
            current_afterstates.append(state['placements_result'][0][r, c])
            
        current_afterstates_t = torch.FloatTensor(np.array(current_afterstates)).to(self.device)
        current_v = self.policy_net(current_afterstates_t)
        
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device).view(-1, 1)
        dones = torch.FloatTensor([s[4] for s in batch]).to(self.device).view(-1, 1)
        
        target_v = torch.zeros((self.batch_size, 1), device=self.device)

        final_indices = torch.where(dones == 1)[0].cpu().numpy()
        if len(final_indices) > 0:
            target_v[final_indices] = self.punish_for_invalid/self.gamma 

        non_final_indices = torch.where(dones == 0)[0].cpu().numpy()
        
        if len(non_final_indices) > 0:
            for idx in non_final_indices:
                next_state = batch[idx][3]
                valid_mask = next_state['valid_placements'].flatten()
                valid_actions = np.flatnonzero(valid_mask)
                
                if len(valid_actions) == 0:
                    continue
                    
                next_afterstates = []
                next_rewards = []
                for a in valid_actions:
                    r, c = self._from_action_to_coordinates(a)
                    next_afterstate, next_reward = next_state['placements_result'][0][r, c], next_state['placements_result'][1][r, c]
                    next_afterstates.append(next_afterstate)
                    next_rewards.append(next_reward)
                    
                n_boards_t = torch.FloatTensor(np.array(next_afterstates)).to(self.device)
                
                with torch.no_grad():
                    v_vals = self.target_net(n_boards_t).squeeze(-1)
                
                next_rewards = np.array(next_rewards)
                next_reward = torch.FloatTensor(next_rewards).to(self.device)
                q_vals = next_reward + self.gamma * v_vals
                target_v[idx] = torch.max(q_vals)
                
            
        loss = self.loss_fn(current_v, target_v) 
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
        
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

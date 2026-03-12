import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.dqn.agent import BaseAgent
from abc import abstractmethod, ABC


class DVNAgent1P(BaseAgent):
    def __init__(self, policy_net: type[nn.Module], action_size=64, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64, punish_for_invalid=-500.0, device = None):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.punish_for_invalid = punish_for_invalid
        if device is None :
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.policy_net = policy_net().to(self.device)
        self.target_net = policy_net().to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)
        self.memory = deque(maxlen=buffer_size)
        self.loss_fn = nn.SmoothL1Loss()
        self.grid_size = 8
        self.base_points = 10
        self.action_rows = np.arange(self.action_size, dtype=np.int64) // self.grid_size
        self.action_cols = np.arange(self.action_size, dtype=np.int64) % self.grid_size

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

        rows = self.action_rows[valid_actions]
        cols = self.action_cols[valid_actions]
        afterstates = state['placements_result'][0][rows, cols]
        rewards = state['placements_result'][1][rows, cols]

        boards_tensor = torch.from_numpy(afterstates.astype(np.float32, copy=False)).to(self.device)

        with torch.inference_mode():
            v_values = self.policy_net(boards_tensor).squeeze(-1).cpu().numpy()

        q_estimates = rewards + self.gamma * v_values
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

        # --- Current afterstates (single batched forward pass) ---
        current_afterstates = np.empty((self.batch_size, self.grid_size, self.grid_size), dtype=np.float32)
        for i, s in enumerate(batch):
            state, action = s[0], s[1]
            r, c = self._from_action_to_coordinates(action)
            current_afterstates[i] = state['placements_result'][0][r, c]

        current_afterstates_t = torch.from_numpy(current_afterstates).to(self.device)
        current_v = self.policy_net(current_afterstates_t)

        dones_np = np.array([s[4] for s in batch], dtype=np.float32)
        target_v = torch.zeros((self.batch_size, 1), device=self.device)

        final_indices = np.where(dones_np == 1)[0]
        if len(final_indices) > 0:
            target_v[final_indices] = self.punish_for_invalid / self.gamma

        non_final_indices = np.where(dones_np == 0)[0]

        if len(non_final_indices) > 0:
            # Collect ALL next afterstates across all non-final samples,
            # then do ONE batched forward pass instead of one per sample.
            all_next_afterstates = []
            all_next_rewards = []
            sample_sizes = []

            for idx in non_final_indices:
                next_state = batch[idx][3]
                valid_mask = next_state['valid_placements'].flatten()
                valid_actions = np.flatnonzero(valid_mask)
                n = len(valid_actions)
                sample_sizes.append(n)
                if n == 0:
                    continue
                rows = self.action_rows[valid_actions]
                cols = self.action_cols[valid_actions]
                all_next_afterstates.extend(next_state['placements_result'][0][rows, cols])
                all_next_rewards.extend(next_state['placements_result'][1][rows, cols])

            if all_next_afterstates:
                n_boards_t = torch.from_numpy(
                    np.array(all_next_afterstates, dtype=np.float32)
                ).to(self.device)
                next_rewards_t = torch.from_numpy(
                    np.array(all_next_rewards, dtype=np.float32)
                ).to(self.device)

                with torch.inference_mode():
                    all_v_vals = self.target_net(n_boards_t).squeeze(-1)

                all_q_vals = next_rewards_t + self.gamma * all_v_vals

                offset = 0
                for i, idx in enumerate(non_final_indices):
                    n = sample_sizes[i]
                    if n > 0:
                        target_v[idx] = torch.max(all_q_vals[offset:offset + n])
                    offset += n

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

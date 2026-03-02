import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .model import BlockBlastCNNNet

class DQNAgent:
    def __init__(self, action_size=192, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # réseaux : Policy (celui qu'on entraîne) et Target (celui qui sert de référence)
        self.policy_net = BlockBlastCNNNet(output_size=action_size).to(self.device)
        self.target_net = BlockBlastCNNNet(output_size=action_size).to(self.device)
        self.update_target_model() # Initialisation identique
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size) # Replay Buffer
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss (plus stable que MSE)

    def update_target_model(self):
        """Copie les poids du Policy Net vers le Target Net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """Ajoute une expérience en mémoire."""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, epsilon):
        """Epsilon-Greedy avec traitement des dictionnaires."""
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            # Conversion des arrays numpy du dict en tensors
            board = torch.FloatTensor(state['board']).unsqueeze(0).to(self.device)
            pieces = torch.FloatTensor(state['pieces']).unsqueeze(0).to(self.device)
            used = torch.FloatTensor(state['pieces_used']).unsqueeze(0).to(self.device)
            combo = torch.FloatTensor(state['combo']).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(board, pieces, used, combo)
            return torch.argmax(q_values).item()

    def update_model(self):
        """Entraînement par Batch (Replay)."""
        if len(self.memory) < self.batch_size:
            return None

        # Échantillonnage aléatoire
        batch = random.sample(self.memory, self.batch_size)
        
        # On regroupe les éléments du batch (c'est la partie la plus technique avec les dict)
        states = {key: torch.stack([torch.FloatTensor(s[0][key]) for s in batch]).to(self.device) 
                  for key in ['board', 'pieces', 'pieces_used', 'combo']}
        actions = torch.LongTensor([s[1] for s in batch]).to(self.device).view(-1, 1)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device).view(-1, 1)
        next_states = {key: torch.stack([torch.FloatTensor(s[3][key]) for s in batch]).to(self.device) 
                       for key in ['board', 'pieces', 'pieces_used', 'combo']}
        dones = torch.FloatTensor([s[4] for s in batch]).to(self.device).view(-1, 1)

        # Q-valeurs actuelles prédites par le Policy Net
        current_q = self.policy_net(states['board'], states['pieces'], 
                                    states['pieces_used'], states['combo']).gather(1, actions)

        # Q-valeurs max pour l'état suivant via le Target Net
        with torch.no_grad():
            next_q = self.target_net(next_states['board'], next_states['pieces'], 
                                     next_states['pieces_used'], next_states['combo']).max(1)[0].view(-1, 1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Calcul de la perte et optimisation
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_model(self, path):
        """Sauvegarde l'architecture et les poids."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Charge le modèle sauvegardé."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
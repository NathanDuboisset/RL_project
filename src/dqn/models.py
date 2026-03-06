import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BlockBlastCNNNet(nn.Module):
    def __init__(self, output_size=192):
        super(BlockBlastCNNNet, self).__init__()
        
        # board + valid_placements = 1 + 3 channels = 4 channels total for board_conv
        self.board_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1), # Sortie: 32 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Sortie: 64 x 8 x 8
            nn.ReLU(),
            nn.Flatten() # Sortie: 64 * 8 * 8 = 4096
        )
        
        self.pieces_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Sortie: 32 x 5 x 5
            nn.ReLU(),
            nn.Flatten() # Sortie: 32 * 5 * 5 = 800
        )
        
        combined_input_dim = 4096 + 800 + 3 + 1
        
        self.fc = nn.Sequential(
            nn.Linear(combined_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size) # Les 192 Q-values
        )

    def forward(self, board, pieces, pieces_used, combo, valid_placements):
        """
        We need to give the parameters extracted for the dictionary observation space
        """
        x_board = board.unsqueeze(1).float() 
        x_valid = valid_placements.float()  # (batch_size, 3, 8, 8)
        
        # Concatenate board and valid_placements along the channel dimension
        x_board_combined = torch.cat([x_board, x_valid], dim=1) # (batch_size, 4, 8, 8)
        
        x_pieces = pieces.float()
        
        board_features = self.board_conv(x_board_combined)
        pieces_features = self.pieces_conv(x_pieces)

        x_combo = combo.float().view(-1, 1)
        x_used = pieces_used.float()

        # We combine everything
        combined = torch.cat([board_features, pieces_features, x_used, x_combo], dim=1)
        
        return self.fc(combined)
    
class BlockBlastCNNNet1P(nn.Module):
    def __init__(self, output_size=64):
        super(BlockBlastCNNNet1P, self).__init__()
        
        self.board_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), # Sortie: 32 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Sortie: 64 x 8 x 8
            nn.ReLU(),
            nn.Flatten() # Sortie: 64 * 8 * 8 = 4096
        )
        
        self.pieces_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Sortie: 32 x 5 x 5
            nn.ReLU(),
            nn.Flatten() # Sortie: 32 * 5 * 5 = 800
        )
        
        combined_input_dim = 4096 + 800
        
        self.fc = nn.Sequential(
            nn.Linear(combined_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size) # Les 192 Q-values
        )

    def forward(self, board, pieces, valid_placements):
        """
        We need to give the parameters extracted for the dictionary observation space
        """
        x_valid = valid_placements.unsqueeze(1).float()
        x_board = board.unsqueeze(1).float() 
        
        # Concatenate board and valid_placements along the channel dimension
        x_board_combined = torch.cat([x_board, x_valid], dim=1) # (batch_size, 4, 8, 8)
        
        x_pieces = pieces.unsqueeze(1).float()
        board_features = self.board_conv(x_board_combined)
        pieces_features = self.pieces_conv(x_pieces)

        # We combine everything
        combined = torch.cat([board_features, pieces_features], dim=1)
        
        return self.fc(combined)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class RainbowCNNNet1P(nn.Module):
    def __init__(self, action_size=64, num_atoms=51, v_min=-10.0, v_max=10.0):
        super(RainbowCNNNet1P, self).__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.board_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # 64 * 8 * 8 = 4096
        )
        
        self.pieces_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # 32 * 5 * 5 = 800
        )
        
        combined_input_dim = 4096 + 800
        
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_input_dim, 512),
            nn.ReLU()
        )
        
        # Dueling Network Architecture with Noisy Networks
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, num_atoms)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, action_size * num_atoms)
        )

    def forward(self, board, pieces, valid_placements):
        x_valid = valid_placements.unsqueeze(1).float()
        x_board = board.unsqueeze(1).float() 
        x_board_combined = torch.cat([x_board, x_valid], dim=1) # (batch_size, 2, 8, 8)
        
        x_pieces = pieces.unsqueeze(1).float()
        board_features = self.board_conv(x_board_combined)
        pieces_features = self.pieces_conv(x_pieces)

        combined = torch.cat([board_features, pieces_features], dim=1)
        shared_features = self.shared_fc(combined)
        
        # Distributional RL components
        value = self.value_stream(shared_features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(shared_features).view(-1, self.action_size, self.num_atoms)
        
        # Combine value and advantage: Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        # Apply Softmax to get probabilities (over atoms dimension)
        prob = F.softmax(q_dist, dim=2)
        
        return prob
    
    def reset_noise(self):
        """Reset all noisy layers in the network"""
        for module in self.value_stream:
            if hasattr(module, 'reset_noise'):
                module.reset_noise()
        for module in self.advantage_stream:
            if hasattr(module, 'reset_noise'):
                module.reset_noise()
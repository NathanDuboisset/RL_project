import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Sortie: 32 x 8 x 8
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

    def forward(self, board, pieces):
        """
        We need to give the parameters extracted for the dictionary observation space
        """
        x_board = board.unsqueeze(1).float() 
        x_pieces = pieces.float()
        board_features = self.board_conv(x_board)
        pieces_features = self.pieces_conv(x_pieces)

        # We combine everything
        combined = torch.cat([board_features, pieces_features], dim=1)
        
        return self.fc(combined)
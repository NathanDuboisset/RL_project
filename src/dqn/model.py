import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockBlastCNNNet(nn.Module):
    def __init__(self, output_size=192):
        super(BlockBlastCNNNet, self).__init__()
        
        self.board_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Sortie: 32 x 8 x 8
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

    def forward(self, board, pieces, pieces_used, combo):
        x_board = board.unsqueeze(1).float() 
        x_pieces = pieces.float()
        board_features = self.board_conv(x_board)
        pieces_features = self.pieces_conv(x_pieces)

        x_combo = combo.float().view(-1, 1)
        x_used = pieces_used.float()

        combined = torch.cat([board_features, pieces_features, x_used, x_combo], dim=1)
        
        return self.fc(combined)
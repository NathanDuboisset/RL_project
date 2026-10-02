import torch
import torch.nn as nn


class BlockBlastCNNNet(nn.Module):
    def __init__(self, output_size=192):
        super(BlockBlastCNNNet, self).__init__()

        self.board_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.pieces_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        combined_input_dim = 4096 + 800 + 3 + 1

        self.fc = nn.Sequential(
            nn.Linear(combined_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, board, pieces, pieces_used, combo, valid_placements):
        x_board = board.unsqueeze(1).float()
        x_valid = valid_placements.float()
        x_board_combined = torch.cat([x_board, x_valid], dim=1)

        x_pieces = pieces.float()
        board_features = self.board_conv(x_board_combined)
        pieces_features = self.pieces_conv(x_pieces)

        x_combo = combo.float().view(-1, 1)
        x_used = pieces_used.float()

        combined = torch.cat([board_features, pieces_features, x_used, x_combo], dim=1)
        return self.fc(combined)

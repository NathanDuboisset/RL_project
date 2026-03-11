import torch
import torch.nn as nn

class BlockBlastValueNet1P(nn.Module):
    """
    CNN
    """
    def __init__(self):
        super(BlockBlastValueNet1P, self).__init__()
        
        kernel_size = 4
        self.board_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        size_entree = 2304 
        
        self.fc = nn.Sequential(
            nn.Linear(size_entree, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, board):
        x_board = board.unsqueeze(1).float() 
        features = self.board_conv(x_board)
        out = self.fc(features)
        
        return out 
    
class BlockBlastValueNet1Pmultikernel(nn.Module):
    """
    CNN more kernels
    """
    def __init__(self):
        super(BlockBlastValueNet1Pmultikernel, self).__init__()

        sizes = [1, 2, 3, 4, 5, 8]
        line_sizes = [2, 3, 4, 5, 8] 
        n_filters = 32
        board = 8

        all_kernels = (
            [(k, k) for k in sizes] +
            [(1, k) for k in line_sizes] +
            [(k, 1) for k in line_sizes]
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, n_filters, kernel_size=(kh, kw), padding=0),
                nn.LeakyReLU(),
                nn.Flatten()
            )
            for kh, kw in all_kernels
        ])

        flat_sizes = [(board - kh + 1) * (board - kw + 1) * n_filters for kh, kw in all_kernels]
        size_entree = sum(flat_sizes)

        self.fc = nn.Sequential(
            nn.Linear(size_entree, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, board):
        x = board.unsqueeze(1).float()
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fc(features)

if __name__ == "__main__":
    model = BlockBlastValueNet1P()
    dummy_board = torch.rand(4, 8, 8)
    output = model(dummy_board)
    print(output.shape)
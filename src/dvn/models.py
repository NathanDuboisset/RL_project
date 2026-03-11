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

        sizes = [1, 3, 5, 8]
        filter_sizes = [2, 8, 8, 16]
        line_sizes = [8]
        board = 8

        all_kernels = (
            [(k, k, filter_size) for k,filter_size in zip(sizes, filter_sizes)] +
            [(1, k, 4) for k in line_sizes] +
            [(k, 1, 4) for k in line_sizes]
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filter_size, kernel_size=(kh, kw), padding=0),
                nn.LeakyReLU(),
                nn.Flatten()
            )
            for (kh, kw, filter_size) in all_kernels
        ])

        flat_sizes = [(board - kh + 1) * (board - kw + 1) * filter_size for kh, kw, filter_size in all_kernels]
        size_entree = sum(flat_sizes)

        self.fc = nn.Sequential(
            nn.Linear(size_entree, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, board):
        x = board.unsqueeze(1).float()
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fc(features)

class BlockBlastValueNet1PmultikernelFlattenned(nn.Module):
    """
    CNN more kernels
    """
    def __init__(self):
        super(BlockBlastValueNet1PmultikernelFlattenned, self).__init__()

        sizes = [1, 2, 3, 4, 5, 8]
        filter_sizes = [1, 8, 16, 16, 16, 64]
        line_sizes = [8]
        board = 8
        output_kernels=8

        all_kernels = (
            [(k, k, filter_size) for k,filter_size in zip(sizes, filter_sizes)] +
            [(1, k, 4) for k in line_sizes] +
            [(k, 1, 4) for k in line_sizes]
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filter_size, kernel_size=(kh, kw), padding=0),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear((board - kh + 1) * (board - kw + 1) * filter_size, 16),
                nn.LeakyReLU(),
                nn.Linear(16, output_kernels),
            )
            for (kh, kw, filter_size) in all_kernels
        ])

        flat_sizes = [output_kernels for kh, kw, filter_size in all_kernels]
        size_entree = sum(flat_sizes)

        self.fc = nn.Sequential(
            nn.Linear(size_entree, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
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
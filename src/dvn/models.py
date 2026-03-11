import torch
import torch.nn as nn

class BlockBlastValueNet1P(nn.Module):
    """
    CNN pour estimer la valeur d'une position dans Block Blast, 
    en se basant uniquement sur la grille résultante après un coup.
    """
    def __init__(self):
        super(BlockBlastValueNet1P, self).__init__()
        
        # Le réseau prend en entrée la grille résultante (1 seul canal)
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
            nn.Linear(256, 1) # V(S') : Valeur scalaire unique
        )

    def forward(self, board):
        """
        Calcule V(s) pour un lot (batch) de grilles.
        
        Args:
            board (torch.Tensor): Tenseur de dimension (batch_size, 8, 8)
            
        Returns:
            torch.Tensor: Tenseur de dimension (batch_size, 1) contenant les valeurs estimées.
        """
        x_board = board.unsqueeze(1).float() 
        features = self.board_conv(x_board)
        out = self.fc(features)
        
        return out 
    
class BlockBlastValueNet1Pmultikernel(nn.Module):
    """
    CNN pour estimer la valeur d'une position dans Block Blast.
    Convolutions parallèles avec kernels carrés (k×k), horizontaux (1×k)
    et verticaux (k×1) pour k in [1,2,3,4,5,8], concaténées puis MLP.
    """
    def __init__(self):
        super(BlockBlastValueNet1Pmultikernel, self).__init__()

        sizes = [1, 2, 3, 4, 5, 8]
        line_sizes = [2, 3, 4, 5, 8]  # 1×1 already covered by square k=1
        n_filters = 32
        board = 8

        # All kernel shapes: square (k,k), horizontal (1,k), vertical (k,1)
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
        """
        Args:
            board (torch.Tensor): (batch_size, 8, 8)
        Returns:
            torch.Tensor: (batch_size, 1)
        """
        x = board.unsqueeze(1).float()
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fc(features)

if __name__ == "__main__":
    # Test rapide du modèle
    model = BlockBlastValueNet1P()
    dummy_board = torch.rand(4, 8, 8)  # Batch de 4 grilles aléatoires
    output = model(dummy_board)
    print(output.shape)  # Devrait être (4, 1)
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
        self.board_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        size_entree = 2304 
        
        self.fc = nn.Sequential(
            nn.Linear(size_entree, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
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
    
if __name__ == "__main__":
    # Test rapide du modèle
    model = BlockBlastValueNet1P()
    dummy_board = torch.rand(4, 8, 8)  # Batch de 4 grilles aléatoires
    output = model(dummy_board)
    print(output.shape)  # Devrait être (4, 1)
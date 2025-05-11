import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, embedding_dim=32, num_moves=4272):
        super().__init__()
        self.embed = nn.Embedding(13, embedding_dim)  # 13 tokens -> vector
        self.fc = nn.Sequential(
            nn.Flatten(),                # [8,8,32] -> [2048]
            nn.Linear(8*8*embedding_dim, 512),
            nn.Tanh(),
            nn.Linear(512, num_moves)   # Final logits
        )


    def forward(self, x):  # x: [B, 8, 8]
        x = self.embed(x)  # [B, 8, 8, D]
        return self.fc(x)

import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, embedding_dim=32, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(13, embedding_dim)
        input_dim = 64 * embedding_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4272)
        )

    def forward(self, x):
        x = self.embed(x)  # x: [B, 64] → [B, 64, emb]
        return self.fc(x)

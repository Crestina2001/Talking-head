import torch
from torch import nn
from torch.nn import functional as F


from .conv import Conv2d
class SelfAttentionModule(nn.Module):
    def __init__(self, in_dim, heads, layer_norm = False):
        super(SelfAttentionModule, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=heads)
        self.layer_norm = nn.LayerNorm(in_dim) if layer_norm else None  # Add layer normalization

    def forward(self, x):
        # Change shape to (sequence_length, batch, features)
        B, _, _, L = x.shape
        x = x.permute(3, 0, 1, 2).squeeze(2)  # Shape: (16, B, 768)

        attn_output, _ = self.self_attention(x, x, x)
        if self.layer_norm:
            x = self.layer_norm(x + attn_output)
        # Average across the time dimension and reshape
        return attn_output.mean(dim=0).view(B, 1, -1, 1)  # Shape: (B, 1, 768, 1)

class contentvec_SyncNet_color(nn.Module):
    def __init__(self):
        super(contentvec_SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        # Define the self-attention layer
        self.audio_encoder = nn.Sequential(
            SelfAttentionModule(in_dim=768, heads=4),
            nn.Flatten(),  # Flatten the output to feed into the linear layer
            nn.Linear(768, 512),
            nn.ReLU(),  # Add ReLU activation function
            nn.Unflatten(1, (512, 1, 1))  # Reshape the output to the desired shape
        )

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding

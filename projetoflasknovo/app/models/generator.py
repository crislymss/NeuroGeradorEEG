"""Utilitários do modelo de rede neural geradora."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Rede densa que gera vetores de sinais de EEG a partir de ruído."""

    def __init__(self, latent_dim, output_dim):
        """Criar o conjunto de camadas do gerador."""
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        """Propagar o vetor de ruído `z` para sintetizar uma amostra de EEG."""
        return self.model(z)

"""Definição da rede geradora utilizada para sintetizar sinais de EEG."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Rede totalmente conectada responsável por gerar amostras de sinais de EEG."""

    def __init__(self, latent_dim, output_dim):
        """Inicializar as camadas do gerador com base nas dimensões de entrada e saída."""
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
        """Gerar uma amostra de onda a partir do vetor latente `z`."""
        return self.model(z)

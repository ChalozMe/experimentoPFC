#!/usr/bin/env python3
"""
Demo: Genera audio original y steered con parámetro modificable.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_FRAMES = 128
DURATION = 3.0
TARGET_LENGTH = int(SR * DURATION)

DATA_DIR = Path("/home/chalo/pfc/data")
MODELS_DIR = Path("/home/chalo/pfc/models")
ARRAYS_DIR = Path("/home/chalo/pfc/arrays")
DEMO_DIR = Path("/home/chalo/pfc/results/demo")


class VAEConv(nn.Module):
    def __init__(self, n_mels=128, n_frames=128, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 4, 4)
        return self.decoder(h)


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, k=16):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = k
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        h = self.encoder(x)
        topk_values, topk_indices = torch.topk(h, self.k, dim=-1)
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(-1, topk_indices, F.relu(topk_values))
        x_recon = self.decoder(h_sparse)
        return x_recon, h_sparse
    
    def encode(self, x):
        h = self.encoder(x)
        topk_values, topk_indices = torch.topk(h, self.k, dim=-1)
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(-1, topk_indices, F.relu(topk_values))
        return h_sparse


def spec_to_audio(spec_log):
    spec_db = spec_log * 80 - 80
    spec_mag = librosa.db_to_power(spec_db)
    audio = librosa.feature.inverse.mel_to_audio(spec_mag, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return audio


def load_audio(path):
    y, sr = sf.read(path, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]
    elif len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)), mode='constant')
    return y


def audio_to_spec(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_log = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_log - S_log.min()) / (S_log.max() - S_log.min() + 1e-8)
    if S_norm.shape[1] > N_FRAMES:
        S_norm = S_norm[:, :N_FRAMES]
    elif S_norm.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    return S_norm.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Demo de steering en VAE+SAE")
    parser.add_argument("--input", type=str, default="data/raw/audio/DistortionFX/Crush/0/5-5_3.wav",
                        help="Path al audio de input")
    parser.add_argument("--steering", type=float, default=2.0,
                        help="Valor de steering (modificación de la característica)")
    parser.add_argument("--feature", type=int, default=94,
                        help="Índice de característica a manipular")
    parser.add_argument("--output", type=str, default="results/demo",
                        help="Carpeta de salida")
    args = parser.parse_args()
    
    DEMO_DIR = Path(args.output)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DEMO: Steering en VAE + SAE")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Steering: {args.steering}")
    print(f"Feature: {args.feature}")
    print(f"Output: {DEMO_DIR}")
    
    # Cargar modelos
    print("\nCargando modelos...")
    vae = VAEConv().to(DEVICE)
    vae.load_state_dict(torch.load(MODELS_DIR / "vae_best.pt"))
    vae.eval()
    
    sae = SparseAutoencoder().to(DEVICE)
    sae.load_state_dict(torch.load(MODELS_DIR / "sae_best.pt"))
    sae.eval()
    
    # Cargar audio
    print(f"Cargando audio: {args.input}")
    audio_input = load_audio(args.input)
    spec_input = audio_to_spec(audio_input)
    
    # Codificar
    print("Codificando...")
    spec_tensor = torch.from_numpy(spec_input).unsqueeze(0).unsqueeze(1).float().to(DEVICE)
    with torch.no_grad():
        mu, _ = vae.encode(spec_tensor)
        z_original = mu.cpu().numpy().flatten()
        h_original = sae.encode(torch.from_numpy(z_original).float().to(DEVICE)).cpu().numpy().flatten()
    
    # Steering
    print(f"Aplicando steering: feature {args.feature} += {args.steering}")
    h_modificado = h_original.copy()
    h_modificado[args.feature] = h_original[args.feature] + args.steering
    
    # Decodificar
    print("Decodificando...")
    h_tensor = torch.from_numpy(h_modificado).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_modificado = sae.decoder(h_tensor).cpu().numpy().flatten()
        z_tensor = torch.from_numpy(z_modificado).float().unsqueeze(0).to(DEVICE)
        spec_modificado = vae.decode(z_tensor).cpu().numpy()[0, 0]
    
    # Convertir a audio
    print("Generando audios...")
    audio_original = spec_to_audio(spec_input)
    audio_steered = spec_to_audio(spec_modificado)
    
    # Guardar
    sf.write(DEMO_DIR / "demo_original.wav", audio_original, SR)
    sf.write(DEMO_DIR / "demo_steered.wav", audio_steered, SR)
    
    # Guardar imagen comparativa
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(spec_input, origin='lower', aspect='auto', cmap='viridis')
    axes[0].set_title('Original')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Mel bin')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(spec_modificado, origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_title(f'Steered (feature {args.feature} +{args.steering})')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Mel bin')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(DEMO_DIR / "demo_steering.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nDemo guardada en: {DEMO_DIR}")
    print(f"  - demo_original.wav")
    print(f"  - demo_steered.wav")
    print(f"  - demo_steering.png")


if __name__ == "__main__":
    main()

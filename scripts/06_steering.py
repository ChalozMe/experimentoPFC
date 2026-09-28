#!/usr/bin/env python3
"""
Fase 6: Experimento de Steering
Manipula características del SAE en el espacio latente y genera audio modificado.
Demuestra que las características aprendidas son causales.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import soundfile as sf
import librosa
from tqdm import tqdm
import json

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
RESULTS_DIR = Path("/home/chalo/pfc/results")
STEERING_DIR = RESULTS_DIR / "steering"

STEERING_DIR.mkdir(parents=True, exist_ok=True)


class VAEConv(nn.Module):
    """VAE convolucional (misma arquitectura que en entrenamiento)."""
    
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
    """SAE con TopK (misma arquitectura que en entrenamiento)."""
    
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
    """Convierte espectrograma log-magnitud a audio usando Griffin-Lim."""
    # Des-normalizar de [0,1] a dB aproximado
    spec_db = spec_log * 80 - 80  # rango aproximado [-80, 0] dB
    
    # Convertir de dB a magnitud
    spec_mag = librosa.db_to_power(spec_db)
    
    # Griffin-Lim para reconstruir fase
    audio = librosa.feature.inverse.mel_to_audio(
        spec_mag,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    return audio


def load_audio(path):
    """Carga un archivo de audio."""
    y, sr = sf.read(path, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    
    # Ajustar longitud
    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]
    elif len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)), mode='constant')
    
    return y


def audio_to_spec(y):
    """Convierte audio a espectrograma Mel log-magnitud normalizado [0,1]."""
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    S_log = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_log - S_log.min()) / (S_log.max() - S_log.min() + 1e-8)
    
    if S_norm.shape[1] > N_FRAMES:
        S_norm = S_norm[:, :N_FRAMES]
    elif S_norm.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    
    return S_norm.astype(np.float32)


def main():
    print("=" * 60)
    print("FASE 6: Experimento de Steering")
    print("=" * 60)
    
    # Cargar modelos
    print("Cargando modelos...")
    vae = VAEConv().to(DEVICE)
    vae.load_state_dict(torch.load(MODELS_DIR / "vae_best.pt"))
    vae.eval()
    
    sae = SparseAutoencoder().to(DEVICE)
    sae.load_state_dict(torch.load(MODELS_DIR / "sae_best.pt"))
    sae.eval()
    
    # Cargar datos
    H = np.load(ARRAYS_DIR / "caracteristicas_sae.npy")
    Z = np.load(ARRAYS_DIR / "latentes_vae.npy")
    etiquetas = np.load(DATA_DIR / "labels.npy").flatten()
    
    print(f"Características: {H.shape}, Latentes: {Z.shape}")
    
    # Encontrar la característica más correlacionada con el nivel de distorsión
    print("\nBuscando característica más correlacionada con el descriptor...")
    correlaciones = []
    for j in range(H.shape[1]):
        corr = np.corrcoef(H[:, j], etiquetas)[0, 1]
        if not np.isnan(corr):
            correlaciones.append((j, abs(corr)))
    
    correlaciones.sort(key=lambda x: x[1], reverse=True)
    top_features = correlaciones[:5]
    
    print("Top 5 características más correlacionadas:")
    for j, corr in top_features:
        print(f"  Característica {j}: r = {corr:.3f}")
    
    # Usar la más correlacionada para steering
    feature_idx = top_features[0][0]
    print(f"\nUsando característica {feature_idx} para steering")
    
    # Seleccionar muestras de entrada (nivel bajo de distorsión)
    idx_bajo = np.where(etiquetas < 30)[0][:5]
    print(f"Seleccionadas {len(idx_bajo)} muestras con nivel bajo de distorsión")
    
    # Para cada muestra, generar versión modificada
    resultados = []
    
    for i, idx in enumerate(tqdm(idx_bajo, desc="Generando steering")):
        # Cargar audio original
        spec_path = DATA_DIR / "specs" / f"{idx:06d}.npy"
        spec_original = np.load(spec_path)
        
        # Codificar a latente
        spec_tensor = torch.from_numpy(spec_original).unsqueeze(0).unsqueeze(1).float().to(DEVICE)
        with torch.no_grad():
            mu, _ = vae.encode(spec_tensor)
            z_original = mu.cpu().numpy()[0]
        
        # Obtener características SAE
        with torch.no_grad():
            h_original = sae.encode(torch.from_numpy(z_original).float().to(DEVICE)).cpu().numpy().flatten()
        
        # STEERING: aumentar la característica seleccionada
        h_modificado = h_original.copy()
        h_modificado[feature_idx] = h_original[feature_idx] + 2.0  # Aumentar activación
        
        # Decodificar a latente modificado
        h_tensor = torch.from_numpy(h_modificado).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z_modificado = sae.decoder(h_tensor).cpu().numpy().flatten()
        
        # Decodificar a espectrograma
        z_tensor = torch.from_numpy(z_modificado).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            spec_modificado = vae.decode(z_tensor).cpu().numpy()[0, 0]
        
        # Convertir a audio
        audio_original = spec_to_audio(spec_original)
        audio_modificado = spec_to_audio(spec_modificado)
        
        # Guardar audios
        sf.write(STEERING_DIR / f"original_{i}.wav", audio_original, SR)
        sf.write(STEERING_DIR / f"steered_{i}.wav", audio_modificado, SR)
        
        # Guardar espectrogramas como imágenes
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(spec_original, origin='lower', aspect='auto')
        axes[0].set_title(f'Original (nivel={etiquetas[idx]:.0f})')
        axes[1].imshow(spec_modificado, origin='lower', aspect='auto')
        axes[1].set_title(f'Steered (feature {feature_idx} +2)')
        plt.tight_layout()
        plt.savefig(STEERING_DIR / f"steering_{i}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        resultados.append({
            "idx": int(idx),
            "nivel_original": float(etiquetas[idx]),
            "feature_idx": int(feature_idx),
            "activacion_original": float(h_original[feature_idx]),
            "activacion_modificada": float(h_modificado[feature_idx])
        })
    
    # Guardar resumen
    with open(STEERING_DIR / "steering_summary.json", 'w') as f:
        json.dump({
            "feature_idx": int(feature_idx),
            "top_features": [(int(j), float(c)) for j, c in top_features],
            "resultados": resultados
        }, f, indent=2)
    
    print(f"\nSteering completado. Archivos guardados en: {STEERING_DIR}")
    print(f"  - {len(idx_bajo)} pares de audios (original + steered)")
    print(f"  - {len(idx_bajo)} imágenes comparativas")
    print(f"  - steering_summary.json")


if __name__ == "__main__":
    main()

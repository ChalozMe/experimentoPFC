#!/usr/bin/env python3
"""
Fase 2: Entrenamiento del VAE de timbre.
Arquitectura convolucional con latent de 128 dimensiones.
Reducido a 30 épocas para resultados rápidos.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import json
from tqdm import tqdm

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 30  # Reducido para resultados rápidos
LR = 1e-3
BETA = 0.001  # Peso KL
LATENT_DIM = 128
N_MELS = 128
N_FRAMES = 128

DATA_DIR = Path("/home/chalo/pfc/data")
MODELS_DIR = Path("/home/chalo/pfc/models")
ARRAYS_DIR = Path("/home/chalo/pfc/arrays")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARRAYS_DIR.mkdir(parents=True, exist_ok=True)


class VAEConv(nn.Module):
    """VAE convolucional para espectrogramas."""
    
    def __init__(self, n_mels=128, n_frames=128, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
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
        
        # Decoder: 4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128
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
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def vae_loss(x_recon, x, mu, logvar, beta=0.001):
    """Pérdida VAE = Reconstrucción (MSE) + β * KL divergence."""
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def load_spectrograms(data_dir: Path) -> torch.Tensor:
    """Carga todos los espectrogramas cacheados."""
    specs_dir = data_dir / "specs"
    spec_files = sorted(specs_dir.glob("*.npy"))
    
    print(f"Cargando {len(spec_files)} espectrogramas...")
    specs = []
    for f in tqdm(spec_files, desc="Cargando"):
        spec = np.load(f)
        specs.append(spec)
    
    specs_array = np.stack(specs)
    # Añadir dimensión de canal: (N, 1, H, W)
    specs_tensor = torch.from_numpy(specs_array).unsqueeze(1).float()
    return specs_tensor


def train_vae():
    print("=" * 60)
    print("FASE 2: Entrenamiento del VAE")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, Épocas: {EPOCHS}, LR: {LR}, β: {BETA}")
    
    # Cargar datos
    specs = load_spectrograms(DATA_DIR)
    print(f"Shape de datos: {specs.shape}")
    
    # Split train/val/test (80/10/10)
    n_total = len(specs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_data = specs[train_idx]
    val_data = specs[val_idx]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE, shuffle=False)
    
    # Modelo
    model = VAEConv(n_mels=N_MELS, n_frames=N_FRAMES, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()
    
    history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Entrenamiento
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            x = batch[0].to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                x_recon, mu, logvar, _ = model(x)
                loss, recon, kl = vae_loss(x_recon, x, mu, logvar, BETA)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                with autocast():
                    x_recon, mu, logvar, _ = model(x)
                    loss, _, _ = vae_loss(x_recon, x, mu, logvar, BETA)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_recon = train_recon / len(train_loader)
        avg_kl = train_kl / len(train_loader)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")
        
        # Guardar mejor modelo
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODELS_DIR / "vae_best.pt")
    
    # Guardar modelo final
    torch.save(model.state_dict(), MODELS_DIR / "vae_final.pt")
    
    # Guardar historial
    with open(MODELS_DIR / "vae_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nEntrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")
    
    # Extraer latentes
    print("\nExtrayendo latentes...")
    model.load_state_dict(torch.load(MODELS_DIR / "vae_best.pt"))
    model.eval()
    
    all_latents = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(TensorDataset(specs), batch_size=BATCH_SIZE), desc="Extrayendo latentes"):
            x = batch[0].to(DEVICE)
            mu, _ = model.encode(x)
            all_latents.append(mu.cpu().numpy())
    
    latents = np.concatenate(all_latents, axis=0)
    np.save(ARRAYS_DIR / "latentes_vae.npy", latents)
    print(f"Latentes guardados: {latents.shape}")


if __name__ == "__main__":
    train_vae()

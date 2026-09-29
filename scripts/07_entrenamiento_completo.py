#!/usr/bin/env python3
"""
Entrenamiento completo del pipeline con dataset completo (~275K muestras).
Tiempo estimado: 6-8 horas.
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
import time

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE_VAE = 16
BATCH_SIZE_SAE = 512
EPOCHS_VAE = 150  # ~5-6 horas
EPOCHS_SAE = 300  # ~1-2 horas
LR = 1e-3
BETA = 0.001
LATENT_DIM = 128
N_MELS = 128
N_FRAMES = 128
HIDDEN_DIM = 512
K = 16

DATA_DIR = Path("/home/chalo/pfc/data")
MODELS_DIR = Path("/home/chalo/pfc/models")
ARRAYS_DIR = Path("/home/chalo/pfc/arrays")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARRAYS_DIR.mkdir(parents=True, exist_ok=True)


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


def vae_loss(x_recon, x, mu, logvar, beta=0.001):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def load_spectrograms(data_dir: Path) -> torch.Tensor:
    specs_dir = data_dir / "specs"
    spec_files = sorted(specs_dir.glob("*.npy"))
    print(f"Cargando {len(spec_files)} espectrogramas...")
    specs = []
    for f in tqdm(spec_files, desc="Cargando"):
        spec = np.load(f)
        specs.append(spec)
    specs_array = np.stack(specs)
    specs_tensor = torch.from_numpy(specs_array).unsqueeze(1).float()
    return specs_tensor


def train_vae():
    print("=" * 60)
    print("ENTRENAMIENTO VAE - DATASET COMPLETO")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE_VAE}, Épocas: {EPOCHS_VAE}, LR: {LR}, β: {BETA}")
    
    specs = load_spectrograms(DATA_DIR)
    print(f"Shape de datos: {specs.shape}")
    
    n_total = len(specs)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    
    train_data = specs[train_idx]
    val_data = specs[val_idx]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE_VAE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE_VAE, shuffle=False)
    
    model = VAEConv(n_mels=N_MELS, n_frames=N_FRAMES, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    scaler = GradScaler()
    
    history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(EPOCHS_VAE):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_VAE}")
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
        
        scheduler.step(avg_val)
        
        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (EPOCHS_VAE - epoch - 1)
        
        print(f"Epoch {epoch+1}/{EPOCHS_VAE} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | ETA: {eta/3600:.1f}h")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODELS_DIR / "vae_best.pt")
    
    torch.save(model.state_dict(), MODELS_DIR / "vae_final.pt")
    
    with open(MODELS_DIR / "vae_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nVAE completado. Mejor val_loss: {best_val_loss:.4f}")
    
    # Extraer latentes
    print("Extrayendo latentes...")
    model.load_state_dict(torch.load(MODELS_DIR / "vae_best.pt"))
    model.eval()
    
    all_latents = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(TensorDataset(specs), batch_size=BATCH_SIZE_VAE), desc="Extrayendo latentes"):
            x = batch[0].to(DEVICE)
            mu, _ = model.encode(x)
            all_latents.append(mu.cpu().numpy())
    
    latents = np.concatenate(all_latents, axis=0)
    np.save(ARRAYS_DIR / "latentes_vae.npy", latents)
    print(f"Latentes guardados: {latents.shape}")


def train_sae():
    print("=" * 60)
    print("ENTRENAMIENTO SAE - DATASET COMPLETO")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    print(f"Input: {LATENT_DIM}, Hidden: {HIDDEN_DIM}, K: {K}")
    print(f"Batch size: {BATCH_SIZE_SAE}, Épocas: {EPOCHS_SAE}, LR: {LR}")
    
    latents = np.load(ARRAYS_DIR / "latentes_vae.npy")
    print(f"Shape de latentes: {latents.shape}")
    
    n_total = len(latents)
    n_train = int(0.9 * n_total)
    
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_data = torch.from_numpy(latents[train_idx]).float()
    val_data = torch.from_numpy(latents[val_idx]).float()
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE_SAE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE_SAE, shuffle=False)
    
    model = SparseAutoencoder(input_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, k=K).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    history = {'train_loss': [], 'val_loss': [], 'sparsity': []}
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(EPOCHS_SAE):
        model.train()
        train_loss = 0
        train_sparsity = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_SAE}")
        for batch in pbar:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            x_recon, h_sparse = model(x)
            loss = F.mse_loss(x_recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            sparsity = (h_sparse > 0).float().mean().item()
            train_sparsity += sparsity
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'sparsity': f'{sparsity:.4f}'})
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                x_recon, _ = model(x)
                loss = F.mse_loss(x_recon, x)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_sparsity = train_sparsity / len(train_loader)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['sparsity'].append(avg_sparsity)
        
        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (EPOCHS_SAE - epoch - 1)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_SAE} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Sparsity: {avg_sparsity:.4f} | ETA: {eta/3600:.1f}h")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODELS_DIR / "sae_best.pt")
    
    torch.save(model.state_dict(), MODELS_DIR / "sae_final.pt")
    
    with open(MODELS_DIR / "sae_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nSAE completado. Mejor val_loss: {best_val_loss:.4f}")
    
    # Extraer características
    print("Extrayendo características...")
    model.load_state_dict(torch.load(MODELS_DIR / "sae_best.pt"))
    model.eval()
    
    all_features = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(TensorDataset(torch.from_numpy(latents).float()), batch_size=BATCH_SIZE_SAE), desc="Extrayendo características"):
            x = batch[0].to(DEVICE)
            h = model.encode(x)
            all_features.append(h.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    np.save(ARRAYS_DIR / "caracteristicas_sae.npy", features)
    print(f"Características guardadas: {features.shape}")


def main():
    print("=" * 60)
    print("ENTRENAMIENTO COMPLETO - DATASET COMPLETO")
    print("=" * 60)
    print(f"Épocas VAE: {EPOCHS_VAE}")
    print(f"Épocas SAE: {EPOCHS_SAE}")
    print(f"Tiempo estimado: 6-8 horas")
    
    train_vae()
    train_sae()
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()

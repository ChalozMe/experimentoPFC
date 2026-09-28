#!/usr/bin/env python3
"""
Fase 3: Entrenamiento del SAE (Sparse Autoencoder) sobre latentes del VAE.
Reducido a 50 épocas para resultados rápidos.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
from tqdm import tqdm

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512
EPOCHS = 50  # Reducido para resultados rápidos
LR = 1e-3
INPUT_DIM = 128
HIDDEN_DIM = 512
K = 16  # TopK activaciones

DATA_DIR = Path("/home/chalo/pfc/data")
MODELS_DIR = Path("/home/chalo/pfc/models")
ARRAYS_DIR = Path("/home/chalo/pfc/arrays")


class SparseAutoencoder(nn.Module):
    """Autoencoder disperso con TopK."""
    
    def __init__(self, input_dim=128, hidden_dim=512, k=16):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = k
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        h = self.encoder(x)
        # TopK: solo mantener las k activaciones más grandes
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


def train_sae():
    print("=" * 60)
    print("FASE 3: Entrenamiento del SAE")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    print(f"Input: {INPUT_DIM}, Hidden: {HIDDEN_DIM}, K: {K}")
    print(f"Batch size: {BATCH_SIZE}, Épocas: {EPOCHS}, LR: {LR}")
    
    # Cargar latentes
    latents = np.load(ARRAYS_DIR / "latentes_vae.npy")
    print(f"Shape de latentes: {latents.shape}")
    
    # Split train/val (90/10)
    n_total = len(latents)
    n_train = int(0.9 * n_total)
    
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_data = torch.from_numpy(latents[train_idx]).float()
    val_data = torch.from_numpy(latents[val_idx]).float()
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=BATCH_SIZE, shuffle=False)
    
    # Modelo
    model = SparseAutoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, k=K).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    history = {'train_loss': [], 'val_loss': [], 'sparsity': []}
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Entrenamiento
        model.train()
        train_loss = 0
        train_sparsity = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
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
        
        # Validación
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Sparsity: {avg_sparsity:.4f}")
        
        # Guardar mejor modelo
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODELS_DIR / "sae_best.pt")
    
    # Guardar modelo final
    torch.save(model.state_dict(), MODELS_DIR / "sae_final.pt")
    
    # Guardar historial
    with open(MODELS_DIR / "sae_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nEntrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")
    
    # Extraer características
    print("\nExtrayendo características...")
    model.load_state_dict(torch.load(MODELS_DIR / "sae_best.pt"))
    model.eval()
    
    all_features = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(TensorDataset(torch.from_numpy(latents).float()), batch_size=BATCH_SIZE), desc="Extrayendo características"):
            x = batch[0].to(DEVICE)
            h = model.encode(x)
            all_features.append(h.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    np.save(ARRAYS_DIR / "caracteristicas_sae.npy", features)
    print(f"Características guardadas: {features.shape}")


if __name__ == "__main__":
    train_sae()

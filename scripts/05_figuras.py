#!/usr/bin/env python3
"""
Fase 5: Generación de figuras y visualizaciones.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuración
ARRAYS_DIR = Path("/home/chalo/pfc/arrays")
MODELS_DIR = Path("/home/chalo/pfc/models")
RESULTS_DIR = Path("/home/chalo/pfc/results")
FIGURAS_DIR = RESULTS_DIR / "figuras"

FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def plot_convergencia_vae():
    """Figura 1: Convergencia del VAE."""
    with open(MODELS_DIR / "vae_history.json") as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('VAE: Pérdida Total')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history['recon_loss'])
    axes[1].set_title('VAE: Pérdida Reconstrucción')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('MSE')
    
    axes[2].plot(history['kl_loss'])
    axes[2].set_title('VAE: Divergencia KL')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('KL')
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "fig1_convergencia_vae.png", bbox_inches='tight')
    plt.close()
    print("✓ Figura 1: Convergencia VAE")


def plot_convergencia_sae():
    """Figura 2: Convergencia del SAE."""
    with open(MODELS_DIR / "sae_history.json") as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('SAE: Pérdida de Reconstrucción')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    
    axes[1].plot(history['sparsity'])
    axes[1].set_title('SAE: Esparsidad')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Proporción activa')
    axes[1].axhline(y=16/512, color='r', linestyle='--', label='Objetivo (16/512)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "fig2_convergencia_sae.png", bbox_inches='tight')
    plt.close()
    print("✓ Figura 2: Convergencia SAE")


def plot_galaxia():
    """Figura 3: Ley de potencias (galaxia)."""
    with open(RESULTS_DIR / "metricas.json") as f:
        metricas = json.load(f)
    
    # Recalcular eigenvalores para la figura
    H = np.load(ARRAYS_DIR / "caracteristicas_sae.npy")
    cov = np.cov(H.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    log_rank = np.log(np.arange(1, len(eigenvalues) + 1))
    log_eigen = np.log(eigenvalues)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(log_rank, log_eigen, alpha=0.6, s=20)
    
    # Ajuste lineal
    coeffs = np.polyfit(log_rank, log_eigen, 1)
    ax.plot(log_rank, np.polyval(coeffs, log_rank), 'r--', 
            label=f'Ajuste: α = {metricas["galaxia"]["alpha"]:.3f}')
    
    ax.set_xlabel('log(rank)')
    ax.set_ylabel('log(eigenvalue)')
    ax.set_title('Análisis de Galaxia: Distribución de Eigenvalores')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "fig3_galaxia.png", bbox_inches='tight')
    plt.close()
    print("✓ Figura 3: Galaxia")


def plot_sondas():
    """Figura 4: F1 por descriptor."""
    with open(RESULTS_DIR / "metricas.json") as f:
        metricas = json.load(f)
    
    sondas = metricas['sondas_lineales']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(['Descriptor'], [sondas['mejor_f1']], color='steelblue')
    ax.axvline(x=0.5, color='r', linestyle='--', label='Umbral F1=0.5')
    ax.set_xlabel('Mejor F1 Score')
    ax.set_title('Sondas Lineales: F1 del Descriptor')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "fig4_sondas.png", bbox_inches='tight')
    plt.close()
    print("✓ Figura 4: Sondas lineales")


def plot_lobulos():
    """Figura 5: Distribución de clusters (lóbulos)."""
    with open(RESULTS_DIR / "metricas.json") as f:
        metricas = json.load(f)
    
    dist = metricas['lobulos']['distribucion_clusters']
    clusters = list(dist.keys())
    counts = list(dist.values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(clusters, counts, color='coral')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Número de características')
    ax.set_title(f"Análisis de Lóbulos: {metricas['lobulos']['mejor_n_clusters']} clusters\nSilhouette: {metricas['lobulos']['silhouette_score']:.3f}")
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "fig5_lobulos.png", bbox_inches='tight')
    plt.close()
    print("✓ Figura 5: Lóbulos")


def main():
    print("=" * 60)
    print("FASE 5: Generación de Figuras")
    print("=" * 60)
    
    plot_convergencia_vae()
    plot_convergencia_sae()
    plot_galaxia()
    plot_sondas()
    plot_lobulos()
    
    print(f"\nFiguras guardadas en: {FIGURAS_DIR}")


if __name__ == "__main__":
    main()

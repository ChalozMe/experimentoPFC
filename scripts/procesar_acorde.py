#!/usr/bin/env python3
"""
Procesa un acorde aplicando steering a cada cuerda por separado mediante VAE+SAE.

Flujo:
1. Carga espectrogramas de notas individuales (cuerdas)
2. Para cada cuerda:
   a. Codifica a latente Z con VAE
   ̇   b. Codifica a características H con SAE
   c. Aplica steering (activa/desactiva características específicas)
   d. Decodifica de vuelta a latente Z'
   d. Decodifica a espectrograma modificado
3. Suma todos los espectrogramas modificados
4. Genera audio final con Griffin-Lim

Uso:
    python scripts/08_procesar_acorde.py --notas "5-5_3,6-5_3,4-5_3,2-5_3,0-5_3,5-4_3" --feature 94 --steering 2.0
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import soundfile as sf
import librosa

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
SPECS_DIR = DATA_DIR / "specs"


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


def audio_to_spec(y):
    """Convierte audio a espectrograma Mel log-magnitud normalizado [0,1]."""
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_log = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_log - S_log.min()) / (S_log.max() - S_log.min() + 1e-8)
    if S_norm.shape[1] > N_FRAMES:
        S_norm = S_norm[:, :N_FRAMES]
    elif S_norm.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    return S_norm.astype(np.float32)


def spec_to_audio(spec_log, duration=DURATION):
    """Convierte espectrograma log-magnitud a audio usando Griffin-Lim."""
    spec_db = spec_log * 80 - 80
    spec_mag = librosa.db_to_power(spec_db)
    audio = librosa.feature.inverse.mel_to_audio(spec_mag, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return audio


def cargar_modelos():
    """Carga VAE y SAE entrenados."""
    vae = VAEConv().to(DEVICE)
    vae.load_state_dict(torch.load(MODELS_DIR / "vae_best.pt"))
    vae.eval()

    sae = SparseAutoencoder().to(DEVICE)
    sae.load_state_dict(torch.load(MODELS_DIR / "sae_best.pt"))
    sae.eval()

    return vae, sae


def procesar_cuadra(spec, vae, sae, feature_idx, steering_val):
    """
    Procesa un espectrograma de una cuerda: codifica, aplica steering, decodifica.
    """
    spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(1).float().to(DEVICE)

    # Codificar a latente
    mu, _ = vae.encode(spec_tensor)
    z = mu

    # Codificar a características SAE
    h = sae.encode(z)

    # Aplicar steering: modificar la característica seleccionada
    h_mod = h.clone()
    h_mod[0, feature_idx] = h[0, feature_idx] + steering_val

    # Decodificar de características a latente
    h_tensor = h_mod.float().to(DEVICE)
    z_mod = sae.decoder(h_tensor)

    # Decodificar de latente a espectrograma
    z_tensor = z_mod.float().to(DEVICE)
    spec_mod = vae.decode(z_tensor).detach()

    return spec_mod.cpu().numpy()[0, 0]


def procesar_acorde(notas_ids, feature_idx, steering_val, output_path=None):
    """
    Procesa un acorde completo aplicando steering a cada cuerda por separado.
    """
    print(f"Procesando acorde con {len(notas_ids)} notas: {notas_ids}")
    print(f"Característica: {feature_idx}, Steering: {steering_val}")

    # Cargar modelos
    print("Cargando modelos VAE y SAE...")
    vae, sae = cargar_modelos()
    print("Modelos cargados.")

    # Procesar cada cuerda
    specs_modificados = []
    for nota_id in notas_ids:
        spec_path = DATA_DIR / "specs" / f"{nota_id}.npy"
        if not spec_path.exists():
            print(f"  ADVERTENCIA: No se encontró {spec_path}, saltando...")
            continue

        spec = np.load(spec_path)
        print(f"  Procesando cuerda: {nota_id} -> shape {spec.shape}")

        spec_mod = procesar_cuadra(spec, vae, sae, feature_idx, steering_val)
        specs_modificados.append(spec_mod)
        print(f"    -> Procesado exitosamente")

    if not specs_modificados:
        raise ValueError("No se pudo procesar ninguna cuerda")

    # Sumar espectrogramas modificados (mezcla del acorde)
    print("\nCombinando espectrogramas modificados...")
    spec_combinado = np.sum(specs_modificados, axis=0)

    # Normalizar para evitar saturación
    spec_combinado = np.clip(spec_combinado, 0, 1)

    # Convertir a audio
    print("Generando audio con Griffin-Lim...")
    audio_final = spec_to_audio(spec_combinado)

    # Guardar
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio_final, SR)
        print(f"\nAudio guardado en: {output_path}")

    # También guardar el espectrograma combinado
    spec_path_out = str(output_path).replace('.wav', '.npy') if output_path else None
    if spec_path_out:
        np.save(spec_path_out, spec_combinado)
        print(f"Espectrograma guardado en: {spec_path_out}")

    return audio_final, spec_combinado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa un acorde aplicando steering a cada cuerda")
    parser.add_argument("--notas", type=str, required=True,
                        help="Lista de IDs de notas separados por comas")
    parser.add_argument("--feature", type=int, default=94,
                        help="Índice de característica a manipular")
    parser.add_argument("--steering", type=float, default=2.0,
                        help="Valor de steering (magnitud de modificación)")
    parser.add_argument("--output", type=str, default="results/demo/acorde_steered.wav",
                        help="Archivo de salida")

    args = parser.parse_args()

    notas = [n.strip() for n in args.notas.split(",")]
    procesar_acorde(notas, args.feature, args.steering, args.output)

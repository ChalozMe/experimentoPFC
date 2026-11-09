#!/usr/bin/env python3
"""
Fase 3: Procesamiento de una cuerda individual con VAE + SAE + Steering.

Este módulo contiene las funciones para:
1. Cargar modelos VAE y SAE entrenados
2. Codificar un espectrograma a latente (VAE.encode)
3. Codificar latente a características SAE (SAE.encode)
4. Aplicar steering a una característica específica
5. Decodificar características a latente (SAE.decoder)
6. Decodificar latente a espectrograma (VAE.decode)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import soundfile as sf
import librosa

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
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


# =============================================================================
# MODELO VAE (misma arquitectura que en entrenamiento)
# =============================================================================
class VAEConv(nn.Module):
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

        # Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
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


# =============================================================================
# MODELO SAE (Autoencoder Disperso)
# =============================================================================
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


# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# =============================================================================

def audio_to_spec(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                  n_mels=N_MELS, n_frames=N_FRAMES):
    """Convierte señal de audio a espectrograma Mel log-magnitud normalizado [0,1]."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    S_log = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_log - S_log.min()) / (S_log.max() - S_log.min() + 1e-8)
    if S_norm.shape[1] > n_frames:
        S_norm = S_norm[:, :n_frames]
    elif S_norm.shape[1] < n_frames:
        pad_width = n_frames - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    return S_norm.astype(np.float32)


def spec_to_audio(spec_log, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convierte espectrograma log-magnitud a audio usando Griffin-Lim."""
    spec_db = spec_log * 80 - 80
    spec_mag = librosa.db_to_power(spec_db)
    audio = librosa.feature.inverse.mel_to_audio(spec_mag, sr=sr,
                                                  n_fft=n_fft,
                                                  hop_length=hop_length)
    return audio


def cargar_espectrograma(ruta_npy):
    """Carga un espectrograma .npy existente."""
    return np.load(ruta_npy).astype(np.float32)


# =============================================================================
# CARGA DE MODELOS
# =============================================================================

def cargar_modelos():
    """Carga modelos VAE y SAE entrenados."""
    print("Cargando modelos VAE y SAE...")
    vae = VAEConv().to(DEVICE)
    vae.load_state_dict(torch.load(MODELS_DIR / "vae_best.pt"))
    vae.eval()

    sae = SparseAutoencoder().to(DEVICE)
    sae.load_state_dict(torch.load(MODELS_DIR / "sae_best.pt"))
    sae.eval()

    print("Modelos cargados correctamente.")
    return vae, sae


# =============================================================================
# PROCESAMIENTO DE UNA CUERDA
# =============================================================================

def procesar_cuadra(spec, vae, sae, feature_idx, steering_val):
    """
    Procesa una sola cuerda (espectrograma de una nota):
      1. Codificar a latente Z con VAE
      2. Codificar a características H con SAE
      3. Aplicar steering a la característica especificada
      4. Decodificar a latente Z modificado
      5. Decodificar a espectrograma modificado
    """
    spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(1).float().to(DEVICE)

    # Codificar a latente Z
    with torch.no_grad():
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
    with torch.no_grad():
        spec_mod = vae.decode(z_tensor).detach().cpu().numpy()[0, 0]

    return spec_mod


# =============================================================================
# PROCESAMIENTO DE ACORDE COMPLETO
# =============================================================================

def procesar_acorde(notas_ids, feature_idx, steering_val,
                    output_path=None):
    """
    Procesa un acorde completo aplicando steering a cada cuerda por separado.

    Args:
        notas_ids: Lista de IDs de archivos .wav (ej: ["1-0.wav", "2-5_1.wav"])
        feature_idx: Índice de característica SAE a manipular
        steering_val: Valor de steering (magnitud de modificación)
        output_path: Ruta para guardar el audio de salida

    Returns:
        (audio_final, espectrograma_combinado)
    """
    print(f"\n{'='*60}")
    print(f"PROCESANDO ACORDE CON {len(notas_ids)} NOTAS")
    print(f"Notas: {notas_ids}")
    print(f"Característica SAE: {feature_idx}")
    print(f"Steering: {steering_val}")
    print("=" * 60)

    # Cargar modelos
    print("\nCargando modelos VAE y SAE...")
    vae, sae = cargar_modelos()
    print("Modelos cargados correctamente.\n")

    # Procesar cada cuerda
    specs_modificados = []
    notas_procesadas = []

    for nota_id in notas_ids:
        spec_path = SPECS_DIR / f"{nota_id}.npy"
        if not spec_path.exists():
            print(f"  ADVERTENCIA: No se encontró {spec_path}, saltando...")
            continue

        spec = np.load(spec_path)
        print(f"  Procesando cuerda: {nota_id} -> shape {spec.shape}")

        spec_mod = procesar_cuadra(spec, vae, sae, feature_idx, steering_val)
        specs_modificados.append(spec_mod)
        notas_procesadas.append(nota_id)
        print(f"    -> Procesado exitosamente")

    if not specs_modificados:
        raise ValueError("No se pudo procesar ninguna cuerda. "
                         "Verifica que existan los espectrogramas .npy")

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
        spec_path_out = str(output_path).replace('.wav', '.npy')
        np.save(spec_path_out, spec_combinado)
        print(f"Espectrograma guardado en: {spec_path_out}")

    return audio_final, spec_combinado


# =============================================================================
# FUNCIONES DE DESCARGA DE ETIQUETAS (descriptores)
# =============================================================================

def cargar_etiquetas():
    """Carga las etiquetas de descriptores si existen."""
    labels_path = DATA_DIR / "labels.npy"
    if labels_path.exists():
        return np.load(labels_path)
    return None


# =============================================================================
# FUNCIÓN PRINCIPAL PARA USO DIRECTO
# =============================================================================

def procesar_archivo_audio(archivo_audio, feature_idx, steering_val,
                           output_path=None):
    """
    Procesa un archivo de audio de acorde completo:
    1. Preprocesa el audio a espectrograma
    2. Codifica a latente Z con VAE
    3. Codifica a características H con SAE
    4. Aplica steering a la característica especificada
    5. Decodifica a latente modificado
    6. Decodifica a espectrograma modificado
    7. Guarda el resultado

    Nota: Esta función procesa un solo archivo como una sola "cuerda".
    Para procesar múltiples notas (acorde real), usa procesar_acorde.
    """
    print(f"\n{'='*60}")
    print("PROCESANDO AUDIO INDIVIDUAL")
    print(f"Archivo: {archivo_audio}")
    print(f"Característica: {feature_idx}")
    print(f"Steering: {steering_val}")
    print("=" * 60)

    # Cargar modelos
    print("\nCargando modelos VAE y SAE...")
    vae, sae = cargar_modelos()
    print("Modelos cargados correctamente.\n")

    # Cargar audio y convertir a espectrograma
    y, sr = sf.read(archivo_audio, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Ajustar longitud
    if len(y) > TARGET_LENGTH:
        y = y[:TARGET_LENGTH]
    elif len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)), mode='constant')

    # Convertir a espectrograma
    spec = audio_to_spec(y)
    print(f"Espectrograma: {spec.shape}")

    # Procesar como una sola cuerda
    spec_mod = procesar_cuadra(spec, vae, sae, feature_idx, steering_val)
    print(f"  -> Procesado exitosamente")

    # Guardar
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_final = spec_to_audio(spec_mod)
        sf.write(output_path, audio_final, SR)
        print(f"\nAudio guardado en: {output_path}")

        spec_path_out = str(output_path).replace('.wav', '.npy')
        np.save(spec_path_out, spec_mod)
        print(f"Espectrograma guardado en: {spec_path_out}")

    return audio_final, spec_mod


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesa un acorde aplicando steering a cada cuerda"
    )
    parser.add_argument("--notas", type=str, required=True,
                        help="Lista de IDs de notas separados por comas")
    parser.add_argument("--feature", type=int, default=94,
                        help="Índice de característica a manipular")
    parser.add_argument("--steering", type=float, default=2.0,
                        help="Valor de steering (magnitud de modificación)")
    parser.add_argument("--output", type=str,
                        default="results/demo/acorde_steered.wav",
                        help="Archivo de salida")

    args = parser.parse_args()

    notas = [n.strip() for n in args.notas.split(",")]
    procesar_acorde(notas, args.feature, args.steering, args.output)

#!/usr/bin/env python3
"""
===============================================================================
PIPELINE DE DECOMPOSICIÓN Y STEERING DE ACORDES DE GUITARRA
===============================================================================

Este script toma un archivo de audio de acorde de guitarra, lo descompone en
sus notas individuales, procesa cada nota por separado con VAE+SAE, aplica
steering a una característica específica, y reconstruye el acorde modificado.

Flujo completo:
  Audio de acorde → Separación de notas (Basic Pitch) → Espectrogramas por nota
    → VAE (encoder) → Latentes Z
    → SAE (encoder) → Características H
    → Steering (modificar característica específica)
    → SAE (decoder) → Latentes Z modificado
    → VAE (decoder) → Espectrogramas modificados
    → Mezcla → Espectrograma del acorde modificado
    → Griffin-Lim → Audio de salida

Uso:
    python scripts/pipelineDescomposicion.py \
        --input audio/acorde.wav \
        --feature 94 \
        --steering 2.0 \
        --output resultados/acorde_modificado.wav

O si no tienes archivo de audio y quieres usar notas predefinidas:
    python scripts/pipelineDescomposicion.py \
        --notas "1-0.wav,2-5_1.wav,3-0.wav,4-2.wav,5-3.wav,6-3.wav" \
        --feature 94 --steering 2.0 --output resultado.wav

Requisitos:
    - Dataset SemanticTimbreDataset en data/raw/
    - Modelos entrenados en models/ (vae_best.pt, sae_best.pt)
    - Librerías: torch, librosa, soundfile, numpy, basic-pitch
===============================================================================
"""

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import soundfile as sf
import librosa

# =============================================================================
# CONFIGURACIÓN GLOBAL
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
RAW_DIR = DATA_DIR / "raw"


# =============================================================================
# MODELOS: VAE y SAE (misma arquitectura que en entrenamiento)
# =============================================================================

class VAEConv(nn.Module):
    """VAE convolucional para espectrogramas de guitarra."""
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
    """Autoencoder disperso con TopK para características tímbricas."""
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
# FUNCIONES DE PREPROCESAMIENTO Y POSTPROCESAMIENTO
# =============================================================================

def audio_a_espectrograma(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                          n_mels=N_MELS, n_frames=N_FRAMES):
    """
    Convierte señal de audio a espectrograma Mel log-magnitud normalizado [0,1].
    """
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


def espectrograma_a_audio(spec_log, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
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
# FUNCIONES DE PROCESAMIENTO POR CUERDA
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
    spec_mod = vae.decode(z_tensor).detach().cpu().numpy()[0, 0]

    return spec_mod


# =============================================================================
# FUNCIONES DE SEPARACIÓN DE NOTAS
# =============================================================================

def detectar_notas_con_basic_pitch(audio_path):
    """
    Usa Spotify Basic Pitch para detectar notas en un archivo de audio.
    Retorna lista de notas detectadas como nombres de archivos .wav
    que existen en el dataset.
    """
    try:
        from basic_pitch.inference import Model
        from basic_pitch.inference import predict
        from basic_ptich import ICASSP_2022_MODEL_PATH
    except ImportError:
        print("ADVERTENCIA: basic-pitch no está instalado. "
              "Instala con: pip install basic-pitch")
        return None

    print("Usando Basic Pitch para detectar notas en el acorde...")

    try:
        # Cargar modelo de Basic Pitch
        model_path = ICASSP_2022_MODEL_PATH
        model = Model(model_path)

        # Predecir notas
        model_output, midi_data, note_events = predict(
            audio_path,
            model,
            onset_threshold=0.5,
            frame_threshold=0.3,
            min_note_length=56,
            melodia_tricks=False
        )

        # Extraer notas detectadas (MIDI numbers)
        notas_detectadas = []
        for note in midi_data.notes:
            nota_midi = note.pitch
            notas_detectadas.append(nota_midi)

        print(f"Notas detectadas (MIDI): {notas_detectadas}")

        # Mapear notas MIDI a archivos .wav del dataset
        # Esto es una aproximación: buscamos archivos cuyo nombre contenga
        # el número de traste correspondiente
        notas_wav = []
        for nota_midi in notas_detectadas:
            # Convertir MIDI a número de traste aproximado (simplificado)
            # MIDI 40 = C2, 52 = E3, etc.
            # Aquí hacemos un mapeo aproximado basado en los nombres de archivo
            wav_encontrado = buscar_wav_por_nota_midi(nota_midi)
            if wav_encontrado:
                notas_wav.append(wav_encontrado)

        print(f"Notas mapeadas a archivos: {notas_wav}")
        return notas_wav

    except Exception as e:
        print(f"Error con Basic Pitch: {e}")
        return None


def buscar_wav_por_nota_midi(nota_midi):
    """
    Busca un archivo .wav que corresponda a la nota MIDI dada.
    Este es un mapeo aproximado basado en la nomenclatura del dataset.
    """
    # Los archivos del dataset tienen nombres como "X-Y_Y.wav" donde
    # X parece ser el traste y Y la cuerda
    # Hacemos una búsqueda heurística en el dataset
    import glob
    import re

    # Intentar encontrar archivos que puedan corresponder
    raw_dir = DATA_DIR / "raw" / "audio"
    if not raw_dir.exists():
        return None

    candidatos = []
    for wav_path in raw_dir.rglob("*.wav"):
        nombre = wav_path.stem
        # Intentar extraer número del nombre
        match = re.search(r'(\d+)', nombre)
        if match:
            num = int(match.group(1))
            # Aproximación: MIDI 40 = traste 0, cada traste es un semitono
            traste_estimado = nota_midi - 40
            if num == traste_estimado:
                candidatos.append(wav_path.name)

    if candidatos:
        return candidatos[0]  # Devolver el primero encontrado
    return None


def encontrar_espectrograma_para_nota(nota_id):
    """
    Busca el espectrograma .npy correspondiente a una nota.
    Intenta varios formatos de nombre.
    """
    # Intentar directamente
    ruta = SPECS_DIR / f"{nota_id}.npy"
    if ruta.exists():
        return ruta

    # Intentar buscar en todo el directorio specs
    for f in SPECS_DIR.glob("*.npy"):
        if f.stem == nota_id:
            return f

    return None


# =============================================================================
# FUNCIÓN PRINCIPAL DEL PIPELINE
# =============================================================================

def procesar_acorde_geometrico(notas_ids, feature_idx, steering_val,
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

        spec = np.load(spec_path).astype(np.float32)
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


def procesar_archivo_audio(archivo_audio, feature_idx, steering_val,
                           output_path=None):
    """
    Procesa un archivo de audio de acorde completo:
    1. Detecta notas usando Basic Pitch
    2. Busca espectrogramas correspondientes
    3. Procesa cada cuerda con VAE+SAE + steering
    4. Reconstruye el acorde modificado
    """
    print("=" * 60)
    print("PIPELINE DE DESCOMPOSICIÓN DE ACORDE")
    print("=" * 60)
    print(f"Archivo de entrada: {audio_path}")
    print(f"Característica: {feature_idx}")
    print(f"Steering: {steering_val}")
    print("=" * 60)

    # Detectar notas
    notas_detectadas = detectar_notas_con_basic_pitch(audio_path)

    if not notas_detectadas or len(notas_detectadas) == 0:
        print("ADVERTENCIA: No se detectaron notas. "
              "Usando notas por defecto.")
        notas_ids = ["1-0.wav", "2-5_1.wav", "3-0.wav",
                     "4-2.wav", "5-3.wav", "6-3.wav"]
    else:
        notas_ids = notas_detectadas

    print(f"\nNotas a procesar: {notas_ids}")

    # Procesar acorde
    audio_final, spec_combinado = procesar_acorde(
        notas_ids, feature_idx, steering_val, output_path
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)

    return audio_final, spec_combinado


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de descomposición y steering de acordes de guitarra"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Archivo de audio de acorde de entrada (.wav)"
    )
    parser.add_argument(
        "--notas", type=str, default=None,
        help="Lista de IDs de notas separados por comas "
             "(ej: '1-0.wav,2-5_1.wav,3-0.wav')"
    )
    parser.add_argument(
        "--feature", type=int, default=94,
        help="Índice de característica SAE a manipular (default: 94)"
    )
    parser.add_argument(
        "--steering", type=float, default=2.0,
        help="Valor de steering (magnitud de modificación, default: 2.0)"
    )
    parser.add_argument(
        "--output", type=str, default="results/demo/acorde_steered.wav",
        help="Archivo de audio de salida"
    )

    args = parser.parse_args()

    # Verificar que existen los modelos
    if not (MODELS_DIR / "vae_best.pt").exists():
        print("ERROR: No se encuentra vae_best.pt en models/")
        return
    if not (MODELS_DIR / "sae_best.pt").exists():
        print("ERROR: No se encuentra sae_best.pt en models/")
        return

    # Si se especifica archivo de audio, usarlo para detectar notas
    if args.input:
        audio_path = Path(args.input)
        if not audio_path.exists():
            print(f"ERROR: No se encuentra el archivo: {args.input}")
            return
        audio_final, spec_combinado = procesar_acorde_geometrico(
            [], args.feature, args.steering, args.output
        )
        # Nota: si args.input se especifica, detectar_notas_con_basic_pitch
        # se llama dentro de procesar_acorde_archivo, pero aquí estamos
        # usando la versión simplificada con notas explícitas.
        # Para usar detección automática, usa la función procesar_archivo_audio
        print("\nNota: Para usar detección automática de notas desde audio,")
        print("usa la función procesar_archivo_audio directamente.")
    elif args.notas:
        # Usar notas proporcionadas directamente
        notas = [n.strip() for n in args.notas.split(",")]
        print(f"\n{'='*60}")
        print("DEMO: Steering en acorde de guitarra (multi-cuerda)")
        print("=" * 60)
        print(f"Notas: {notas}")
        print(f"Característica: {args.feature}")
        print(f"Steering: {args.steering}")
        print(f"Salida: {args.output}")
        print("=" * 60)

        audio, spec = procesar_acorde_geometrico(
            notas, args.feature, args.steering, args.output
        )

        print("\n" + "=" * 60)
        print("Demo completada exitosamente")
        print("=" * 60)
    else:
        print("ERROR: Debes especificar --input o --notas")
        parser.print_help()


if __name__ == "__main__":
    main()

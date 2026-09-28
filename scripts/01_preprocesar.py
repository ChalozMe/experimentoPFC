#!/usr/bin/env python3
"""
Fase 1: Preprocesamiento de audio a espectrogramas Mel log-magnitud.
Extrae etiquetas del path: audio/Categoria/Subcategoria/Nivel/archivo.wav
"""

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import json
import soundfile as sf

# Configuración
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
DURATION = 3.0  # segundos
TARGET_LENGTH = int(SR * DURATION)  # 66150 muestras
N_FRAMES = 128  # frames temporales objetivo

DATA_DIR = Path("/home/chalo/pfc/data/raw")
OUTPUT_DIR = Path("/home/chalo/pfc/data/specs")
LABELS_FILE = Path("/home/chalo/pfc/data/labels.npy")
METADATA_FILE = Path("/home/chalo/pfc/data/metadata.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_audio_files(data_dir: Path) -> list[Path]:
    """Encuentra todos los archivos .wav recursivamente."""
    return sorted(data_dir.rglob("*.wav"))


def load_and_fix_audio(path: Path, target_length: int) -> np.ndarray:
    """Carga un archivo de audio y ajusta la longitud (crop o pad)."""
    y, sr = sf.read(path, dtype='float32')
    
    # Convertir a mono si es estéreo
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    # Resamplear si es necesario
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    
    # Ajustar longitud
    if len(y) > target_length:
        # Crop aleatorio
        start = np.random.randint(0, len(y) - target_length)
        y = y[start:start + target_length]
    elif len(y) < target_length:
        # Pad con ceros
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    
    return y


def audio_to_melspectrogram(y: np.ndarray) -> np.ndarray:
    """Convierte audio a espectrograma Mel log-magnitud normalizado [0,1]."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    S_log = librosa.power_to_db(S, ref=np.max)
    
    # Normalizar a [0, 1]
    S_norm = (S_log - S_log.min()) / (S_log.max() - S_log.min() + 1e-8)
    
    # Ajustar número de frames
    if S_norm.shape[1] > N_FRAMES:
        S_norm = S_norm[:, :N_FRAMES]
    elif S_norm.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    
    return S_norm.astype(np.float32)


def extract_label_from_path(path: Path, data_dir: Path) -> dict:
    """Extrae etiqueta del path: audio/Categoria/Subcategoria/Nivel/archivo.wav"""
    rel_path = path.relative_to(data_dir)
    parts = rel_path.parts  # ['audio', 'DistortionFX', 'Crush', '0', 'file.wav']
    
    if len(parts) >= 4:
        categoria = parts[1]      # DistortionFX
        subcategoria = parts[2]   # Crush / Crunch
        nivel = parts[3]          # 0, 5, 10, ..., 100
        
        try:
            nivel_int = int(nivel)
        except ValueError:
            nivel_int = 0
    else:
        categoria = "unknown"
        subcategoria = "unknown"
        nivel_int = 0
    
    return {
        "path": str(rel_path),
        "categoria": categoria,
        "subcategoria": subcategoria,
        "nivel": nivel_int
    }


def main():
    print("=" * 60)
    print("FASE 1: Preprocesamiento de Audio")
    print("=" * 60)
    
    # Encontrar archivos de audio
    audio_files = find_audio_files(DATA_DIR)
    print(f"Archivos .wav encontrados: {len(audio_files)}")
    
    if len(audio_files) == 0:
        print("ERROR: No se encontraron archivos .wav")
        return
    
    # Procesar cada archivo
    metadata = []
    failed_files = []
    
    for i, audio_path in enumerate(tqdm(audio_files, desc="Procesando audio")):
        try:
            # Cargar y procesar audio
            y = load_and_fix_audio(audio_path, TARGET_LENGTH)
            spec = audio_to_melspectrogram(y)
            
            # Guardar espectrograma
            spec_path = OUTPUT_DIR / f"{i:06d}.npy"
            np.save(spec_path, spec)
            
            # Extraer etiqueta del path
            meta = extract_label_from_path(audio_path, DATA_DIR)
            meta["spec_index"] = i
            meta["spec_path"] = str(spec_path.relative_to(OUTPUT_DIR.parent))
            metadata.append(meta)
            
        except Exception as e:
            failed_files.append((str(audio_path), str(e)))
            continue
    
    # Guardar metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Crear array de etiquetas (nivel del descriptor)
    etiquetas = np.array([m["nivel"] for m in metadata], dtype=np.float32).reshape(-1, 1)
    np.save(LABELS_FILE, etiquetas)
    
    print(f"\nProcesados: {len(metadata)} / {len(audio_files)}")
    print(f"Fallidos: {len(failed_files)}")
    
    if failed_files:
        print("\nPrimeros 5 archivos fallidos:")
        for path, error in failed_files[:5]:
            print(f"  {path}: {error}")
    
    print(f"\nEspectrogramas guardados en: {OUTPUT_DIR}")
    print(f"Metadata guardada en: {METADATA_FILE}")
    print(f"Etiquetas guardadas en: {LABELS_FILE} (shape: {etiquetas.shape})")
    
    # Verificar shape
    if len(metadata) > 0:
        sample_spec = np.load(OUTPUT_DIR / f"{metadata[0]['spec_index']:06d}.npy")
        print(f"Shape del espectrograma: {sample_spec.shape}")
        print(f"Rango de valores: [{sample_spec.min():.3f}, {sample_spec.max():.3f}]")
        
        # Mostrar distribución de etiquetas
        print(f"\nDistribución de etiquetas (nivel):")
        print(f"  Min: {etiquetas.min()}, Max: {etiquetas.max()}, Mean: {etiquetas.mean():.1f}")
        print(f"  Subcategorías únicas: {set(m['subcategoria'] for m in metadata)}")


if __name__ == "__main__":
    main()

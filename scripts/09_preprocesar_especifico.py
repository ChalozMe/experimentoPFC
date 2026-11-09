#!/usr/bin/env python3
"""
Preprocesa archivos .wav específicos a espectrogramas .npy para el experimento de acorde.
"""

import numpy as np
import librosa
from pathlib import Path
import soundfile as sf

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_FRAMES = 128

DATA_DIR = Path("/home/chalo/pfc/data")
SPECS_DIR = DATA_DIR / "specs"
RAW_DIR = DATA_DIR / "raw"

def audio_to_spec(audio_path, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, n_frames=N_FRAMES):
    """Convierte un archivo .wav a espectrograma Mel log-magnitud normalizado [0,1]."""
    y, sr_orig = sf.read(audio_path, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr_orig != sr:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
    
    # Ajustar longitud
    target_length = int(sr * 3.0)  # 3 segundos
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    
    # Espectrograma Mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_log = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_log - S_log.min()) / (S_log.max() - S_log.min() + 1e-8)
    
    # Ajustar frames
    if S_norm.shape[1] > n_frames:
        S_norm = S_norm[:, :n_frames]
    elif S_norm.shape[1] < n_frames:
        pad_width = n_frames - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode='constant')
    
    return S_norm.astype(np.float32)

def procesar_nota(nota_id, specs_dir=SPECS_DIR, raw_dir=DATA_DIR / "raw"):
    """Procesa un archivo .wav a espectrograma .npy."""
    wav_path = raw_dir / "audio" / "DistortionFX" / "Dirt" / "65" / nota_id
    if not wav_path.exists():
        # Intentar buscar en otras carpetas
        for subdir in RAW_DIR.rglob("*.wav"):
            if subdir.name == nota_id:
                wav_path = subdir
                break
        else:
            print(f"  No encontrado: {nota_id}")
            return None
    
    spec = audio_to_spec(wav_path)
    out_path = specs_dir / f"{nota_id}.npy"
    np.save(out_path, spec)
    print(f"  Procesado: {nota_id} -> {spec.shape}")
    return spec

if __name__ == "__main__":
    import sys
    notas = [
        "1-0.wav", "2-5_1.wav", "3-0.wav", "4-2.wav", "5-3.wav", "6-3.wav"
    ]
    
    print(f"Procesando {len(notas)} archivos .wav a espectrogramas...")
    for nota in notas:
        procesar_nota(nota)
    
    print(f"\nCompletado. Espectrogramas guardados en {SPECS_DIR}")
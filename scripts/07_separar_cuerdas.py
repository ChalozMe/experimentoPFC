#!/usr/bin/env python3
"""
Script para preparar espectrogramas de notas individuales (simula separación de cuerdas).
Toma archivos .wav individuales y sus espectrogramas ya generados para usarlos
como las "cuerdas" de un acorde.

Uso:
    python scripts/07_separar_cuerdas.py --notas "5-5_3,6-5_3,4-5_3,2-5_3,0-5_3,5-4_3"
"""

import argparse
import numpy as np
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path("/home/chalo/pfc/data")
SPECS_DIR = DATA_DIR / "specs"

def separar_cuerdas(notas_ids, output_dir=None):
    """
    Dado una lista de IDs de notas, carga sus espectrogramas y los prepara
    como "cuerdas" individuales para el procesamiento posterior.
    
    Args:
        notas_ids: Lista de IDs de archivos .wav (sin extensión)
        output_dir: Directorio para guardar los espectrogramas individuales
    
    Returns:
        lista de tuplas (nombre_nota, espectrograma_array)
    """
    especs = []
    for nota_id in notas_ids:
        spec_path = SPECS_DIR / f"{nota_id}.npy"
        if not spec_path.exists():
            print(f"ADVERTENCIA: No se encontró {spec_path}")
            continue
        spec = np.load(spec_path)
        especs.append((nota_id, spec))
        print(f"  Cargado: {nota_id} -> {spec.shape}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for nombre, spec in especs:
            np.save(output_dir / f"{nombre}.npy", spec)
    
    return especs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara notas individuales como 'cuerdas' para procesamiento de acorde")
    parser.add_argument("--notas", type=str, required=True, 
                        help="Lista de IDs de notas separados por comas (ej: '5-5_3,6-5_3,4-5_3,2-5_3,0-5_3,5-4_3')")
    parser.add_argument("--output", type=str, default="data/saidas/cuerdas",
                        help="Directorio para guardar espectrogramas individuales")
    
    args = parser.parse_args()
    
    notas = [n.strip() for n in args.notas.split(",")]
    print(f"Procesando {len(notas)} notas: {notas}")
    
    especs = separar_cuerdas(notas, args.output)
    print(f"\nTotal cargadas: {len(especs)} notas")

#!/usr/bin/env python3
"""
Fase 1b: Extraer etiquetas del dataset.
El dataset tiene 19 anotados. Este script lee el CSV/JSON de metadatos
y crea un array labels.npy con shape (N, 19).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

DATA_DIR = Path("/home/chalo/pfc/data/raw")
OUTPUT_FILE = Path("/home/chalo/pfc/data/labels.npy")
METADATA_OUTPUT = Path("/home/chalo/pfc/data/metadata.json")


def main():
    print("Buscando archivo de metadatos del dataset...")
    
    # Buscar CSV o JSON con las anotaciones
    csv_files = list(DATA_DIR.rglob("*.csv"))
    json_files = list(DATA_DIR.rglob("*.json"))
    
    print(f"Archivos CSV encontrados: {len(csv_files)}")
    print(f"Archivos JSON encontrados: {len(json_files)}")
    
    if csv_files:
        # Usar el primer CSV encontrado
        csv_path = csv_files[0]
        print(f"\nUsando: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Shape: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        print(f"\nPrimeras 5 filas:")
        print(df.head())
        
        # Intentar identificar columnas de descriptores
        # El dataset tiene 19 descriptores con valores 0-100
        descriptor_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if df[col].min() >= 0 and df[col].max() <= 100:
                    if col not in ['id', 'index']:
                        descriptor_cols.append(col)
        
        print(f"\nPosibles columnas de descriptores ({len(descriptor_cols)}):")
        for col in descriptor_cols:
            print(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.1f}")
        
        # Guardar etiquetas
        if len(descriptor_cols) >= 19:
            labels = df[descriptor_cols[:19]].values
            np.save(OUTPUT_FILE, labels)
            print(f"\nEtiquetas guardadas: {labels.shape}")
            
            # Guardar metadata
            metadata = {
                "n_muestras": len(df),
                "descriptores": descriptor_cols[:19],
                "archivo_fuente": str(csv_path)
            }
            with open(METADATA_OUTPUT, 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            print(f"\nERROR: Solo se encontraron {len(descriptor_cols)} columnas de descriptores (se esperaban 19)")
    
    elif json_files:
        json_path = json_files[0]
        print(f"\nUsando: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        print(f"Tipo: {type(data)}")
        if isinstance(data, list):
            print(f"Elementos: {len(data)}")
            print(f"Primer elemento: {data[0] if data else 'vacio'}")
    
    else:
        print("\nNo se encontraron archivos de metadatos.")
        print("El dataset puede tener las anotaciones en los nombres de archivo o carpetas.")
        print("\nEstructura de carpetas encontrada:")
        for d in sorted(DATA_DIR.iterdir())[:20]:
            if d.is_dir():
                n_files = len(list(d.rglob('*.wav')))
                print(f"  {d.name}/ ({n_files} archivos .wav)")


if __name__ == "__main__":
    main()

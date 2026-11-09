#!/usr/bin/env python3
"""
Demo de steering en acordes de guitarra.
Toma un acorde (múltiples notas), procesa cada cuerda por separado con VAE+SAE,
aplica steering y reconstruye el acorde modificado.

Uso:
    python scripts/demo_acorde.py --notas "5-5_3,6-5_3,4-5_3,2-5_3,0-5_3,5-4_3" --feature 94 --steering 2.0
"""

import argparse
import sys
import os
sys.path.insert(0, '/home/chalo/pfc')
from scripts.procesar_acorde import procesar_acorde

def main():
    parser = argparse.ArgumentParser(description="Demo de steering en acordes de guitarra")
    parser.add_argument("--notas", type=str, default="5-5_3,6-5_3,4-5_3,2-5_3,0-5_3,5-4_3",
                        help="Lista de IDs de notas separados por comas")
    parser.add_argument("--feature", type=int, default=94,
                        help="Índice de característica a manipular")
    parser.add_argument("--steering", type=float, default=2.0,
                        help="Valor de steering (magnitud de modificación)")
    parser.add_argument("--output", type=str, default="results/demo/acorde_steered.wav",
                        help="Archivo de salida")

    args = parser.parse_args()

    notas = [n.strip() for n in args.notas.split(",")]

    print("=" * 60)
    print("DEMO: Steering en acorde de guitarra (multi-cuerda)")
    print("=" * 60)
    print(f"Notas: {notas}")
    print(f"Característica: {args.feature}")
    print(f"Steering: {args.steering}")
    print(f"Salida: {args.output}")
    print("=" * 60)

    audio, spec = procesar_acorde(notas, args.feature, args.steering, args.output)

    print("\n" + "=" * 60)
    print("Demo completada exitosamente")
    print("=" * 60)

if __name__ == "__main__":
    main()

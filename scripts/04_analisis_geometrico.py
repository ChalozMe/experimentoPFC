#!/usr/bin/env python3
"""
Fase 4: Análisis geométrico de las características del SAE.
- Sondas lineales (mapeo descriptor → característica)
- Cristales (relaciones análogas)
- Lóbulos (modularidad)
- Galaxia (ley de potencias)
"""

import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import pandas as pd

# Configuración
DATA_DIR = Path("/home/chalo/pfc/data")
ARRAYS_DIR = Path("/home/chalo/pfc/arrays")
RESULTS_DIR = Path("/home/chalo/pfc/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sondas_lineales(H, etiquetas):
    """Fase 4a: Sondas lineales para mapear descriptor a características."""
    print("\n" + "=" * 60)
    print("Fase 4a: Sondas Lineales")
    print("=" * 60)
    
    # Binarizar etiqueta (nivel > mediana)
    y = etiquetas.flatten()
    y_bin = (y > np.median(y)).astype(int)
    
    print(f"Etiqueta binaria: {y_bin.sum()} positivos / {len(y_bin) - y_bin.sum()} negativos")
    
    # Probar cada característica individualmente
    scores = []
    for j in range(H.shape[1]):
        clf = LogisticRegression(max_iter=1000)
        score = cross_val_score(clf, H[:, j:j+1], y_bin, cv=5, scoring='f1').mean()
        scores.append(score)
    
    scores = np.array(scores)
    mejores = np.where(scores > 0.5)[0]
    
    resultados = {
        "mejor_f1": float(scores.max()),
        "n_caracteristicas_f1>0.5": int(len(mejores)),
        "caracteristicas_top10": np.argsort(scores)[-10:].tolist(),
        "scores_top10": np.sort(scores)[-10:].tolist()
    }
    
    print(f"Mejor F1: {scores.max():.3f}")
    print(f"Características con F1 > 0.5: {len(mejores)}")
    print(f"Top 5 características: {np.argsort(scores)[-5:]}")
    
    return resultados


def analizar_cristales(H, etiquetas):
    """Fase 4b: Análisis de cristales (relaciones análogas)."""
    print("\n" + "=" * 60)
    print("Fase 4b: Análisis de Cristales")
    print("=" * 60)
    
    y = etiquetas.flatten()
    
    # Dividir en cuartiles
    q25, q75 = np.percentile(y, [25, 75])
    
    # Vector de diferencia: alto - bajo
    alto = H[y > q75].mean(axis=0)
    bajo = H[y < q25].mean(axis=0)
    delta = alto - bajo
    
    # Normalizar
    delta_norm = delta / (np.linalg.norm(delta) + 1e-8)
    
    # Comparar con vectores aleatorios (test de permutación)
    n_perms = 100
    random_deltas = []
    for _ in range(n_perms):
        y_perm = np.random.permutation(y)
        alto_p = H[y_perm > q75].mean(axis=0)
        bajo_p = H[y_perm < q25].mean(axis=0)
        delta_p = alto_p - bajo_p
        delta_p_norm = delta_p / (np.linalg.norm(delta_p) + 1e-8)
        random_deltas.append(delta_p_norm)
    
    random_deltas = np.array(random_deltas)
    
    # ¿El delta real es más grande que los aleatorios?
    delta_norm_real = np.linalg.norm(delta)
    random_norms = np.linalg.norm(random_deltas, axis=1)
    
    resultados = {
        "norma_delta_real": float(delta_norm_real),
        "norma_delta_random_mean": float(random_norms.mean()),
        "norma_delta_random_std": float(random_norms.std()),
        "z_score": float((delta_norm_real - random_norms.mean()) / (random_norms.std() + 1e-8))
    }
    
    print(f"Norma delta real: {delta_norm_real:.4f}")
    print(f"Norma delta random (mean ± std): {random_norms.mean():.4f} ± {random_norms.std():.4f}")
    print(f"Z-score: {resultados['z_score']:.2f}")
    
    if resultados['z_score'] > 2:
        print("→ ¡Estructura significativa! (z > 2)")
    else:
        print("→ Estructura no significativa (z < 2)")
    
    return resultados


def analizar_lobulos(H):
    """Fase 4c: Análisis de lóbulos (modularidad)."""
    print("\n" + "=" * 60)
    print("Fase 4c: Análisis de Lóbulos")
    print("=" * 60)
    
    # Matriz de correlación entre características
    print("Calculando matriz de correlación...")
    corr_matrix = np.corrcoef(H.T)
    
    # Reemplazar NaN con 0 (características con varianza cero)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Distancia = 1 - correlación
    dist_matrix = 1 - np.abs(corr_matrix)
    
    # Asegurar que no haya NaN ni infinitos
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Poner diagonal en cero (requerido por silhouette_score)
    np.fill_diagonal(dist_matrix, 0.0)
    
    # Clustering jerárquico
    print("Aplicando clustering jerárquico...")
    Z = linkage(pdist(dist_matrix), method='ward')
    
    # Probar diferentes números de clusters
    mejor_score = -1
    mejor_n = 2
    
    for n_clusters in [5, 10, 15]:
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        score = silhouette_score(dist_matrix, clusters, metric='precomputed')
        print(f"  {n_clusters} clusters: silhouette = {score:.3f}")
        
        if score > mejor_score:
            mejor_score = score
            mejor_n = n_clusters
    
    clusters_final = fcluster(Z, mejor_n, criterion='maxclust')
    
    resultados = {
        "mejor_n_clusters": int(mejor_n),
        "silhouette_score": float(mejor_score),
        "distribucion_clusters": {int(k): int(v) for k, v in zip(*np.unique(clusters_final, return_counts=True))}
    }
    
    print(f"\nMejor clustering: {mejor_n} clusters, silhouette = {mejor_score:.3f}")
    
    return resultados


def analizar_galaxia(H):
    """Fase 4d: Análisis de galaxia (ley de potencias)."""
    print("\n" + "=" * 60)
    print("Fase 4d: Análisis de Galaxia")
    print("=" * 60)
    
    # Matriz de covarianza
    print("Calculando matriz de covarianza...")
    cov = np.cov(H.T)
    
    # Eigenvalores
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    # Ajuste de ley de potencias en log-log
    log_rank = np.log(np.arange(1, len(eigenvalues) + 1))
    log_eigen = np.log(eigenvalues)
    
    # Regresión lineal
    coeffs = np.polyfit(log_rank, log_eigen, 1)
    alpha = -coeffs[0]
    
    # Dimensión efectiva (90% varianza)
    var_explicada = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    dim_efectiva = np.searchsorted(var_explicada, 0.90) + 1
    
    resultados = {
        "alpha": float(alpha),
        "dim_efectiva_90": int(dim_efectiva),
        "total_eigenvalores": int(len(eigenvalues)),
        "top_10_eigenvalores": eigenvalues[:10].tolist()
    }
    
    print(f"α (pendiente ley de potencias): {alpha:.3f}")
    print(f"Dimensión efectiva (90% varianza): {dim_efectiva}")
    print(f"Total eigenvalores positivos: {len(eigenvalues)}")
    
    return resultados


def main():
    print("=" * 60)
    print("FASE 4: Análisis Geométrico")
    print("=" * 60)
    
    # Cargar datos
    H = np.load(ARRAYS_DIR / "caracteristicas_sae.npy")
    print(f"Características SAE: {H.shape}")
    
    # Cargar etiquetas
    labels_path = DATA_DIR / "labels.npy"
    if labels_path.exists():
        etiquetas = np.load(labels_path)
        print(f"Etiquetas: {etiquetas.shape}")
    else:
        print("ERROR: No se encontraron etiquetas")
        return
    
    # Ejecutar análisis
    resultados_sondas = sondas_lineales(H, etiquetas)
    resultados_cristales = analizar_cristales(H, etiquetas)
    resultados_lobulos = analizar_lobulos(H)
    resultados_galaxia = analizar_galaxia(H)
    
    # Guardar resultados
    resultados = {
        "sondas_lineales": resultados_sondas,
        "cristales": resultados_cristales,
        "lobulos": resultados_lobulos,
        "galaxia": resultados_galaxia
    }
    
    with open(RESULTS_DIR / "metricas.json", 'w') as f:
        json.dump(resultados, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Análisis completado. Resultados guardados en results/metricas.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

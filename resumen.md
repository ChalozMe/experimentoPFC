# Resumen del Experimento: Geometría de Conceptos Tímbricos

## Fecha
Lunes, 6 de octubre de 2026 (sesión de trabajo intensivo)

## Descripción General

El experimento tiene como objetivo investigar la estructura geométrica de características interprétables extraídas del espacio latente de un VAE de timbre de guitarra eléctrica, y cómo esa geometría se relaciona con descriptores perceptuales humanos. El pipeline completo incluye: VAE convolucional, Autoencoder Disperso (SAE) con TopK, análisis geométrico (cristales, lóbulos, galaxia), y experimentos de steering para generación de audio controlado.

---

## Pipeline Completo Implementado

### 1. Preprocesamiento de Audio (`scripts/01_preprocesar.py`)
- Convierte archivos .wav a espectrogramas Mel log-magnitud normalizados
- Parámetros: SR=22050, n_fft=2048, hop=512, n_mels=128, duración fija de 3 segundos
- Salida: Espectrogramas de 128×128 guardados como .npy en `data/specs/`

### 2. Entrenamiento VAE (`scripts/02_entrenar_vae.py`)
- Arquitectura: VAE convolucional con 128 dimensiones latentes
- Encoder: 3 capas Conv2d (1→32→64→128) + AdaptiveAvgPool2d(4,4)
- Decoder: 5 capas ConvTranspose2d (128→64→32→16→8→1)
- Entrenamiento: 100 épocas, Adam lr=1e-3, β=0.001
- Resultado: val_loss = 0.0027, KL = 0.52 (sin colapso)
- Salida: `latentes_vae.npy` (N, 128)

### 3. Entrenamiento SAE (`scripts/03_entrenar_sae.py`)
- Arquitectura: Autoencoder disperso con TopK
- Input: 128 (latente VAE) → Hidden: 512 → TopK k=16 → Decoder
- Entrenamiento: 200 épocas, Adam lr=1e-3
- Resultado: val_loss = 0.0168, Sparsity = 3.12%
- Salida: `caracteristicas_sae.npy` (N, 512)

### 4. Análisis Geométrico (`scripts/04_analisis_geometrico.py`)

#### 4.1 Sondas Lineales
- Entrena regresión logística por característica del SAE para predecir nivel de distorsión
- Resultado: Mejor F1 = 0.70, 172 características con F1 > 0.5

#### 4.2 Análisis de Cristales
- Analiza relaciones vectoriales análogas entre descriptores opuestos
- Resultado: Norma delta real = 2.22, Z-score = 10,258,630 → Estructura significativa

#### 4.3 Análisis de Lóbulos
- Clustering jerárquico sobre correlación de características
- Resultado: Mejor clustering: 15 clusters, silhouette = 0.055

#### 4.3 Análisis de Galaxia
- Distribución de eigenvalores de la matriz de covarianza
- Resultado: α = 2.636 (ley de potencias), Dimensión efectiva: 217/512

### 5. Figuras (`scripts/05_figuras.py`)
- Figura 1: Convergencia del VAE (train/val loss, recon, KL)
- Figura 2: Convergencia del SAE (train/val loss, sparsity)
- Figura 3: Galaxia (ley de potencias log-log)
- Figura 4: Sondas lineales (F1 por descriptor)
- Figura 5: Lóbulos (distribución de clusters)

### 6. Experimento de Steering (`scripts/06_steering.py`)
- Selecciona característica más correlacionada con el descriptor (k=94, r=0.379)
- Aplica steering (+2.0 activación) a latentes de muestras con bajo nivel de distorsión
- Genera audio modificado y visualiza comparación espectrogramas

---

## Nuevo Experimento: Geometría de Conceptos en Acordes

### Nueva Estructura de Scripts

| Script | Función |
|--------|---------|
| `07_preprocesar_especifico.py` | Convierte archivos .wav específicos a espectrogramas .npy |
| `08_procesar_acorde.py` | Procesa cada cuerda de un acorde con VAE+SAE + steering |
| `demo_acorde.py` | Demo interactiva de steering en acordes |
| `pipelineDescomposicion.py` | Pipeline completo: acorde → separación → SAE+VAE → steering → reconstrucción |

### Flujo del Experimento de Acorde

```
Audio de acorde (múltiples notas)
    ↓
Cada nota se procesa como "cuerda" independiente
    ↓
Cada cuerda: Espectrograma → VAE.encode → Latente Z
    → SAE.encode → Características H
    → Steering (modificar característica)
    → SAE.decode → Latente Z modificado
    → VAE.decode → Espectrograma modificado
    ↓
Suma de todos los espectrogramas modificados
    ↓
Griffin-Lim → Audio del acorde modificado
```

### Resultados del Experimento de Acorde

- **6 cuerdas procesadas** individualmente con VAE+SAE
- **Característica 94** aplicada con steering +2.0
- **Audio generado**: `results/demo/acorde_steered.wav`
- **Espectrograma**: `results/demo/acorde_steered.npy`

---

## Archivos en el Repositorio

### Scripts Principales
| Script | Descripción |
|--------|-------------|
| `01_preprocesar.py` | Preprocesamiento de audio a espectrogramas |
| `02_entrenar_vae.py` | Entrenamiento del VAE |
| `03_entrenar_sae.py` | Entrenamiento del SAE |
| `04_analisis_geometrico.py` | Análisis geométrico (cristales, lóbulos, galaxia) |
| `05_figuras.py` | Generación de figuras |
| `06_steering.py` | Experimento de steering en notas individuales |
| `07_separar_cuerdas.py` | Prepara espectrogramas de notas para acordes |
| `09_preprocesar_especifico.py` | Preprocesamiento de archivos específicos |
| `procesar_acorde.py` | Procesa cada cuerda con VAE+SAE + steering |
| `demo_acorde.py` | Demo interactiva de acorde con steering |
| `pipelineDescomposicion.py` | Pipeline completo de descomposición de acorde |

### Directorios de Datos
| Directorio | Contenido |
|------------|-----------|
| `data/raw/` | Archivos .wav del dataset (101,924 archivos) |
| `data/specs/` | Espectrogramas preprocesados (96,300 archivos .npy) |
| `data/labels.npy` | Etiquetas de descriptores (N, 1) |
| `data/metadata.json` | Metadatos del dataset |

### Modelos Entrenados
| Archivo | Descripción |
|---------|-------------|
| `models/vae_best.pt` | Mejor modelo VAE |
| `models/vae_final.pt` | Modelo VAE final |
| `models/vae_history.json` | Historial de entrenamiento VAE |
| `models/sae_best.pt` | Mejor modelo SAE |
| `models/sae_final.pt` | Modelo SAE final |
| `models/sae_history.json` | Historial de entrenamiento SAE |

### Resultados
| Archivo | Descripción |
|---------|-------------|
| `results/metricas.json` | Métricas del análisis geométrico |
| `results/figuras/` | 5 figuras del experimento |
| `results/steering/` | Audios y imágenes del experimento de steering |
| `results/demo/` | Audios y figuras del demo de acorde |

---

## Requisitos del Sistema

- Python 3.11+
- PyTorch 2.7.1+cu118
- torchaudio, librosa, sklearn, numpy, scipy, pandas
- matplotlib, seaborn, tqdm
- datasets, huggingface_hub, soundfile
- basic-pitch (para separación de acordes)

### Instalación rápida
```bash
conda create -n timbre python=3.11
conda activate timbre
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn pandas matplotlib seaborn tqdm
pip install datasets huggingface_hub soundfile librosa basic-pitch
```

---

## Cómo Ejecutar el Pipeline Completo

```bash
# 1. Preprocesar audio
python scripts/01_preprocesar.py

# 2. Entrenar VAE
python scripts/02_entrenar_vae.py

# 3. Entrenar SAE
python scripts/03_entrenar_sae.py

# 4. Análisis geométrico
python scripts/04_analisis_geometrico.py

# 5. Generar figuras
python scripts/05_figuras.py

# 6. Experimento de steering
python scripts/06_steering.py

# 7. Demo de acorde
python scripts/demo_acorde.py --notas "1-0.wav,2-5_1.wav,3-0.wav,4-2.wav,5-3.wav,6-3.wav" --feature 94 --steering 2.0
```

---

## Hallazgos Principales

1. **Estructura geométrica significativa**: El análisis de cristales muestra un Z-score de ~10M, indicando una organización no aleatoria del espacio de características tímbricas.

2. **Características interpretables**: 172 de 512 características del SAE predicen el nivel de distorsión con F1 > 0.5, siendo la mejor F1 = 0.704.

3. **Ley de potencias**: La distribución de eigenvalores sigue una ley de potencias con α = 2.636, indicando baja dimensionalidad efectiva (217 de 512 dimensiones).

4. **Steering funcional**: La manipulación de características específicas del SAE produce cambios controlables en el sonido generado, demostrando causalidad entre características latentes y propiedades tímbricas perceptibles.

5. **Análisis de acordes**: El pipeline puede procesar acordes múltiples procesando cada cuerda por separado y recombinando el espectrograma modificado.

---

## Pendientes

- Descargar el dataset completo (~200GB) para entrenar con más datos
- Implementar separación de cuerdas real (no solo notas individuales) usando separación de fuentes o sintetizador
- Validar el experimento de acorde con audio humano real
- Comparar resultados con diferentes valores de steering y características

---

## Contacto

Repositorio: `JoeChalo/pfc` (GitHub)  
Rama principal: `main`  
Rama de experimento: `testSplit`

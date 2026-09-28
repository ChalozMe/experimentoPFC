experimento_timbre.md
markdown

# Experimento: Geometría de Conceptos Tímbricos

## Contexto

**Título tentativo:** Geometría de Conceptos Tímbricos: Análisis de la Estructura Latente de Autoencoders Dispersos para el Timbre de la Guitarra Eléctrica.

**Pregunta de investigación:** ¿Qué estructura geométrica emerge en las características interpretables extraídas del espacio latente de un VAE de timbre de guitarra eléctrica, y cómo se relaciona esa geometría con los 19 descriptores perceptuales humanos?

**Hipótesis:** Las características del SAE no se organizan aleatoriamente, sino que forman cristales (relaciones análogas), lóbulos (agrupaciones espaciales) y una estructura de galaxia (distribución no isotrópica de eigenvalores) que refleja la organización perceptual del timbre.

**Aporte novedoso:** Nadie ha aplicado el marco geométrico de Tegmark et al. (2025) al timbre musical. El dataset de guitarra eléctrica existe y está anotado, pero no ha sido analizado con SAEs ni con análisis geométrico.

---

## Pipeline del Experimento

INPUT
│
│ Semantic Timbre Dataset
│ • 275,310 notas de guitarra eléctrica
│ • 19 descriptores perceptuales (0-100)
│
▼
ETAPA 1 — Representación acústica
│ INPUT: Señales de audio (.wav)
│ OUTPUT: Espectrogramas log-magnitud → X ∈ R^(N×F×T)
│
▼
ETAPA 2 — VAE de timbre
│ INPUT: X ∈ R^(N×F×T)
│ OUTPUT: Latentes → Z ∈ R^(N×128)
│
▼
ETAPA 3 — SAE sobre latentes
│ INPUT: Z ∈ R^(N×128)
│ OUTPUT: Características dispersas → H ∈ R^(N×K), K = 512-1024
│
▼
ETAPA 4 — Análisis geométrico
│ INPUT: H ∈ R^(N×K)
│ OUTPUT: Cristales → {Δv_i}
│ Lóbulos → {C_1, ..., C_m}
│ Galaxia → α (pendiente ley de potencias)
│
▼
ETAPA 5 — Validación semántica
│ INPUT: H ∈ R^(N×K) + etiquetas de descriptores
│ OUTPUT: F1-score por característica
│ Correlación con valencia / arousal
│
▼
OUTPUT
│ • Diccionario de características tímbricas interpretables
│ • Estructura geométrica del espacio de timbre
│ • Mapa de correspondencia timbre → emoción
text


**Notación:**
- N = número de muestras
- F = número de bins de frecuencia
- T = número de frames temporales
- K = número de características del SAE (512-1024)

**Especificación de dimensiones:**

| Etapa | Módulo | Input | Dimensiones | Output | Dimensiones |
| :---: | :--- | :--- | :---: | :--- | :---: |
| 1 | Espectrograma | Señal de audio | N × T_audio | Espectrograma log | N × F × T |
| 2 | VAE | Espectrograma | N × F × T | Vector latente | N × 128 |
| 3 | SAE | Vector latente | N × 128 | Activaciones | N × K |
| 4 | Geometría | Activaciones | N × K | Métricas | {Δv}, {C}, α |
| 5 | Validación | Activaciones + etiquetas | N × (K+19) | F1, r | Escalares |

---

## FASE 0: Preparación del Entorno

### 0.1 — Dependencias

Instalar las siguientes librerías de Python:

- torch (con soporte CUDA si hay GPU)
- torchaudio
- numpy
- scipy
- scikit-learn
- pandas
- matplotlib
- seaborn
- umap-learn
- librosa
- soundfile
- tqdm
- tensorboard
- datasets (Hugging Face)
- huggingface_hub

**Alternativa sin instalación local:** Google Colab (GPU T4 gratuita) o Kaggle Notebooks (30h/semana gratuitas).

### 0.2 — Repositorios de Referencia

- `ksadov/FREUD` — Código base para entrenar SAEs en audio
- `ejmichaud/feature-geometry` — Código para análisis geométrico
- `AudioCommons/timbral_models` — Modelos perceptuales de timbre

### 0.3 — Dataset

**Semantic Timbre Dataset** (Hugging Face: `JoeCameron1/SemanticTimbreDataset`)

- 275,310 notas de guitarra eléctrica en formato `.wav`
- 19 descriptores perceptuales con magnitudes graduadas de 0 a 100
- Categorías de descriptores: DistortionFX, Modulation, EQ/Filter, Ambience, Dynamics

**Recomendación:** Usar un subconjunto curado de ~1,771 espectrogramas para el experimento inicial. Expandir si hay tiempo.

### 0.4 — Verificación de Hardware

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Sin GPU. Usar Colab o Kaggle.")

Plan B: Si no hay GPU local, usar Colab o Kaggle.
FASE 1: VAE de Timbre
1.1 — Preprocesamiento de Audio

Convertir cada señal de audio a un espectrograma Mel log-magnitud.

Parámetros recomendados:
Parámetro	Valor
Sample rate (sr)	22050 Hz
n_fft	2048
hop_length	512
n_mels	128

Salida: Array de shape (n_mels, T) por muestra.
python

import librosa
import numpy as np

def audio_a_espectrograma(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_log = librosa.power_to_db(S, ref=np.max)
    return S_log

1.2 — Arquitectura del VAE

Tipo: VAE convolucional ligero.

Especificación:

    Encoder: 3-4 capas convolucionales con stride 2, activación ReLU, AdaptiveAvgPool2d a (4,4)

    Capa latente: 128 dimensiones

    Decoder: simétrico al encoder, con ConvTranspose2d

    Reparametrización: estándar (mu + eps * std)

    β (peso KL): 0.001 - 0.01

Justificación: El paper del dataset usó exactamente 128 dimensiones latentes y obtuvo buenos resultados. No se necesita un VAE más grande.
python

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEConv(nn.Module):
    def __init__(self, n_mels=128, latent_dim=128):
        super().__init__()
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
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

1.3 — Función de Pérdida

Pérdida VAE = Reconstrucción (MSE) + β × KL divergence
python

def vae_loss(x_recon, x, mu, logvar, beta=0.001):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

1.4 — Entrenamiento

Hiperparámetros:
Parámetro	Valor
Optimizer	Adam
Learning rate	1e-3
Scheduler	ReduceLROnPlateau (patience=10)
Batch size	32-64
Épocas	50-100
Split	80% train / 10% val / 10% test
python

from torch.utils.data import DataLoader, TensorDataset

def entrenar_vae(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device).unsqueeze(1)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(x)
            loss, _, _ = vae_loss(x_recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device).unsqueeze(1)
                x_recon, mu, logvar, _ = model(x)
                loss, _, _ = vae_loss(x_recon, x, mu, logvar)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    return history

Criterio de éxito: El VAE debe reconstruir los espectrogramas con un error aceptable. Monitorear pérdida de reconstrucción y KL.
1.5 — Extracción de Latentes
python

def extraer_latentes(model, dataloader, device='cuda'):
    model.eval()
    latentes = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device).unsqueeze(1)
            mu, _ = model.encode(x)
            latentes.append(mu.cpu().numpy())
    return np.concatenate(latentes, axis=0)

Z = extraer_latentes(vae, train_loader)
print(f"Shape de latentes: {Z.shape}")
np.save('latentes_vae.npy', Z)

Salida: latentes_vae.npy con shape (N, 128).
FASE 2: SAE sobre Latentes
2.1 — Arquitectura del SAE

Tipo: Autoencoder disperso con TopK.

Especificación:

    Input dim: 128 (latente del VAE)

    Hidden dim (K): 512-1024

    Activación: ReLU o TopK (k = 16-32)

    Pérdida: Reconstrucción (MSE) + esparsidad (L1 o constraint TopK)

python

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

2.2 — Entrenamiento

Hiperparámetros:
Parámetro	Valor
Optimizer	Adam
Learning rate	1e-3
Batch size	512
Épocas	200
k (TopK)	16-32
python

def entrenar_sae(model, Z, epochs=200, lr=1e-3, batch_size=512, device='cuda'):
    Z_tensor = torch.tensor(Z, dtype=torch.float32)
    dataset = TensorDataset(Z_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    history = {'loss': [], 'sparsity': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_sparsity = 0
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_recon, h_sparse = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            sparsity = (h_sparse > 0).float().mean()
            recon_loss.backward()
            optimizer.step()
            epoch_loss += recon_loss.item()
            epoch_sparsity += sparsity.item()

        avg_loss = epoch_loss / len(loader)
        avg_sparsity = epoch_sparsity / len(loader)
        history['loss'].append(avg_loss)
        history['sparsity'].append(avg_sparsity)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Sparsity: {avg_sparsity:.4f}")

    return history

Criterio de éxito: El SAE debe reconstruir los latentes con error bajo, y las características deben ser esparsas (pocas activas por muestra).
2.3 — Extracción de Características
python

def extraer_caracteristicas(sae, Z, device='cuda'):
    sae.eval()
    Z_tensor = torch.tensor(Z, dtype=torch.float32).to(device)
    with torch.no_grad():
        H = sae.encode(Z_tensor)
    return H.cpu().numpy()

H = extraer_caracteristicas(sae, Z)
print(f"Shape de características: {H.shape}")
np.save('caracteristicas_sae.npy', H)

Salida: caracteristicas_sae.npy con shape (N, K).
FASE 3: Análisis Geométrico
3.1 — Análisis de Cristales (Relaciones Análogas)

Buscar relaciones vectoriales del tipo A : B :: C : D entre las características del SAE que corresponden a descriptores tímbricos.

Metodología:

    Identificar las características del SAE que mejor predicen cada descriptor (sonda lineal)

    Para pares de descriptores opuestos o relacionados, calcular el vector de diferencia

    Comparar si los vectores de diferencia son paralelos (coseno > 0.7) entre pares análogos

    Aplicar LDA para proyectar fuera direcciones distractoras globales

python

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def identificar_caracteristicas_por_descriptor(H, etiquetas, descriptores, umbral_f1=0.7):
    resultados = {}
    for i, desc in enumerate(descriptores):
        y = etiquetas[:, i]
        y_bin = (y > np.median(y)).astype(int)
        scores = []
        for j in range(H.shape[1]):
            clf = LogisticRegression(max_iter=1000)
            score = cross_val_score(clf, H[:, j:j+1], y_bin, cv=5, scoring='f1').mean()
            scores.append(score)
        scores = np.array(scores)
        indices = np.where(scores > umbral_f1)[0]
        resultados[desc] = indices
        print(f"{desc}: {len(indices)} características con F1 > {umbral_f1}")
    return resultados

Pregunta concreta: ¿Existe una relación análoga consistente entre brillante - oscuro y suave - áspero? ¿O entre cálido - metálico y redondo - cortante?
3.2 — Análisis de Lóbulos (Modularidad)

Medir si los descriptores relacionados se agrupan espacialmente en el espacio de características.

Metodología:

    Calcular la matriz de similitud coseno entre características

    Usar clustering jerárquico o DBSCAN para encontrar grupos

    Comparar cohesión espacial con la esperada al azar (test de permutación)

    Proyectar en 2D con UMAP y colorear por grupo de descriptor

python

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
import umap

def analizar_lobulos(H, n_clusters=10):
    sim_matrix = np.corrcoef(H.T)
    dist_matrix = 1 - sim_matrix
    Z = linkage(pdist(dist_matrix), method='ward')
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    sil_score = silhouette_score(dist_matrix, clusters, metric='precomputed')
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(H.T)
    return {
        'clusters': clusters,
        'silhouette': sil_score,
        'embedding': embedding,
        'linkage': Z
    }

Métrica clave: Silhouette score y entropía de clustering (pureza de cada cluster en términos de descriptores).
3.3 — Análisis de Galaxia (Ley de Potencias)

Analizar la distribución de eigenvalores de la matriz de covarianza de las características.

Metodología:

    Calcular la matriz de covarianza de las activaciones del SAE

    Calcular los eigenvalores

    Graficar en escala log-log

    Ajustar una ley de potencias (regresión lineal en log-log) y medir la pendiente α

    Comparar con datos aleatorios (test de permutación)

python

def analizar_galaxia(H):
    cov = np.cov(H.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]
    log_rank = np.log(np.arange(1, len(eigenvalues) + 1))
    log_eigen = np.log(eigenvalues)
    coeffs = np.polyfit(log_rank, log_eigen, 1)
    alpha = -coeffs[0]
    var_explicada = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    dim_efectiva = np.searchsorted(var_explicada, 0.90) + 1
    return {
        'eigenvalues': eigenvalues,
        'alpha': alpha,
        'dim_efectiva': dim_efectiva,
        'log_rank': log_rank,
        'log_eigen': log_eigen
    }

Pregunta concreta: ¿Sigue la distribución de eigenvalores una ley de potencias? ¿Cuál es la dimensionalidad efectiva del espacio de características tímbricas?
FASE 4: Validación Semántica
4.1 — Sondas Lineales
python

def validar_con_sondas(H, etiquetas, descriptores):
    resultados = {}
    for i, desc in enumerate(descriptores):
        y = etiquetas[:, i]
        y_bin = (y > np.median(y)).astype(int)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(clf, H, y_bin, cv=5, scoring='f1')
        resultados[desc] = {
            'f1_mean': scores.mean(),
            'f1_std': scores.std()
        }
    return resultados

Criterio: Una característica es interpretable si predice al menos un descriptor con F1 > 0.7.
4.2 — Correlación con Emoción
python

from scipy.stats import pearsonr

def correlacion_emocion(H, etiquetas, descriptores, mapa_emocional):
    valencia = np.zeros(len(etiquetas))
    arousal = np.zeros(len(etiquetas))
    for i, desc in enumerate(descriptores):
        v, a = mapa_emocional.get(desc, (0, 0))
        valencia += etiquetas[:, i] * v
        arousal += etiquetas[:, i] * a
    valencia = (valencia - valencia.mean()) / valencia.std()
    arousal = (arousal - arousal.mean()) / arousal.std()
    correlaciones = []
    for j in range(H.shape[1]):
        r_val, p_val = pearsonr(H[:, j], valencia)
        r_aro, p_aro = pearsonr(H[:, j], arousal)
        correlaciones.append({
            'feature': j,
            'r_valencia': r_val,
            'p_valencia': p_val,
            'r_arousal': r_aro,
            'p_arousal': p_aro
        })
    return pd.DataFrame(correlaciones)

Mapa emocional de ejemplo (basado en literatura, ajustar según criterio):
Descriptor	Valencia	Arousal
Bright	0.5	0.7
Dark	-0.3	-0.5
Warm	0.6	0.3
Harsh	-0.5	0.8
Soft	0.7	-0.3
Crunchy	-0.2	0.6
4.3 — Experimentos de Steering (Opcional)
python

def steering_experiment(vae, sae, H, Z, feature_idx, n_samples=10):
    idx_alto = np.argsort(H[:, feature_idx])[-n_samples:]
    Z_alto = Z[idx_alto]
    vae.eval()
    with torch.no_grad():
        Z_tensor = torch.tensor(Z_alto, dtype=torch.float32).to(device)
        x_gen = vae.decode(Z_tensor)
    return x_gen.cpu().numpy()

Si identificas una característica que predice bien un descriptor, actívala o desactívala y genera nuevas notas. Pide a voluntarios que califiquen las notas generadas. Si la manipulación cambia la percepción en la dirección esperada, has demostrado causalidad.
FASE 5: Documentación de Resultados
5.1 — Tablas
Tabla	Contenido
Tabla 1	Características interpretables por descriptor (F1 > 0.7)
Tabla 2	Métricas geométricas (α, dim_efectiva, silhouette)
Tabla 3	Correlaciones característica-emoción (r, p)
Tabla 4	Comparación con literatura (Paek et al., Tegmark et al.)
5.2 — Figuras
Figura	Contenido
Figura 1	Pipeline del experimento (input/output)
Figura 2	Espacio latente del VAE (PCA/t-SNE)
Figura 3	Matriz de similitud entre características (heatmap)
Figura 4	Proyección UMAP de características (lóbulos)
Figura 5	Gráfico log-log de eigenvalores (galaxia)
Figura 6	Correlación característica-emoción (scatter)
5.3 — Checklist de Reproducibilidad

    □

    Semilla aleatoria fijada (torch.manual_seed(42))
    □

    Hiperparámetros documentados en tabla
    □

    Versiones de librerías en requirements.txt
    □

    Dataset descargable con script
    □

    Código de entrenamiento reproducible
    □

    Métricas con intervalos de confianza
    □

    Análisis estadístico (test de permutación donde aplique)
    □

    Figuras con código generador

Referencias Clave
Paper	Relevancia
Cameron & Blackwell (2026) — Semantic Timbre Dataset	Dataset principal
Cameron & Blackwell (2026) — Evaluating Latent Space Structure in Timbre VAEs	Base metodológica para VAE
Paek et al. (2025) — Learning Interpretable Features in Audio Latent Spaces via SAEs	Base metodológica para SAE
Tegmark et al. (2025) — Geometry of Concepts	Marco para análisis geométrico
MIT Media Lab (2026) — Discovering Interpretable Concepts in Music Models	Precedente en música
Grekow (2025) — Regularized Latent Space for Emotion	Puente a emoción
Notas Finales para el Agente

    Priorizar reproducibilidad: Cada paso debe ser ejecutable desde cero.

    Documentar decisiones: Por qué 128 dimensiones, por qué TopK=16, etc.

    Manejar resultados negativos: Si los cristales no aparecen, reportarlo como hallazgo.

    Escalar gradualmente: Empezar con 1,771 muestras, expandir si hay tiempo.

    Validar con humanos: Si es posible, pedir a compañeros que califiquen notas generadas.

    Comparar con baselines: Usar PCA o t-SNE como línea base para el análisis geométrico.

text


---

Ese es el documento único. Todo el código está en bloques de Python dentro del `.md`, sin bash ni comandos de shell. Listo para dárselo a un agente.

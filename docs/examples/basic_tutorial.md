# Tutorial Básico

Este tutorial te guiará paso a paso para ejecutar tu primer experimento de neurodecodificación con IbioML.

## Preparación del Entorno

### 1. Instalación

```bash
pip install ibioml
```

### 2. Descargar Datos de Ejemplo

Para este tutorial, asumiremos que tienes un archivo `.mat` con datos neuronales. Si no tienes datos propios, puedes usar datos sintéticos:

```python
import numpy as np
import scipy.io as sio

# Generar datos sintéticos para el tutorial
np.random.seed(42)
n_samples = 10000
n_neurons = 50
n_trials = 100

# Simular actividad neuronal
neural_activity = np.random.poisson(0.1, (n_samples, n_neurons))

# Simular posición y velocidad
time = np.linspace(0, 100, n_samples)
position = 5 * np.sin(0.1 * time) + np.random.normal(0, 0.5, n_samples)
velocity = np.gradient(position) + np.random.normal(0, 0.2, n_samples)

# Simular contexto de recompensa
reward_context = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

# Crear marcadores de trials
trial_final_bins = np.sort(np.random.choice(
    range(n_samples-100), n_trials, replace=False
)) + 100

# Simular d-prime (medida de rendimiento)
d_prime = np.random.normal(2.8, 0.5, n_trials)

# Simular duración de trials
trial_durations = np.diff(np.concatenate([[0], trial_final_bins]))

# Guardar como archivo .mat
synthetic_data = {
    'neuronActivity': neural_activity,
    'position': position.reshape(-1, 1),
    'velocity': velocity.reshape(-1, 1),
    'rewCtxt': reward_context.reshape(-1, 1),
    'trialFinalBin': trial_final_bins,
    'dPrime': d_prime,
    'criterion': np.random.normal(0, 0.3, n_trials),
    'trialDurationInBins': trial_durations
}

# Crear directorio si no existe
import os
os.makedirs('tutorial_data', exist_ok=True)

# Guardar datos
sio.savemat('tutorial_data/synthetic_experiment.mat', synthetic_data)
print("✅ Datos sintéticos generados en 'tutorial_data/synthetic_experiment.mat'")
```

## Paso 1: Preprocesamiento de Datos

```python
from ibioml.preprocess_data import preprocess_data
import os

# Crear directorio para datos procesados
os.makedirs('tutorial_data/processed', exist_ok=True)

# Preprocesar los datos
preprocess_data(
    file_path='tutorial_data/synthetic_experiment.mat',
    file_name_to_save='tutorial_data/processed/experiment',
    bins_before=5,      # 5 bins hacia atrás
    bins_after=5,       # 5 bins hacia adelante
    bins_current=1,     # 1 bin actual
    threshDPrime=2.5,   # Filtrar trials con d' < 2.5
    firingMinimo=50     # Mínimo 50 spikes por neurona
)

print("✅ Preprocesamiento completado")
```

Esto generará 12 archivos `.pickle` con diferentes configuraciones:
- Con/sin contexto de recompensa
- Solo posición, solo velocidad, o ambos
- Datos aplanados (`_flat`) para MLP o datos temporales para RNNs

## Paso 2: Cargar Datos Procesados

```python
import pickle

# Cargar datos para decodificación de posición con MLP
with open('tutorial_data/processed/experiment_withCtxt_onlyPosition_flat.pickle', 'rb') as f:
    X, y, trial_markers = pickle.load(f)

print(f"Forma de X (datos de entrada): {X.shape}")
print(f"Forma de y (objetivo): {y.shape}")
print(f"Número de trials únicos: {len(np.unique(trial_markers))}")
```

## Paso 3: Configurar el Experimento

```python
from ibioml.models import MLPModel

# Configuración del modelo con optimización de hiperparámetros
config = {
    # Parámetros fijos del modelo
    "model_class": MLPModel,
    "output_size": 1,           # Decodificar solo posición
    "device": "cpu",            # Usar CPU para el tutorial
    "num_epochs": 100,          # Pocas épocas para rapidez
    "es_patience": 10,          # Early stopping
    "reg_type": None,           # Sin regularización
    "lambda_reg": None,
    "batch_size": 32,
    
    # Hiperparámetros a optimizar (formato: tipo, min, max, step/log)
    "hidden_size": (int, 64, 256, 32),     # Entre 64 y 256, paso 32
    "num_layers": (int, 1, 3),             # Entre 1 y 3 capas
    "dropout": (float, 0.0, 0.5),          # Dropout entre 0 y 0.5
    "lr": (float, 1e-4, 1e-2, True),       # Learning rate (escala log)
}

print("✅ Configuración del modelo lista")
```

## Paso 4: Ejecutar el Experimento

```python
from ibioml.tuner import run_study
import os

# Crear directorio para resultados
os.makedirs('tutorial_results', exist_ok=True)

# Ejecutar experimento (versión rápida para tutorial)
print("🚀 Iniciando experimento...")
run_study(
    X, y, trial_markers,
    model_space=config,
    num_trials=5,           # Solo 5 configuraciones para rapidez
    outer_folds=3,          # 3-fold cross-validation
    inner_folds=1,          # Sin CV interna para rapidez
    save_path="tutorial_results/position_decoding",
    search_alg="random"     # Búsqueda aleatoria
)

print("✅ Experimento completado!")
```

## Paso 5: Analizar Resultados

```python
import json
import os

# Encontrar la carpeta de resultados más reciente
results_base = "tutorial_results/position_decoding"
study_folders = [f for f in os.listdir(results_base) if f.startswith('study_')]
latest_study = sorted(study_folders)[-1]
results_path = os.path.join(results_base, latest_study)

# Cargar resultados finales
with open(os.path.join(results_path, 'final_results.json'), 'r') as f:
    final_results = json.load(f)

print("📊 Resultados del Experimento:")
print(f"   Mejor R² obtenido: {final_results['best_r2_score_test']:.4f}")
print(f"   R² promedio: {final_results['mean_r2_test']:.4f} ± {final_results['std_r2_test']:.4f}")
print(f"   Mejores hiperparámetros:")
for param, value in final_results['best_params'].items():
    print(f"     {param}: {value}")
```

## Paso 6: Visualización Básica

```python
import matplotlib.pyplot as plt
import numpy as np

# Crear un gráfico simple de los resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: R² scores por fold
fold_r2_scores = []
for fold_idx in range(3):  # 3 folds en nuestro experimento
    fold_path = os.path.join(results_path, 'training_results', f'fold_{fold_idx}', 'results.json')
    if os.path.exists(fold_path):
        with open(fold_path, 'r') as f:
            fold_data = json.load(f)
            fold_r2_scores.append(fold_data.get('r2_score', 0))

ax1.bar(range(len(fold_r2_scores)), fold_r2_scores, color='skyblue', alpha=0.7)
ax1.set_xlabel('Fold')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score por Fold')
ax1.set_ylim(0, 1)

# Gráfico 2: Distribución de una muestra de datos
sample_indices = np.random.choice(len(X), 1000, replace=False)
X_sample = X[sample_indices]
y_sample = y[sample_indices]

ax2.scatter(np.mean(X_sample, axis=1), y_sample, alpha=0.5, s=10)
ax2.set_xlabel('Actividad Neuronal Promedio')
ax2.set_ylabel('Posición')
ax2.set_title('Relación Actividad-Posición (Muestra)')

plt.tight_layout()
plt.savefig('tutorial_results/experiment_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("📈 Gráficos guardados en 'tutorial_results/experiment_summary.png'")
```

## Paso 7: Comparar Diferentes Configuraciones

```python
# Experimento adicional: comparar con y sin contexto
print("🔄 Ejecutando experimento sin contexto para comparación...")

# Cargar datos sin contexto
with open('tutorial_data/processed/experiment_onlyPosition_flat.pickle', 'rb') as f:
    X_no_ctx, y_no_ctx, T_no_ctx = pickle.load(f)

# Mismo config pero para datos sin contexto
config_no_ctx = config.copy()

run_study(
    X_no_ctx, y_no_ctx, T_no_ctx,
    model_space=config_no_ctx,
    num_trials=5,
    outer_folds=3,
    inner_folds=1,
    save_path="tutorial_results/position_decoding_no_context",
    search_alg="random"
)

print("✅ Experimento de comparación completado!")

# Comparar resultados
results_no_ctx_base = "tutorial_results/position_decoding_no_context"
study_folders_no_ctx = [f for f in os.listdir(results_no_ctx_base) if f.startswith('study_')]
latest_study_no_ctx = sorted(study_folders_no_ctx)[-1]

with open(os.path.join(results_no_ctx_base, latest_study_no_ctx, 'final_results.json'), 'r') as f:
    final_results_no_ctx = json.load(f)

print("\n📊 Comparación de Resultados:")
print(f"Con contexto:    R² = {final_results['best_r2_score_test']:.4f}")
print(f"Sin contexto:    R² = {final_results_no_ctx['best_r2_score_test']:.4f}")
print(f"Mejora:          {final_results['best_r2_score_test'] - final_results_no_ctx['best_r2_score_test']:.4f}")
```

## Próximos Pasos

¡Felicitaciones! Has completado tu primer experimento con IbioML. Ahora puedes:

### 🧠 Explorar Diferentes Modelos
```python
from ibioml.models import LSTMModel, GRUModel

# Para datos temporales (sin '_flat'), usar modelos recurrentes
lstm_config = config.copy()
lstm_config["model_class"] = LSTMModel
# ... ejecutar experimento con LSTM
```

### 📊 Análisis Avanzado
- Usar la nueva clase `Visualizer` para análisis más detallados
- Comparar múltiples experimentos simultáneamente
- Analizar importancia de características

### ⚙️ Optimización
- Aumentar `num_trials` para mejor optimización
- Usar `search_alg="bayes"` para optimización bayesiana
- Implementar validación cruzada anidada completa

### 📚 Recursos Adicionales
- [Guía de Experimentos Avanzados](../experiments.md)
- [Documentación de Modelos](../api/models.md)
- [Visualización de Resultados](../visualization.md)

---

**💡 Consejo:** Para experimentos de producción, usa configuraciones más robustas con más trials, más folds y épocas de entrenamiento más largas.

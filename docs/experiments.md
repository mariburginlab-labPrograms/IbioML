# Configuración y Ejecución de Experimentos

IbioML proporciona un sistema flexible y potente para ejecutar experimentos de neurodecodificación con optimización automática de hiperparámetros y validación cruzada anidada.

## 🎯 Visión General

Un experimento típico en IbioML incluye:

1. **Carga de datos** preprocesados
2. **Configuración del modelo** y hiperparámetros
3. **Optimización** con Optuna (Bayesian, Random, Grid Search)
4. **Validación cruzada anidada** para evaluación robusta
5. **Guardado automático** de resultados y modelos

## 🚀 Experimento Básico

### Configuración Mínima

```python
import pickle
from ibioml.models import MLPModel
from ibioml.tuner import run_study

# 1. Cargar datos preprocesados
with open('data/experimento_withCtxt_flat.pickle', 'rb') as f:
    X, y, trial_markers = pickle.load(f)

# 2. Configuración básica del modelo
config = {
    "model_class": MLPModel,
    "output_size": y.shape[1],  # 1 para single target, 2 para position+velocity
    "device": "cuda",           # "cuda" o "cpu"
    "num_epochs": 200,
    "es_patience": 10,          # Early stopping patience
    "reg_type": None,           # Regularización: None, 'l1', 'l2'
    "lambda_reg": None,
    "batch_size": 32,
    
    # Hiperparámetros fijos
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3
}

# 3. Ejecutar experimento
run_study(
    X, y, trial_markers,
    model_space=config,
    num_trials=1,              # Solo un trial (sin optimización)
    outer_folds=5,
    inner_folds=1,
    save_path="results/experimento_basico"
)
```

## 🔧 Optimización de Hiperparámetros

### Configuración con Optimización

Para activar la optimización, define hiperparámetros como tuplas:

```python
config_optimized = {
    # Parámetros fijos
    "model_class": MLPModel,
    "output_size": 1,
    "device": "cuda",
    "num_epochs": 200,
    "es_patience": 10,
    "reg_type": None,
    "lambda_reg": None,
    "batch_size": 32,
    
    # Hiperparámetros a optimizar
    "hidden_size": (int, 128, 512, 64),     # (tipo, min, max, step)
    "num_layers": (int, 1, 4),              # (tipo, min, max)
    "dropout": (float, 0.0, 0.5),           # (tipo, min, max)
    "lr": (float, 1e-5, 1e-2, True),        # (tipo, min, max, log_scale)
}

run_study(
    X, y, trial_markers,
    model_space=config_optimized,
    num_trials=50,              # 50 configuraciones diferentes
    outer_folds=5,
    inner_folds=3,              # Validación cruzada interna
    save_path="results/experimento_optimizado",
    search_alg="bayes"          # "bayes", "random", "grid"
)
```

### Formato de Hiperparámetros

#### Parámetros Enteros
```python
"hidden_size": (int, min_val, max_val, step)
"num_layers": (int, 1, 5)  # step=1 por defecto
```

#### Parámetros de Punto Flotante
```python
"dropout": (float, 0.0, 0.8)           # Escala lineal
"lr": (float, 1e-6, 1e-1, True)        # Escala logarítmica
```

## 🧠 Modelos Disponibles

### Modelos para Datos Aplanados (`*_flat.pickle`)

```python
from ibioml.models import MLPModel

mlp_config = {
    "model_class": MLPModel,
    "hidden_size": (int, 64, 1024, 32),
    "num_layers": (int, 1, 5),
    "dropout": (float, 0.0, 0.7),
    "lr": (float, 1e-5, 1e-2, True),
    # ... otros parámetros
}
```

### Modelos para Datos Temporales (`.pickle`)

```python
from ibioml.models import RNNModel, LSTMModel, GRUModel

# RNN básica
rnn_config = {
    "model_class": RNNModel,
    "hidden_size": (int, 32, 256, 16),
    "num_layers": (int, 1, 3),
    "dropout": (float, 0.0, 0.5),
    "lr": (float, 1e-5, 1e-2, True),
    # ... otros parámetros
}

# LSTM (recomendado para secuencias largas)
lstm_config = {
    "model_class": LSTMModel,
    "hidden_size": (int, 64, 512, 32),
    "num_layers": (int, 1, 4),
    "dropout": (float, 0.0, 0.6),
    "lr": (float, 1e-5, 1e-2, True),
    # ... otros parámetros
}
```

## 📊 Tipos de Experimentos

### Experimento Single-Target

```python
# Solo decodificar posición
with open('data/experimento_withCtxt_onlyPosition_flat.pickle', 'rb') as f:
    X_pos, y_pos, T = pickle.load(f)

position_config = {
    "model_class": MLPModel,
    "output_size": 1,  # Una sola salida
    "hidden_size": (int, 128, 512, 64),
    "lr": (float, 1e-5, 1e-2, True),
    # ... resto de configuración
}

run_study(X_pos, y_pos, T, model_space=position_config, 
          save_path="results/position_decoding")
```

### Experimento Multi-Target

```python
# Decodificar posición y velocidad simultáneamente
with open('data/experimento_withCtxt_bothTargets_flat.pickle', 'rb') as f:
    X_both, y_both, T = pickle.load(f)

dual_config = {
    "model_class": MLPModel,
    "output_size": 2,  # Posición + velocidad
    "hidden_size": (int, 256, 1024, 64),  # Redes más grandes para dual-output
    "lr": (float, 1e-5, 1e-2, True),
    # ... resto de configuración
}

run_study(X_both, y_both, T, model_space=dual_config,
          save_path="results/dual_target_decoding")
```

### Comparación de Arquitecturas

```python
# Función helper para experimentos comparativos
def run_architecture_comparison(X, y, T, base_path):
    architectures = {
        'mlp': MLPModel,
        'rnn': RNNModel,
        'lstm': LSTMModel,
        'gru': GRUModel
    }
    
    base_config = {
        "output_size": y.shape[1],
        "device": "cuda",
        "num_epochs": 150,
        "batch_size": 32,
        "hidden_size": (int, 128, 256, 32),
        "lr": (float, 1e-4, 1e-2, True),
    }
    
    for arch_name, model_class in architectures.items():
        config = base_config.copy()
        config["model_class"] = model_class
        
        run_study(
            X, y, T,
            model_space=config,
            num_trials=20,
            outer_folds=5,
            save_path=f"{base_path}/{arch_name}"
        )
        print(f"✅ Completado: {arch_name}")

# Ejecutar comparación
run_architecture_comparison(X, y, T, "results/architecture_comparison")
```

## ⚙️ Configuración Avanzada

### Algoritmos de Optimización

```python
# Optimización Bayesiana (recomendado)
run_study(X, y, T, model_space=config, 
          search_alg="bayes", num_trials=50)

# Búsqueda aleatoria (para espacios grandes)
run_study(X, y, T, model_space=config,
          search_alg="random", num_trials=100)

# Búsqueda en grilla (para espacios pequeños)
run_study(X, y, T, model_space=config,
          search_alg="grid", num_trials=25)
```

### Configuración de Validación Cruzada

```python
# Validación cruzada estándar
run_study(X, y, T, model_space=config,
          outer_folds=5,    # 5-fold CV externo
          inner_folds=3)    # 3-fold CV interno

# Para datasets pequeños
run_study(X, y, T, model_space=config,
          outer_folds=3,
          inner_folds=1)    # Sin CV interno

# Para evaluación robusta
run_study(X, y, T, model_space=config,
          outer_folds=10,   # 10-fold CV
          inner_folds=5)
```

### Regularización

```python
# Sin regularización
config = {
    "reg_type": None,
    "lambda_reg": None,
    # ... otros parámetros
}

# Con regularización L2
config = {
    "reg_type": "l2",
    "lambda_reg": (float, 1e-6, 1e-2, True),
    # ... otros parámetros
}

# Con regularización L1
config = {
    "reg_type": "l1", 
    "lambda_reg": (float, 1e-5, 1e-1, True),
    # ... otros parámetros
}
```

## 📁 Estructura de Resultados

### Organización Automática

```
results/
├── experimento_basico/
│   └── study_2024-01-15_14-30-25/    # Timestamp automático
│       ├── final_results.json         # Resultados finales
│       └── training_results/          # Resultados por fold
│           ├── fold_0/
│           │   ├── results.json       # Métricas del fold
│           │   └── best_model.pt      # Mejor modelo del fold
│           ├── fold_1/
│           └── ...
```

### Contenido de Resultados

```python
# final_results.json
{
    "best_r2_score_test": 0.847,
    "best_params": {
        "hidden_size": 256,
        "lr": 0.0031,
        "dropout": 0.23
    },
    "mean_r2_test": 0.821,
    "std_r2_test": 0.045,
    "study_name": "study_2024-01-15_14-30-25",
    "total_trials": 50,
    "experiment_duration_minutes": 23.5
}
```

## 🔍 Monitoreo de Experimentos

### Seguimiento en Tiempo Real

```python
import optuna

# Visualizar progreso (requiere optuna-dashboard)
study = optuna.load_study(
    study_name="mi_experimento",
    storage="sqlite:///results/optuna_studies.db"
)

# Gráficos de optimización
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

fig = optuna.visualization.plot_param_importances(study)
fig.show()
```

### Logs y Debugging

```python
import logging

# Configurar logging detallado
logging.basicConfig(level=logging.INFO)

# Ejecutar con logs verbosos
run_study(X, y, T, model_space=config,
          save_path="results/experimento_debug",
          num_trials=5)  # Pocas iteraciones para debug
```

## 🚨 Solución de Problemas

### Errores Comunes

!!! warning "CUDA out of memory"
    ```python
    config = {
        "batch_size": 16,    # Reducir tamaño de lote
        "hidden_size": (int, 64, 256, 32),  # Redes más pequeñas
        # ... otros parámetros
    }
    ```

!!! warning "Experimento muy lento"
    ```python
    config = {
        "num_epochs": 50,     # Menos épocas por trial
        "es_patience": 5,     # Early stopping más agresivo
        # ... otros parámetros
    }
    
    run_study(X, y, T, model_space=config,
              num_trials=10,   # Menos trials
              outer_folds=3)   # Menos folds
    ```

!!! warning "Resultados inconsistentes"
    ```python
    # Fijar semillas para reproducibilidad
    import torch
    import numpy as np
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = {
        "device": "cpu",  # Para máxima reproducibilidad
        # ... otros parámetros
    }
    ```

## 📈 Mejores Prácticas

### Configuración de Producción

```python
production_config = {
    "model_class": MLPModel,
    "output_size": 1,
    "device": "cuda",
    "num_epochs": 300,
    "es_patience": 15,
    "batch_size": 64,
    
    # Espacio de búsqueda bien definido
    "hidden_size": (int, 128, 512, 32),
    "num_layers": (int, 2, 4),
    "dropout": (float, 0.1, 0.5),
    "lr": (float, 1e-5, 1e-2, True),
}

run_study(
    X, y, T,
    model_space=production_config,
    num_trials=100,           # Búsqueda exhaustiva
    outer_folds=10,           # Evaluación robusta
    inner_folds=5,
    save_path="results/production_experiment",
    search_alg="bayes"
)
```

### Experimentos en Lotes

```python
def batch_experiments():
    datasets = [
        'data/S19_withCtxt_flat.pickle',
        'data/S20_withCtxt_flat.pickle', 
        'data/S21_withCtxt_flat.pickle'
    ]
    
    for dataset_path in datasets:
        subject_id = dataset_path.split('/')[-1].split('_')[0]
        
        with open(dataset_path, 'rb') as f:
            X, y, T = pickle.load(f)
        
        run_study(
            X, y, T,
            model_space=production_config,
            num_trials=50,
            outer_folds=5,
            save_path=f"results/batch_experiment/{subject_id}"
        )
        
        print(f"✅ Completado: {subject_id}")

batch_experiments()
```

## 📊 Próximos Pasos

Después de ejecutar experimentos:

1. **[Visualizar resultados →](visualization.md)** Análisis y gráficos
2. **[API Reference →](api/training.md)** Documentación técnica detallada
3. **[Ejemplos completos →](examples/full_experiment.md)** Casos de uso avanzados

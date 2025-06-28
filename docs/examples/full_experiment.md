# Experimento Completo

Este tutorial muestra cómo realizar un experimento completo de neurodecodificación, desde el preprocesamiento de datos hasta la visualización de resultados.

## Prerrequisitos

Asegúrate de tener instalado IbioML y sus dependencias:

```bash
pip install -e .
```

## 1. Preparación de Datos

### Cargar archivo .mat

```python
from ibioml.preprocessing import preprocess_data

# Preprocesar los datos
preprocess_data(
    file_path='datasets/L5_bins200ms_completo.mat',
    file_name_to_save='bins200ms_preprocessed',
    bins_before=5,
    bins_after=5,
    bins_current=1,
    threshDPrime=2.5,
    firingMinimo=1000
)
```

### Cargar datos preprocesados

```python
import pickle

# Cargar datos para modelos no recurrentes (MLP)
with open('data/bins200ms_preprocessed_withCtxt_flat.pickle', 'rb') as f:
    X_flat, y_flat, T = pickle.load(f)

# Cargar datos para modelos recurrentes (RNN/LSTM)
with open('data/bins200ms_preprocessed_withCtxt.pickle', 'rb') as f:
    X_tensor, y_tensor, T_tensor = pickle.load(f)

print(f"Forma de datos planos: {X_flat.shape}")
print(f"Forma de datos tensoriales: {X_tensor.shape}")
```

## 2. Configuración del Experimento

### Definir modelos a comparar

```python
from ibioml.models import MLPModel, LSTMModel
from ibioml.utils.model_factory import create_model_class

# Configuración para MLP
mlp_space = {
    "model_class": create_model_class(MLPModel, y_flat.shape[1]),
    "output_size": 2,  # posición y velocidad
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_epochs": 200,
    "batch_size": 32,
    "hidden_layer_sizes": [128, 64],
    "dropout_rate": 0.3,
    "learning_rate": 0.001
}

# Configuración para LSTM
lstm_space = {
    "model_class": create_model_class(LSTMModel, y_tensor.shape[1]),
    "output_size": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_epochs": 200,
    "batch_size": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout_rate": 0.3,
    "learning_rate": 0.001
}
```

## 3. Ejecutar Experimentos

### Experimento con MLP

```python
from ibioml.tuner import run_study

# Ejecutar estudio para MLP
mlp_results = run_study(
    X_flat, y_flat, T,
    model_space=mlp_space,
    num_trials=10,
    outer_folds=5,
    inner_folds=3,
    save_path="results/mlp_comparison"
)
```

### Experimento con LSTM

```python
# Ejecutar estudio para LSTM
lstm_results = run_study(
    X_tensor, y_tensor, T_tensor,
    model_space=lstm_space,
    num_trials=10,
    outer_folds=5,
    inner_folds=3,
    save_path="results/lstm_comparison"
)
```

## 4. Análisis de Resultados

### Cargar resultados guardados

```python
from ibioml.plots import load_results, extract_r2_scores

# Cargar resultados de múltiples experimentos
results_dict = load_results({
    'MLP': 'results/mlp_comparison/study_2024-01-01_12-00-00/final_results.json',
    'LSTM': 'results/lstm_comparison/study_2024-01-01_13-00-00/final_results.json'
})
```

### Extraer métricas de rendimiento

```python
# Extraer scores R²
test_r2_scores, test_r2_pos, test_r2_vel, r2_df = extract_r2_scores(results_dict)

print("Resultados promedio por modelo:")
print(r2_df.groupby('model').agg({
    'r2_both': ['mean', 'std'],
    'r2_position': ['mean', 'std'],
    'r2_velocity': ['mean', 'std']
}))
```

## 5. Visualización

### Gráficos de comparación

```python
from ibioml.plots import (
    boxplot_test_r2_scores_both_targets,
    plot_learning_curves,
    plot_predictions_vs_real
)

# Boxplot comparativo
boxplot_test_r2_scores_both_targets(r2_df)

# Curvas de aprendizaje (si están disponibles)
plot_learning_curves(results_dict)

# Predicciones vs valores reales
plot_predictions_vs_real(results_dict, 'MLP')
```

### Análisis estadístico

```python
from scipy import stats

# Test t para comparar modelos
mlp_scores = r2_df[r2_df['model'] == 'MLP']['r2_both']
lstm_scores = r2_df[r2_df['model'] == 'LSTM']['r2_both']

t_stat, p_value = stats.ttest_ind(mlp_scores, lstm_scores)
print(f"Diferencia significativa entre MLP y LSTM: p = {p_value:.4f}")
```

## 6. Guardar Reporte

```python
import matplotlib.pyplot as plt

# Crear reporte completo
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1: Boxplot de rendimiento
axes[0, 0].boxplot([mlp_scores, lstm_scores], labels=['MLP', 'LSTM'])
axes[0, 0].set_title('Comparación de Rendimiento')
axes[0, 0].set_ylabel('R² Score')

# Gráfico 2: Distribución de scores
axes[0, 1].hist(mlp_scores, alpha=0.7, label='MLP', bins=10)
axes[0, 1].hist(lstm_scores, alpha=0.7, label='LSTM', bins=10)
axes[0, 1].set_title('Distribución de Scores')
axes[0, 1].legend()

# Gráfico 3: Rendimiento por target
targets = ['Posición', 'Velocidad']
mlp_by_target = [r2_df[r2_df['model'] == 'MLP']['r2_position'].mean(),
                 r2_df[r2_df['model'] == 'MLP']['r2_velocity'].mean()]
lstm_by_target = [r2_df[r2_df['model'] == 'LSTM']['r2_position'].mean(),
                  r2_df[r2_df['model'] == 'LSTM']['r2_velocity'].mean()]

x = range(len(targets))
axes[1, 0].bar([i - 0.2 for i in x], mlp_by_target, 0.4, label='MLP')
axes[1, 0].bar([i + 0.2 for i in x], lstm_by_target, 0.4, label='LSTM')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(targets)
axes[1, 0].set_title('Rendimiento por Target')
axes[1, 0].legend()

# Gráfico 4: Tiempo de entrenamiento (si está disponible)
axes[1, 1].text(0.5, 0.5, 'Estadísticas del Experimento\n\n' +
                f'MLP R² promedio: {mlp_scores.mean():.3f} ± {mlp_scores.std():.3f}\n' +
                f'LSTM R² promedio: {lstm_scores.mean():.3f} ± {lstm_scores.std():.3f}\n' +
                f'Diferencia significativa: {"Sí" if p_value < 0.05 else "No"} (p={p_value:.4f})',
                ha='center', va='center', fontsize=12)
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('results/experiment_summary.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Conclusiones

Este flujo de trabajo te permite:

1. **Preparar datos** de manera consistente y reproducible
2. **Configurar experimentos** con diferentes arquitecturas de modelos
3. **Ejecutar estudios** con validación cruzada anidada
4. **Comparar resultados** de manera estadísticamente rigurosa
5. **Visualizar hallazgos** con gráficos informativos
6. **Documentar experimentos** para reproducibilidad

Para experimentos más avanzados, puedes:

- Explorar diferentes arquitecturas de red
- Ajustar hiperparámetros con Optuna
- Implementar técnicas de regularización
- Analizar la importancia de características
- Realizar análisis de sensibilidad

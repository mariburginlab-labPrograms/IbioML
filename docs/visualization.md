# Visualización de Resultados

IBioML incluye un sistema de visualización moderno y flexible para analizar los resultados de tus experimentos de neurodecodificación.

## 🎯 Visión General

El sistema de visualización de IBioML se basa en dos componentes principales:

1. **`ExperimentResults`**: Gestión eficiente de datos de experimentos
2. **`Visualizer`**: Generación de gráficos y análisis visual

## 🚀 Uso Básico

### Visualización de un Solo Experimento

```python
from ibioml.results import Visualizer

# Crear visualizador para un experimento
viz = Visualizer("results/mi_experimento/study_2024-01-15_14-30-25")

# Resumen rápido del experimento
viz.summary()

# Boxplot de R² scores
ax = viz.plot_r2_scores_boxplot()
plt.show()
```

### Comparación de Múltiples Experimentos

```python
from ibioml.results import MultiExperimentResults, MultiExperimentVisualizer

# Cargar múltiples experimentos
multi_results = MultiExperimentResults("results/architecture_comparison")

# Crear visualizador comparativo
multi_viz = MultiExperimentVisualizer(multi_results)

# Comparar R² scores entre experimentos
ax = multi_viz.plot_r2_comparison_boxplot()
plt.show()
```

## 📊 Tipos de Visualizaciones

### 1. Boxplots de Rendimiento

```python
import matplotlib.pyplot as plt

# Visualizar distribución de R² scores
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Comparar diferentes configuraciones
experiments = {
    'MLP con contexto': 'results/mlp_with_context',
    'MLP sin contexto': 'results/mlp_no_context', 
    'LSTM con contexto': 'results/lstm_with_context',
    'RNN con contexto': 'results/rnn_with_context'
}

for i, (name, path) in enumerate(experiments.items()):
    ax = axes[i//2, i%2]
    viz = Visualizer(path)
    viz.plot_r2_scores_boxplot(ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.show()
```

### 2. Predicciones vs. Valores Reales

```python
# Scatter plot de predicciones vs. valores verdaderos
viz = Visualizer("results/mi_experimento/study_2024-01-15_14-30-25")

# Para el primer fold
fold_names = viz.results_loader.get_fold_names()
if fold_names:
    ax = viz.plot_predictions_vs_true(fold_names[0])
    if ax:
        plt.show()
```

### 3. Análisis de Hiperparámetros

```python
import optuna
import matplotlib.pyplot as plt

# Cargar estudio de Optuna (si disponible)
def plot_hyperparameter_analysis(study_path):
    # Cargar estudio desde la base de datos de Optuna
    study = optuna.load_study(
        study_name="mi_estudio",
        storage=f"sqlite:///{study_path}/optuna_study.db"
    )
    
    # Historia de optimización
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.show()
    
    # Importancia de parámetros
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.show()
    
    # Distribución de parámetros
    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    fig3.show()

# plot_hyperparameter_analysis("results/mi_experimento")
```

### 4. Curvas de Aprendizaje

```python
def plot_learning_curves(results_path):
    """
    Graficar curvas de entrenamiento y validación por fold.
    """
    viz = Visualizer(results_path)
    fold_names = viz.results_loader.get_fold_names()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, fold_name in enumerate(fold_names[:6]):  # Máximo 6 folds
        if i >= len(axes):
            break
            
        fold_results = viz.results_loader.get_fold_summary(fold_name)
        
        # Verificar si hay datos de curvas de aprendizaje
        if 'train_losses' in fold_results and 'val_losses' in fold_results:
            epochs = range(len(fold_results['train_losses']))
            
            axes[i].plot(epochs, fold_results['train_losses'], 
                        label='Entrenamiento', color='blue', alpha=0.7)
            axes[i].plot(epochs, fold_results['val_losses'], 
                        label='Validación', color='red', alpha=0.7)
            axes[i].set_title(f'{fold_name}')
            axes[i].set_xlabel('Época')
            axes[i].set_ylabel('Loss')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# plot_learning_curves("results/mi_experimento/study_2024-01-15_14-30-25")
```

## 📈 Visualizaciones Avanzadas

### 1. Heatmap de Rendimiento por Configuración

```python
import seaborn as sns
import pandas as pd

def create_performance_heatmap(multi_results_path):
    """
    Crear heatmap de rendimiento para múltiples experimentos.
    """
    multi_results = MultiExperimentResults(multi_results_path)
    
    # Recopilar datos de todos los experimentos
    performance_data = []
    for exp_name, exp_results in multi_results.experiments.items():
        final_results = exp_results.final_results
        performance_data.append({
            'Experimento': exp_name,
            'R² Test': final_results.get('best_r2_score_test', 0),
            'R² Promedio': final_results.get('mean_r2_test', 0),
            'Desviación Estándar': final_results.get('std_r2_test', 0)
        })
    
    df = pd.DataFrame(performance_data)
    df_pivot = df.set_index('Experimento')[['R² Test', 'R² Promedio', 'Desviación Estándar']]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_pivot.T, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Rendimiento por Experimento')
    plt.tight_layout()
    plt.show()

# create_performance_heatmap("results/architecture_comparison")
```

### 2. Distribución de Errores

```python
def plot_error_distribution(results_path, fold_name):
    """
    Analizar distribución de errores de predicción.
    """
    viz = Visualizer(results_path)
    data = viz.get_predictions_and_true_values(fold_name)
    
    if data is None:
        print(f"No hay datos disponibles para {fold_name}")
        return
    
    predictions, true_values = data
    errors = predictions - true_values
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Histograma de errores
    axes[0].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribución de Errores')
    axes[0].set_xlabel('Error (Predicción - Real)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Q-Q plot para normalidad
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normalidad)')
    
    # Errores vs. valores predichos
    axes[2].scatter(predictions, errors, alpha=0.5, s=10)
    axes[2].set_xlabel('Valores Predichos')
    axes[2].set_ylabel('Errores')
    axes[2].set_title('Errores vs. Predicciones')
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas del error
    print(f"📊 Estadísticas de Error para {fold_name}:")
    print(f"   Error medio: {np.mean(errors):.4f}")
    print(f"   Error estándar: {np.std(errors):.4f}")
    print(f"   MAE: {np.mean(np.abs(errors)):.4f}")
    print(f"   RMSE: {np.sqrt(np.mean(errors**2)):.4f}")

# fold_names = Visualizer("results/mi_experimento").results_loader.get_fold_names()
# plot_error_distribution("results/mi_experimento", fold_names[0])
```

### 3. Análisis Temporal

```python
def plot_temporal_analysis(results_path, fold_name):
    """
    Analizar rendimiento a través del tiempo (dentro de trials).
    """
    viz = Visualizer(results_path)
    
    # Cargar datos de predicciones y trial markers
    with open('data/mi_experimento_flat.pickle', 'rb') as f:
        _, _, trial_markers = pickle.load(f)
    
    data = viz.get_predictions_and_true_values(fold_name)
    if data is None:
        return
    
    predictions, true_values = data
    
    # Calcular R² por trial
    unique_trials = np.unique(trial_markers)
    trial_r2_scores = []
    
    for trial in unique_trials:
        trial_mask = trial_markers == trial
        if np.sum(trial_mask) > 10:  # Suficientes puntos en el trial
            trial_pred = predictions[trial_mask]
            trial_true = true_values[trial_mask]
            
            # Calcular R² para este trial
            ss_res = np.sum((trial_true - trial_pred) ** 2)
            ss_tot = np.sum((trial_true - np.mean(trial_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            trial_r2_scores.append(r2)
        else:
            trial_r2_scores.append(np.nan)
    
    # Visualizar
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # R² por trial
    valid_scores = [s for s in trial_r2_scores if not np.isnan(s)]
    axes[0].plot(range(len(trial_r2_scores)), trial_r2_scores, 
                 marker='o', alpha=0.7, markersize=3)
    axes[0].axhline(np.mean(valid_scores), color='red', linestyle='--', 
                    label=f'Promedio: {np.mean(valid_scores):.3f}')
    axes[0].set_title('R² Score por Trial')
    axes[0].set_xlabel('Número de Trial')
    axes[0].set_ylabel('R² Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribución de R² por trial
    axes[1].hist(valid_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_title('Distribución de R² por Trial')
    axes[1].set_xlabel('R² Score')
    axes[1].set_ylabel('Número de Trials')
    axes[1].axvline(np.mean(valid_scores), color='red', linestyle='--')
    
    plt.tight_layout()
    plt.show()

# plot_temporal_analysis("results/mi_experimento", fold_names[0])
```

## 🛠️ Personalización Avanzada

### Temas y Estilos

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo global
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# O usar tema personalizado
def setup_publication_style():
    """Configurar estilo para publicaciones."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })

setup_publication_style()
```

### Exportación de Gráficos

```python
def save_publication_figures(results_path, output_dir="figures"):
    """
    Generar y guardar figuras listas para publicación.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    viz = Visualizer(results_path)
    
    # Figura 1: Boxplot principal
    fig, ax = plt.subplots(figsize=(6, 4))
    viz.plot_r2_scores_boxplot(ax=ax)
    plt.savefig(f'{output_dir}/r2_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/r2_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figura 2: Predicciones vs. reales
    fold_names = viz.results_loader.get_fold_names()
    if fold_names:
        fig, ax = plt.subplots(figsize=(6, 6))
        viz.plot_predictions_vs_true(fold_names[0], ax=ax)
        plt.savefig(f'{output_dir}/predictions_vs_true.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/predictions_vs_true.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Figuras guardadas en '{output_dir}/'")

# save_publication_figures("results/mi_experimento")
```

## 📊 Dashboard Interactivo

### Usando Plotly para Interactividad

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_interactive_dashboard(multi_results_path):
    """
    Crear dashboard interactivo con Plotly.
    """
    multi_results = MultiExperimentResults(multi_results_path)
    
    # Recopilar datos para dashboard
    all_data = []
    for exp_name, exp_results in multi_results.experiments.items():
        fold_names = exp_results.get_fold_names()
        for fold_name in fold_names:
            try:
                fold_summary = exp_results.get_fold_summary(fold_name)
                all_data.append({
                    'Experimento': exp_name,
                    'Fold': fold_name,
                    'R2_Score': fold_summary.get('r2_score', 0),
                    'Loss_Val': fold_summary.get('loss_val', 0)
                })
            except:
                continue
    
    df = pd.DataFrame(all_data)
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² por Experimento', 'Distribución de R²', 
                       'Loss de Validación', 'Comparación Detallada'),
        specs=[[{"type": "box"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Boxplot de R²
    for exp in df['Experimento'].unique():
        exp_data = df[df['Experimento'] == exp]
        fig.add_trace(
            go.Box(y=exp_data['R2_Score'], name=exp, showlegend=False),
            row=1, col=1
        )
    
    # Histograma de R²
    fig.add_trace(
        go.Histogram(x=df['R2_Score'], nbinsx=20, showlegend=False),
        row=1, col=2
    )
    
    # Scatter de R² vs Loss
    for exp in df['Experimento'].unique():
        exp_data = df[df['Experimento'] == exp]
        fig.add_trace(
            go.Scatter(
                x=exp_data['Loss_Val'], 
                y=exp_data['R2_Score'],
                mode='markers',
                name=exp,
                text=exp_data['Fold'],
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Barplot promedio por experimento
    exp_means = df.groupby('Experimento')['R2_Score'].mean()
    fig.add_trace(
        go.Bar(x=exp_means.index, y=exp_means.values, showlegend=False),
        row=2, col=2
    )
    
    # Actualizar layout
    fig.update_layout(
        height=800,
        title_text="Dashboard de Resultados de Neurodecodificación",
        showlegend=True
    )
    
    # Mostrar dashboard
    fig.show()
    
    # Guardar como HTML
    fig.write_html("dashboard_resultados.html")
    print("✅ Dashboard guardado como 'dashboard_resultados.html'")

# create_interactive_dashboard("results/architecture_comparison")
```

## 🚨 Solución de Problemas

### Errores Comunes

!!! warning "No se encuentran datos de predicciones"
    ```python
    # Verificar estructura de resultados
    viz = Visualizer("results/mi_experimento")
    fold_summary = viz.results_loader.get_fold_summary("fold_0")
    print("Claves disponibles:", fold_summary.keys())
    ```

!!! warning "Gráficos vacíos o con errores"
    ```python
    # Verificar datos antes de graficar
    r2_df = viz.get_r2_scores_dataframe()
    if r2_df.empty:
        print("❌ No hay datos de R² disponibles")
    else:
        print("✅ Datos encontrados:", len(r2_df), "filas")
    ```

!!! warning "Problemas de memoria con datasets grandes"
    ```python
    # Limpiar caché regularmente
    viz.results_loader.clear_cache()
    
    # O usar submuestreo para visualización
    sample_size = min(1000, len(predictions))
    sample_idx = np.random.choice(len(predictions), sample_size, replace=False)
    pred_sample = predictions[sample_idx]
    true_sample = true_values[sample_idx]
    ```

## 📈 Próximos Pasos

- **[API de Resultados →](api/results.md)** Documentación técnica detallada
- **[Ejemplos Avanzados →](examples/full_experiment.md)** Casos de uso complejos
- **[Contribuir →](contributing.md)** Agregar nuevas visualizaciones

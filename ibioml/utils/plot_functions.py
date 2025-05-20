import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe  # Añadir esta importación
from utils.plot_styles import set_plotting_style, get_model_colors, get_strategy_colors

def enhanced_boxplot_test_r2_scores(r2_df, save_path=None, title="Comparación de R2 Scores entre Modelos"):
    """
    Crea un boxplot mejorado para comparar R2 scores entre modelos.
    
    Args:
        r2_df: DataFrame con datos de R2 scores en formato largo
        save_path: Ruta para guardar la figura
        title: Título del gráfico
    """
    set_plotting_style()
    model_colors = get_model_colors()
    
    # Calcular estadísticas para anotaciones
    medians = r2_df.groupby(['Modelo'])['R2 Score'].median()
    means = r2_df.groupby(['Modelo'])['R2 Score'].mean()
    
    # Crear la figura con un tamaño adecuado
    plt.figure(figsize=(12, 8), dpi=200)
    
    # Crear el boxplot principal
    ax = sns.boxplot(x='Modelo', y='R2 Score', data=r2_df, 
                      palette=model_colors, width=0.5,
                      boxprops=dict(alpha=0.7),
                      medianprops=dict(color='#333333', linewidth=2.5))
    
    # Agregar puntos individuales con jitter
    sns.stripplot(x='Modelo', y='R2 Score', data=r2_df, 
                  palette=model_colors, alpha=0.6, size=8,
                  jitter=0.2, marker='o', linewidth=1, 
                  edgecolor='w')
    
    # Mejoras visuales
    ax.set_xlabel('Modelo', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('R² Score', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim([0, 1])
    
    # Ajustar el fondo
    ax.set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('#ffffff')
    
    # Agregar líneas horizontales en valores clave
    plt.axhline(y=0.5, color='#999999', linestyle='--', alpha=0.5)
    plt.axhline(y=0.7, color='#999999', linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='#999999', linestyle='--', alpha=0.5)
    
    # Añadir anotaciones para los valores de mediana y media
    for i, model in enumerate(r2_df['Modelo'].unique()):
        # Anotación de la mediana
        plt.annotate(f'Mediana: {medians[model]:.3f}',
                    xy=(i, medians[model] + 0.03), 
                    xytext=(i, medians[model] + 0.07),
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold',
                    color=model_colors[model],
                    arrowprops=dict(arrowstyle='->', color=model_colors[model]))
        
        # Anotación de la media con un marcador diferente
        plt.scatter(i, means[model], marker='*', s=200, 
                   color=model_colors[model], edgecolor='white', zorder=3)
        plt.annotate(f'Media: {means[model]:.3f}', 
                    xy=(i, means[model] - 0.03),
                    xytext=(i, means[model] - 0.07),
                    ha='center', va='top',
                    fontsize=12, fontweight='bold',
                    color=model_colors[model],
                    arrowprops=dict(arrowstyle='->', color=model_colors[model]))
    
    # Guardar la figura si se proporciona una ruta
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    plt.show()


def enhanced_comparison_boxplot(combined_df, save_path=None):
    """
    Crea un boxplot mejorado para comparar diferentes estrategias de CV.
    
    Args:
        combined_df: DataFrame con datos en formato largo
        save_path: Ruta para guardar la figura
    """
    set_plotting_style()
    model_colors = get_model_colors()
    strategy_colors = get_strategy_colors()
    
    # Calcular medias para anotaciones
    mean_scores = combined_df.groupby(['CV_Strategy', 'Model'])['R2_Score'].mean().reset_index()
    
    plt.figure(figsize=(16, 10), dpi=200)
    
    # Crear el boxplot
    ax = sns.boxplot(x='Model', y='R2_Score', hue='CV_Strategy', data=combined_df,
                     palette=strategy_colors, width=0.7,
                     boxprops=dict(alpha=0.7),
                     medianprops=dict(color='#333333', linewidth=2))
    
    # Añadir puntos individuales
    swarm = sns.swarmplot(x='Model', y='R2_Score', hue='CV_Strategy', data=combined_df,
                         palette=strategy_colors, alpha=0.7, size=5,
                         dodge=True, edgecolor='none')
    
    # Quitar la leyenda duplicada
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:len(strategy_colors)], labels[:len(strategy_colors)],
               title='Estrategia de CV', bbox_to_anchor=(1.05, 1), 
               loc='upper left', borderaxespad=0,
               frameon=True, fancybox=True, shadow=True)
    
    # Mejoras visuales
    plt.title('Comparación de Estrategias de Cross-Validation', 
              fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Modelo', fontsize=16, fontweight='bold', labelpad=15)
    plt.ylabel('R² Score', fontsize=16, fontweight='bold', labelpad=15)
    plt.ylim([0, 1])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Fondo y bordes
    ax.set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('#ffffff')
    
    # Agregar líneas horizontales en valores clave
    plt.axhline(y=0.5, color='#999999', linestyle='--', alpha=0.5)
    plt.axhline(y=0.7, color='#999999', linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='#999999', linestyle='--', alpha=0.5)
    
    # Agregar anotaciones para los valores medios
    dodge_positions = {0: -0.3, 1: 0, 2: 0.3}
    strategies = combined_df['CV_Strategy'].unique()
    
    for model_idx, model in enumerate(combined_df['Model'].unique()):
        for strat_idx, strategy in enumerate(strategies):
            mean_value = mean_scores[(mean_scores['Model'] == model) & 
                                    (mean_scores['CV_Strategy'] == strategy)]['R2_Score'].values[0]
            
            plt.annotate(f'{mean_value:.3f}',
                        xy=(model_idx + dodge_positions[strat_idx], mean_value),
                        xytext=(model_idx + dodge_positions[strat_idx], mean_value + 0.05),
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color=strategy_colors[strategy],
                        arrowprops=dict(arrowstyle='-', color=strategy_colors[strategy], alpha=0.5))
    
    # Guardar la figura si se proporciona una ruta
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    plt.show()


def enhanced_plot_predictions(predictions, true_values, test_r2_scores, fold_to_plot=0, 
                             models=None, save_path=None, limit=1000):
    """
    Crea un gráfico mejorado para mostrar predicciones vs valores reales.
    
    Args:
        predictions: Diccionario de predicciones por modelo y fold
        true_values: Diccionario de valores reales por modelo y fold
        test_r2_scores: Diccionario de R2 scores por modelo
        fold_to_plot: Índice del fold a visualizar
        models: Lista de modelos a incluir
        save_path: Ruta para guardar la figura
        limit: Número de puntos a mostrar
    """
    set_plotting_style()
    model_colors = get_model_colors()
    
    if models is None:
        models = predictions.keys()
    
    # Calcular el número de filas y columnas para los subplots
    n_models = len(models)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Crear la figura con subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), dpi=200, sharex=True)
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Configurar el fondo para toda la figura
    fig.patch.set_facecolor('#ffffff')
    
    for i, (ax, model) in enumerate(zip(axes, models)):
        # Obtener datos para el modelo
        y_test = true_values[model][fold_to_plot][:limit]
        y_pred = predictions[model][fold_to_plot][:limit]
        r2 = test_r2_scores[model][fold_to_plot]
        
        # Configurar el fondo del subplot
        ax.set_facecolor('#f8f9fa')
        
        # Graficar valores reales
        real_line = ax.plot(y_test, label='Valores Reales', 
                           color='#333333', linewidth=2, alpha=0.8,
                           path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
        
        # Graficar predicciones
        model_color = model_colors.get(model.upper(), '#1f77b4')
        pred_line = ax.plot(y_pred, label='Predicciones', 
                           color=model_color, linewidth=2.5, alpha=0.9,
                           path_effects=[pe.Stroke(linewidth=3.5, foreground='white', alpha=0.5), pe.Normal()])
        
        # Mejoras visuales
        ax.set_title(f'{model.upper()} (R² = {r2:.3f})', 
                    fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('Tiempo (bins)', fontsize=12, labelpad=10)
        ax.set_ylabel('Posición (cm)', fontsize=12, labelpad=10)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Área sombreada para mostrar diferencias
        ax.fill_between(range(len(y_test)), y_test, y_pred, 
                       color=model_color, alpha=0.15)
        
        # Destacar áreas con buenas y malas predicciones
        errors = np.abs(y_test - y_pred)
        error_threshold = np.percentile(errors, 90)  # Top 10% errores
        
        # Marcar áreas con errores grandes
        highlight_indices = np.where(errors > error_threshold)[0]
        if len(highlight_indices) > 0:
            clusters = []
            current_cluster = [highlight_indices[0]]
            
            for j in range(1, len(highlight_indices)):
                if highlight_indices[j] - highlight_indices[j-1] <= 5:  # Puntos cercanos
                    current_cluster.append(highlight_indices[j])
                else:
                    if len(current_cluster) >= 3:  # Cluster significativo
                        clusters.append(current_cluster)
                    current_cluster = [highlight_indices[j]]
            
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
                
            # Sombrear áreas con errores grandes
            for cluster in clusters[:3]:  # Limitar a 3 clusters para no sobrecargar
                start, end = max(0, min(cluster)-2), min(len(y_test)-1, max(cluster)+2)
                ax.axvspan(start, end, color='#ff9999', alpha=0.3, label='_nolegend_')
        
        # Leyenda
        ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, 
                 shadow=True, fontsize=10)
    
    # Ajustar el espacio entre subplots
    plt.tight_layout(pad=3.0)
    
    # Título general
    fig.suptitle('Predicciones de Posición vs. Valores Reales', 
                fontsize=20, fontweight='bold', y=1.02)
    
    # Guardar la figura si se proporciona una ruta
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    plt.show()


def enhanced_heatmap(pivot_data, save_path=None, title="Comparación de Estrategias de CV"):
    """
    Crea un mapa de calor mejorado para comparar estrategias.
    
    Args:
        pivot_data: DataFrame con datos en formato pivote
        save_path: Ruta para guardar la figura
        title: Título del gráfico
    """
    set_plotting_style()
    
    # Crear un colormap personalizado
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    
    plt.figure(figsize=(12, 8), dpi=200)
    
    # Crear el heatmap
    ax = sns.heatmap(pivot_data, annot=True, cmap=cmap, 
                    fmt=".3f", vmin=0, vmax=1, 
                    linewidths=0.5, linecolor='#ffffff',
                    cbar_kws={'label': 'R² Score Medio', 
                             'shrink': 0.8})
    
    # Mejoras visuales
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Cambiar el estilo de las etiquetas
    for text in ax.texts:
        value = float(text.get_text())
        if value >= 0.8:
            text.set_weight('bold')
        if value >= 0.9:
            text.set_size(14)
        
        # Cambiar color del texto según el valor para mejorar legibilidad
        if value > 0.7:
            text.set_color('white')
        else:
            text.set_color('black')
    
    # Agregar bordes al gráfico
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(2)
    
    # Cambiar la orientación de las etiquetas del eje y
    plt.yticks(rotation=0)
    
    # Mejorar la barra de colores
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('R² Score Medio', fontsize=14, fontweight='bold', labelpad=15)
    
    # Guardar la figura si se proporciona una ruta
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    plt.show()


def custom_strategy_boxplot(combined_df, save_path=None):
    """
    Crea un boxplot personalizado para comparar estrategias de CV con estilo minimalista.
    
    Args:
        combined_df: DataFrame con datos en formato largo
        save_path: Ruta para guardar la figura
    """
    # Definir la paleta de colores
    palette = sns.color_palette("Set2")
    sns.set_palette(palette)
    
    # Crear el gráfico
    plt.figure(figsize=(12, 7), dpi=200)
    
    # Crear el boxplot para cada modelo usando la misma paleta
    sns.boxplot(x='Model', y='R2_Score', data=combined_df, 
                fill=False, showfliers=False, hue='CV_Strategy')
    
    # Agregar el stripplot para mostrar todos los puntos individuales
    sns.stripplot(x='Model', y='R2_Score', data=combined_df, 
                  jitter=True, s=10, marker="X", alpha=.3, dodge=True, 
                  hue='CV_Strategy', legend=False)
    
    # Personalizar el gráfico
    plt.xlabel('Modelo', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.ylim([0, 1])
    plt.grid(axis='y')
    
    # Remover las leyendas duplicadas
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels[:3], handles[:3]))
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Calcular las medias para anotación opcional
    mean_scores = combined_df.groupby(['CV_Strategy', 'Model'])['R2_Score'].mean().reset_index()
    
    # Añadir líneas de referencia en valores clave
    plt.axhline(y=0.5, color='#999999', linestyle='--', alpha=0.5)
    plt.axhline(y=0.7, color='#999999', linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='#999999', linestyle='--', alpha=0.5)
    
    # Leyenda personalizada
    plt.legend(title='Estrategia de CV', loc='upper right', frameon=True, 
               fancybox=True, shadow=True)
    
    # Guardar la figura si se proporciona una ruta
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    plt.show()
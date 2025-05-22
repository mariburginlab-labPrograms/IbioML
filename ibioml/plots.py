#%% Importación de librerías necesarias
from IPython.display import display, HTML
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json

models = np.array(['mlp', 'rnn', 'gru', 'lstm'])
palette = sns.color_palette("tab10", len(models))
color_dict = {modelo: color for modelo, color in zip(models, palette)}
medians = None

#%% Carga de los resultados desde los archivos JSON
def load_results(paths: dict[str, str]) -> dict[str, dict]:
    results = {}
    for model, path in paths.items():
        try:
            with open(path) as f:
                # Cargar el contenido del archivo JSON
                results[model] = json.load(f)
        except Exception as e:
            print(f"Error al cargar {path}: {e}")
            continue
        
    return results

#%%
# Se crea un diccionario con los scores R2 de cada modelo
def extract_r2_scores(results: dict[str, dict]) -> tuple[dict, pd.DataFrame]:
    # Revisar si existen los campos para ambos targets
    has_both_targets = any('test_r2_scores_vel' in result for result in results.values())

    if has_both_targets:
        test_r2_scores = {}
        test_r2_scores_pos = {}
        test_r2_scores_vel = {}

        for model, content in results.items():
            test_r2_scores[model] = content['test_r2_scores']
            test_r2_scores_pos[model] = content['test_r2_scores_pos']
            test_r2_scores_vel[model] = content['test_r2_scores_vel']

        # Convert to dataframes
        r2_df = pd.DataFrame.from_dict(test_r2_scores, orient='index').T
        r2_pos_df = pd.DataFrame.from_dict(test_r2_scores_pos, orient='index').T
        r2_vel_df = pd.DataFrame.from_dict(test_r2_scores_vel, orient='index').T

        # Melt dataframes for visualization
        r2_test_df = r2_df.melt(var_name='Modelo', value_name='R2 Score')
        r2_test_df['Modelo'] = r2_test_df['Modelo'].str.upper()
        r2_test_df['Target'] = 'Combined'

        r2_pos_test_df = r2_pos_df.melt(var_name='Modelo', value_name='R2 Score')
        r2_pos_test_df['Modelo'] = r2_pos_test_df['Modelo'].str.upper()
        r2_pos_test_df['Target'] = 'Position'

        r2_vel_test_df = r2_vel_df.melt(var_name='Modelo', value_name='R2 Score')
        r2_vel_test_df['Modelo'] = r2_vel_test_df['Modelo'].str.upper()
        r2_vel_test_df['Target'] = 'Velocity'

        # Combine all dataframes
        r2_all_targets_df = pd.concat([r2_test_df, r2_pos_test_df, r2_vel_test_df])
        return test_r2_scores, test_r2_scores_pos, test_r2_scores_vel, r2_all_targets_df

    else:
        test_r2_scores = {}
        for model, content in results.items():
            test_r2_scores[model] = content['test_r2_scores']

        r2_df = pd.DataFrame.from_dict(test_r2_scores, orient='index').T
        r2_test_df = r2_df.melt(var_name='Modelo', value_name='R2 Score')
        r2_test_df['Modelo'] = r2_test_df['Modelo'].str.upper()

        return test_r2_scores, r2_test_df

def boxplot_test_r2_scores(r2_test_df: pd.DataFrame, save_path: str = None, y_lim: list[float] = [0, 1]):
    global palette
    global color_dict
    global medians
    
    # Crear el gráfico
    plt.figure(figsize=(10, 6), dpi=200)

    # Crear el boxplot para cada modelo sin hue para evitar duplicar la leyenda
    ax = sns.boxplot(x='Modelo', y='R2 Score', data=r2_test_df, fill=False, showfliers=False, hue='Modelo')

    # Agregar el stripplot para mostrar todos los puntos individuales
    sns.stripplot(x='Modelo', y='R2 Score', data=r2_test_df, 
                jitter=False, s=20, marker="X", alpha=.2, hue='Modelo')

    plt.xlabel('Modelo')
    plt.ylabel('R2 Score')
    plt.ylim(y_lim)  # Ajustar el rango del eje Y si es necesario
    plt.grid(axis='y')

    # Remover las leyendas duplicadas
    handles, labels = plt.gca().get_legend_handles_labels()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Calcular las medianas
    medians = r2_test_df.groupby(['Modelo'])['R2 Score'].median()
    ordered_medians = [medians[model] for model in r2_test_df['Modelo'].unique()]

    # Crear entradas personalizadas para la leyenda
    custom_lines = [plt.Line2D([0], [0], color=palette[i], lw=4) for i in range(len(r2_test_df['Modelo'].unique()))]
    legend_labels = [f'{model.upper()}: {median:.2f}' for model, median in zip(r2_test_df['Modelo'].unique(), ordered_medians)]

    # Agregar la leyenda personalizada
    ax.legend(custom_lines, legend_labels, title="R2 Score Mediana", bbox_to_anchor=(1, 1))

    # Opcional: guardar la figura
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def boxplot_test_r2_scores_both_targets(r2_all_targets_df: pd.DataFrame, save_path: str = None, y_lim: list[float] = [0, 1]):
    global palette
    global color_dict
    global medians
    
    # Boxplot for all targets
    plt.figure(figsize=(10, 6), dpi=200)

    # Create the boxplot for all targets
    sns.boxplot(x='Modelo', y='R2 Score', data=r2_all_targets_df, fill=False, showfliers=False, hue='Target')

    # Add the stripplot to show all individual points
    sns.stripplot(x='Modelo', y='R2 Score', data=r2_all_targets_df, 
                jitter=False, s=16, marker="X", alpha=.4, dodge=True, hue='Target')

    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('R2 Score', fontsize=12)
    plt.ylim([0, 1])  # Adjust the Y-axis range if necessary
    plt.grid(axis='y')

    # Remove duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Calculate medians
    medians = r2_all_targets_df.groupby(['Modelo', 'Target'])['R2 Score'].median()
    ordered_medians = [
        medians[model][target] for model in r2_all_targets_df['Modelo'].unique() 
        for target in r2_all_targets_df['Target'].unique()
    ]

    cant_models = len(r2_all_targets_df['Target'].unique())
    palette = sns.color_palette("Set2", cant_models)  # <-- Agrega esta línea
    # Create custom legend entries
    custom_lines = [plt.Line2D([0], [0], color=palette[i], lw=cant_models) for i in range(cant_models)]
    legend_labels = [f'{target}: {median:.2f}' for target, median in zip(r2_all_targets_df['Target'].unique(), ordered_medians)]

    # Add the custom legend
    ax.legend(custom_lines, legend_labels, title="R2 Score Mediana", bbox_to_anchor=(1, 1))
    # Optional: save the figure
    if save_path is not None:
        plt.savefig(save_path.replace('.png', '_boxplot.png'))
    plt.show()

    # Create a heatmap of mean scores for all targets
    plt.figure(figsize=(10, 6), dpi=200)

    # Calculate mean scores
    mean_scores = r2_all_targets_df.groupby(['Modelo', 'Target'])['R2 Score'].mean().reset_index()
    pivot_means = mean_scores.pivot(index='Modelo', columns='Target', values='R2 Score')
    # Define color palette
    palette = sns.color_palette("Set2", n_colors=len(r2_all_targets_df['Target'].unique()))
    sns.set_palette(palette)

    # Create heatmap
    sns.heatmap(pivot_means, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0, vmax=1)
    plt.title('Mean R2 Scores by Model and Target')
    plt.xlabel('Target', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path.replace('.png', '_heatmap.png'))
    
    plt.show()
    
# Extraer las predicciones y los valores verdaderos de cada modelo
def extract_predictions(results: dict[str, dict]) -> tuple[dict, dict]:
    predictions = {}
    true_values = {}

    for model in results.keys():
        # Extraer las predicciones y los valores verdaderos
        predictions[model.upper()] = results[model]['predictions_per_fold']
        true_values[model.upper()] = results[model]['true_values_per_fold']

    return predictions, true_values

def get_fold_closest_to_median(test_r2_scores, model):
    # Obtener los R2 scores y encontrar el fold más cercano a la mediana
    global medians
    r2_scores = test_r2_scores[model]
    median_r2 = medians[model.upper()]
    closest_fold = np.argmin(np.abs(np.array(r2_scores) - median_r2))
    return closest_fold

def plot_predictions(predictions, true_values, test_r2_scores, fold_to_plot=None, closest_to='median', save_path=None, limit=(None, None)):
    global palette
    global color_dict
    
    # Crear subplots para cada modelo
    plt.figure(figsize=(10, 15), dpi=200)  # Ajusta el tamaño según necesites

    for idx, model in enumerate(test_r2_scores.keys()):
        if fold_to_plot is None:
            # Obtener el fold más cercano a la mediana
            if closest_to == 'median':
                fold_to_plot = get_fold_closest_to_median(test_r2_scores, model)
            elif closest_to == 'mean':
                fold_to_plot = np.argmax(np.array(test_r2_scores[model]) == np.mean(test_r2_scores[model]))
        
        # Obtener datos de test y predicciones para el fold más cercano a la mediana
        y_test = true_values[model.upper()][fold_to_plot][limit[0]:limit[1]]  
        y_pred = predictions[model.upper()][fold_to_plot][limit[0]:limit[1]] 
        
        # Crear el subplot
        plt.subplot(4, 1, idx + 1)  # 4 filas, 1 columna, posición idx+1
        
        if limit == (None, None):
            # Crear el line plot
            plt.plot(y_test, label='Data', color='black', linewidth=2)
            plt.plot(y_pred, label='Predictions', alpha=1, linewidth=2, color=palette[idx])
        else:
            # Crear el line plot
            plt.plot(np.arange(limit[0], limit[1]), y_test, label='Data', color='black', linewidth=2)
            plt.plot(np.arange(limit[0], limit[1]), y_pred, label='Predictions', alpha=1, linewidth=2, color=palette[idx])

        plt.xlabel('Time (ms)', fontsize=14)
        plt.ylabel('Centered Position (cm)', fontsize=14)
        plt.title(f'Representative Fold for {model.upper()}', fontsize=14)
        plt.legend()

    plt.tight_layout()  # Ajusta automáticamente el espaciado entre subplots
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

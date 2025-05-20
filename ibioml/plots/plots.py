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
def load_results(mlp_path, rnn_path, gru_path, lstm_path):
    with open(mlp_path) as f:
        mlp_results = json.load(f)
        
    with open(rnn_path) as f:
        rnn_results = json.load(f)
        
    with open(gru_path) as f:
        gru_results = json.load(f)
        
    with open(lstm_path) as f:
        lstm_results = json.load(f)
        
    return mlp_results, rnn_results, gru_results, lstm_results

#%%
# Se crea un diccionario con los scores R2 de cada modelo
def extract_r2_scores(mlp_results, rnn_results, gru_results, lstm_results):
    test_r2_scores = {
        'MLP': mlp_results['test_r2_scores'],
        'RNN': rnn_results['test_r2_scores'],
        'GRU': gru_results['test_r2_scores'],
        'LSTM': lstm_results['test_r2_scores'],
    }
    # Conversión del diccionario a DataFrame
    r2_df = pd.DataFrame.from_dict(test_r2_scores, orient='index').T
    
    # Reorganización del DataFrame para formato largo (mejor para visualización)
    r2_test_df = r2_df.melt(var_name='Modelo', value_name='R2 Score')
    
    # Conversión de los nombres de modelo a mayúsculas
    r2_test_df['Modelo'] = r2_test_df['Modelo'].str.upper()
    
    return test_r2_scores, r2_test_df

def boxplot_test_r2_scores(r2_test_df, save_path=None, y_lim=[0, 1]):
    global palette
    global color_dict
    global medians
    global models
    
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
    custom_lines = [plt.Line2D([0], [0], color=palette[i], lw=4) for i in range(len(models))]
    legend_labels = [f'{model.upper()}: {median:.2f}' for model, median in zip(r2_test_df['Modelo'].unique(), ordered_medians)]

    # Agregar la leyenda personalizada
    ax.legend(custom_lines, legend_labels, title="R2 Score Mediana", bbox_to_anchor=(1, 1))

    # Opcional: guardar la figura
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    
# Extraer las predicciones y los valores verdaderos de cada modelo
def extract_predictions(mlp_results, rnn_results, gru_results, lstm_results):
    predictions = {
        'MLP': mlp_results['predictions_per_fold'],
        'RNN': rnn_results['predictions_per_fold'],
        'GRU': gru_results['predictions_per_fold'],
        'LSTM': lstm_results['predictions_per_fold'],
    }

    true_values = {
        'MLP': mlp_results['true_values_per_fold'],
        'RNN': rnn_results['true_values_per_fold'],
        'GRU': gru_results['true_values_per_fold'],
        'LSTM': lstm_results['true_values_per_fold'],
    }
    
    return predictions, true_values

def get_fold_closest_to_median(test_r2_scores, model):
    # Obtener los R2 scores y encontrar el fold más cercano a la mediana
    global medians
    r2_scores = test_r2_scores[model]
    median_r2 = medians[model.upper()]
    closest_fold = np.argmin(np.abs(np.array(r2_scores) - median_r2))
    return closest_fold

def plot_predictions(predictions, true_values, test_r2_scores, fold_to_plot=None, closest_to='median', save_path=None, limit=None):
    global palette
    global color_dict
    
    # Crear subplots para cada modelo
    plt.figure(figsize=(10, 15), dpi=200)  # Ajusta el tamaño según necesites

    for idx, model in enumerate(models):
        if fold_to_plot is None:
            # Obtener el fold más cercano a la mediana
            if closest_to == 'median':
                fold_to_plot = get_fold_closest_to_median(test_r2_scores, model)
            elif closest_to == 'mean':
                fold_to_plot = np.argmax(np.array(test_r2_scores[model]) == np.mean(test_r2_scores[model]))
        
        # Obtener datos de test y predicciones para el fold más cercano a la mediana
        y_test = true_values[model.upper()][fold_to_plot][:limit]  # Limitar a los primeros 1000 datos
        y_pred = predictions[model.upper()][fold_to_plot][:limit]  # Limitar a los primeros 1000 datos
        
        # Crear el subplot
        plt.subplot(4, 1, idx + 1)  # 4 filas, 1 columna, posición idx+1
        
        # Crear el line plot
        plt.plot(y_test, label='Data', color='black', linewidth=2)
        plt.plot(y_pred, label='Predictions', alpha=1, linewidth=2, color=palette[idx])

        plt.xlabel('Time (ms)', fontsize=14)
        plt.ylabel('Centered Position (cm)', fontsize=14)
        plt.title(f'Representative Fold for {model.upper()}', fontsize=14)
        plt.legend()

    plt.tight_layout()  # Ajusta automáticamente el espaciado entre subplots
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

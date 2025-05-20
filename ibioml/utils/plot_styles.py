import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

def set_plotting_style():
    """Configura un estilo visual mejorado para todas las visualizaciones"""
    # Configuración de estilo base
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Paletas personalizadas
    palette = sns.color_palette("tab10", 4)
    sns.set_palette(palette)
    
    # Tamaños y estilos de fuente
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    
    # Estilo de las líneas y gráficos
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 10
    
    # Estilo de los ejes
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

# Paletas de color personalizadas
def get_model_colors(model_count=4):
    """Genera un diccionario de colores para modelos"""
    # Crea una paleta personalizada vibrante pero profesional
    colors = sns.color_palette("tab10", model_count)
    models = ['MLP', 'RNN', 'GRU', 'LSTM'][:model_count]
    return dict(zip(models, colors))

def get_strategy_colors():
    """Genera un diccionario de colores para estrategias de CV"""
    strategies = ['No Shuffle', 'Time Bin Shuffle', 'Trial Shuffle']
    colors = sns.color_palette("mako", len(strategies))
    return dict(zip(strategies, colors))
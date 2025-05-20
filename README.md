# IbioML

Toolkit de Machine Learning para experimentos de neurodecodificación en IBIoBA.

## Instalación

Clona este repositorio y asegúrate de tener un entorno con Python 3.8+ y las dependencias instaladas (puedes usar `requirements.txt` si está disponible):

```bash
git clone https://github.com/tuusuario/IbioML.git
cd IbioML
```

## Preprocesamiento de datos `.mat`

Para convertir archivos `.mat` (de MATLAB) a los formatos de entrada requeridos por los modelos, utiliza el script de preprocesamiento:

```python
from ibioml.preprocess_data import preprocess_data

preprocess_data(
    file_path='datasets/tu_archivo.mat', 
    file_name_to_save='nombre_salida', 
    bins_before=5, 
    bins_after=5, 
    bins_current=1, 
    threshDPrime=2.5, 
    firingMinimo=1000
)
```

Esto generará archivos `.pickle` en la carpeta `data/` listos para usar en los experimentos.

## Cómo correr un experimento

1. **Prepara tus datos** (ver sección anterior).
2. Abre el notebook de ejemplo:  
   `examples/simple_study.ipynb`
3. Ajusta la ruta de los datos si es necesario.
4. Ejecuta las celdas para definir el espacio de modelos y correr el estudio:

```python
from ibioml.models import MLPModel
from ibioml.utils.model_factory import create_model_class
from ibioml.tuner import run_study

# Carga tus datos preprocesados
import pickle
with open('data/bins200ms_preprocessed_withCtxt_flat.pickle', 'rb') as f:
    X_flat, y_flat, T = pickle.load(f)

# Define el espacio de hiperparámetros
mlp_base_space = {
    "model_class": create_model_class(MLPModel, y_flat.shape[1]),
    "output_size": 1,
    "device": "cuda",  # o "cpu"
    "num_epochs": 200,
    "batch_size": 32,
}

# Corre el experimento
run_study(
    X_flat, y_flat, T,
    model_space=mlp_base_space,
    num_trials=2,
    outer_folds=5,
    inner_folds=1,
    save_path="results/mlp_nested_cv"
)
```

Esto ejecutará una validación cruzada anidada con 2 configuraciones probadas por split usando Bayes Optimization para MLP y guardará los resultados en la carpeta `results/mlp_nested_cv` bajo el nombre `study_YYYY-MM-DD_HH-MM-SS`, donde `YYYY-MM-DD_HH-MM-SS` corresponde a la fecha y hora en la que se corre el experimento.

## Visualización de resultados

Para visualizar los resultados y comparar modelos, puedes usar el notebook `examples/simple_plots.ipynb`. Ejemplo de uso:

```python
from ibioml.plots import *

# Carga los resultados
mlp_results = load_results({'mlp': 'results/mlp_nested_cv/study_YYYY-MM-DD_HH-MM-SS/final_results.json'})

# Extrae los scores R2
test_r2_scores, test_r2_scores_pos, test_r2_scores_vel, r2_test_df = extract_r2_scores(mlp_results)

# Grafica boxplots de los scores R2 para cada target
boxplot_test_r2_scores_both_targets(r2_test_df)
```

También puedes graficar predicciones y otros análisis usando las funciones del módulo `ibioml.plots`.

---

## Estructura del repositorio

- `ibioml/` - Código fuente principal (modelos, entrenamiento, preprocesamiento, visualización)
- `examples/` - Notebooks de ejemplo para experimentos y visualización
- `data/` - Datos preprocesados
- `results/` - Resultados de los experimentos

---

## Contacto

Puedes escribir a [jiponce@ibioba-mpsp-conicet.gov.ar](mailto:jiponce@ibioba-mpsp-conicet.gov.ar) para dudas o sugerencias.

# IbioML

Toolkit de Machine Learning para experimentos de neurodecodificación en IBioBA.

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

> **Nota:** El parámetro `file_name_to_save` debe incluir la ruta completa donde deseas guardar los datos preprocesados. Como buena práctica, se recomienda organizar los archivos en subcarpetas que indiquen la configuración utilizada, por ejemplo:
>
> - Carpeta para la historia de bins: `5_5_1` (correspondiente a `bins_before=5`, `bins_after=5`, `bins_current=1`)
> - Carpeta para el threshold: `2_5` (por ejemplo, `threshDPrime=2.5`)
> - Carpeta para la precisión temporal: `bins200ms` (o el valor correspondiente)
>
> Ejemplo de uso:
>
> ```python
> preprocess_data(
>     file_path='datasets/tu_archivo.mat',
>     file_name_to_save='data/bins200ms/5_5_1/2_5/nombre_salida',
>     bins_before=5,
>     bins_after=5,
>     bins_current=1,
>     threshDPrime=2.5,
>     firingMinimo=1000
> )
> ```
>
> Así, los archivos generados quedarán organizados y será más sencillo identificar la configuración utilizada en cada experimento.

### Archivos generados por el preprocesamiento

El preprocesamiento genera un total de 12 archivos `.pickle` en la carpeta `data/`, divididos en dos grupos:

- **Con contexto** (`withCtxt`): 6 archivos
- **Sin contexto** (`noCtxt`): 6 archivos

Cada grupo contiene 3 tipos de archivos, y cada tipo tiene dos variantes: `flat` y `no flat`:

1. **onlyPosition**: Solo incluye información de posición.
2. **onlyVelocity**: Solo incluye información de velocidad.
3. **bothTargets**: Incluye tanto posición como velocidad.

Por cada tipo, se generan dos archivos:
- Variante `flat`: Los datos de entrada están aplanados.
- Variante `no flat`: Los datos mantienen su estructura original.

Ejemplo de nombres de archivos generados:
- `preprocessed_withCtxt_onlyPosition_flat.pickle`
- `preprocessed_withCtxt_onlyPosition.pickle`
- `preprocessed_noCtxt_bothTargets_flat.pickle`
- `preprocessed_noCtxt_onlyVelocity.pickle`

Estos archivos están listos para ser usados en los experimentos de modelado.

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

## Selección de estructura de datos según el modelo

La estructura de los datos de entrada depende del tipo de modelo que desees experimentar:

- **Modelos no recurrentes (por ejemplo, MLP):**  
    Debes utilizar los archivos preprocesados con la variante `flat`, ya que estos modelos requieren que los datos estén aplanados.

- **Modelos recurrentes (por ejemplo, RNN, LSTM, GRU):**  
    Debes utilizar los archivos preprocesados en su variante original (`no flat`), que mantienen la estructura temporal necesaria para estos modelos.

Asegúrate de seleccionar la variante adecuada al cargar los datos para tus experimentos.

A continuación se muestra una imagen ilustrando la estructura de los datos para ambos casos:

**Figura 1:** Ejemplo de estructura de datos *aplanada* (`flat`) utilizada para modelos no recurrentes (MLP). Cada muestra corresponde a un vector que concatena la información de todos los bins de la ventana temporal.
![Estructura de datos para modelos no recurrentes](https://github.com/mariburginlab-labPrograms/IbioML/raw/main/docs/images/flat.jpg)

**Figura 2:** Ejemplo de estructura de datos *tensorial* (`no flat`) utilizada para modelos recurrentes (RNN, LSTM, GRU). Cada muestra mantiene la dimensión temporal, permitiendo que el modelo procese la secuencia completa de bins.
![Estructura de datos para modelos recurrentes](https://github.com/mariburginlab-labPrograms/IbioML/raw/main/docs/images/tensor.jpg)

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

# API Reference - Modelos

Esta sección documenta todos los modelos de machine learning disponibles en IbioML.

## Modelos Base

::: ibioml.models.BaseMLP
    options:
      show_root_heading: true
      show_source: true

::: ibioml.models.BaseRNN
    options:
      show_root_heading: true
      show_source: true

## Modelos de Neurodecodificación

### Perceptrón Multicapa (MLP)

::: ibioml.models.MLPModel
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

### Redes Neuronales Recurrentes

::: ibioml.models.RNNModel
    options:
      show_root_heading: true
      show_source: true

::: ibioml.models.LSTMModel
    options:
      show_root_heading: true
      show_source: true

::: ibioml.models.GRUModel
    options:
      show_root_heading: true
      show_source: true

## Modelos Dual-Output

### MLP Dual-Output

::: ibioml.models.DualOutputMLPModel
    options:
      show_root_heading: true
      show_source: true

### RNN Dual-Output

::: ibioml.models.DualOutputRNNModel
    options:
      show_root_heading: true
      show_source: true

::: ibioml.models.DualOutputLSTMModel
    options:
      show_root_heading: true
      show_source: true

::: ibioml.models.DualOutputGRUModel
    options:
      show_root_heading: true
      show_source: true

## Utilidades de Modelos

::: ibioml.utils.model_factory.create_model_class
    options:
      show_root_heading: true
      show_source: true

## Ejemplos de Uso

### Crear un MLP básico

```python
from ibioml.models import MLPModel
import torch

# Configuración del modelo
model = MLPModel(
    input_size=1000,     # Características de entrada
    hidden_size=256,     # Neuronas en capas ocultas
    output_size=1,       # Número de salidas
    num_layers=3,        # Número de capas ocultas
    dropout=0.2          # Tasa de dropout
)

# Datos de ejemplo
x = torch.randn(32, 1000)  # Batch de 32 muestras
output = model(x)
print(f"Forma de salida: {output.shape}")  # [32, 1]
```

### Crear un LSTM para datos temporales

```python
from ibioml.models import LSTMModel
import torch

# Configuración del modelo
model = LSTMModel(
    input_size=100,      # Características por timestep
    hidden_size=128,     # Tamaño del estado oculto
    output_size=2,       # Posición + velocidad
    num_layers=2,        # Capas LSTM
    dropout=0.3
)

# Datos de ejemplo (batch, sequence, features)
x = torch.randn(16, 10, 100)  # 16 muestras, 10 timesteps, 100 features
output = model(x)
print(f"Forma de salida: {output.shape}")  # [16, 2]
```

### Usar la factory para crear modelos

```python
from ibioml.utils.model_factory import create_model_class
from ibioml.models import MLPModel

# Crear clase de modelo con configuración específica
ModelClass = create_model_class(MLPModel, output_size=1)

# Instanciar modelo
model = ModelClass(
    input_size=500,
    hidden_size=256,
    num_layers=2,
    dropout=0.1
)
```

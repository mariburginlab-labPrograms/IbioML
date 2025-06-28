# Instalación

## Requisitos del Sistema

- **Python:** 3.8 o superior
- **Sistema Operativo:** Windows, macOS, Linux
- **GPU:** Opcional (CUDA compatible para aceleración)

## Métodos de Instalación

### 🎯 Instalación Recomendada (pip)

```bash
pip install ibioml
```

### 🔧 Instalación desde Código Fuente

Para obtener la versión más reciente con las últimas características:

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/IbioML.git
cd IbioML

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar en modo desarrollo
pip install -e .
```

### 🐍 Instalación con Conda

```bash
# Crear entorno conda
conda create -n ibioml python=3.9
conda activate ibioml

# Instalar dependencias principales
conda install numpy pandas scikit-learn matplotlib seaborn scipy

# Instalar PyTorch (ajustar según tu sistema)
conda install pytorch torchvision torchaudio -c pytorch

# Instalar IbioML
pip install ibioml
```

## Dependencias

IbioML requiere las siguientes librerías:

### Dependencias Principales

| Librería | Versión | Propósito |
|----------|---------|-----------|
| `numpy` | >=1.19.0 | Operaciones numéricas |
| `pandas` | >=1.3.0 | Manipulación de datos |
| `scikit-learn` | >=1.0.0 | ML utilities y métricas |
| `torch` | >=1.9.0 | Deep learning framework |
| `matplotlib` | >=3.3.0 | Visualización básica |
| `seaborn` | >=0.11.0 | Visualización estadística |
| `scipy` | >=1.7.0 | Operaciones científicas |

### Dependencias Opcionales

Para funcionalidades avanzadas:

```bash
# Para optimización de hiperparámetros
pip install optuna

# Para documentación interactiva
pip install jupyter ipywidgets

# Para análisis estadísticos avanzados
pip install statsmodels
```

## Configuración del Entorno

### Variables de Entorno

Para un rendimiento óptimo, configura estas variables:

```bash
# Para usar GPU (si está disponible)
export CUDA_VISIBLE_DEVICES=0

# Para reproducibilidad
export PYTHONHASHSEED=42
```

### Configuración de GPU

Para verificar que PyTorch detecta tu GPU:

```python
import torch

print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivos GPU: {torch.cuda.device_count()}")
    print(f"GPU actual: {torch.cuda.get_device_name()}")
```

## Verificación de la Instalación

Ejecuta este script para verificar que todo funciona correctamente:

```python
import sys
import ibioml

print("✅ IbioML instalado correctamente!")
print(f"Versión: {ibioml.__version__}")
print(f"Python: {sys.version}")

# Verificar módulos principales
try:
    from ibioml.models import MLPModel
    from ibioml.preprocessing import preprocess_data
    from ibioml.tuner import run_study
    print("✅ Todos los módulos principales disponibles")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")

# Verificar dependencias
try:
    import torch
    import numpy as np
    import pandas as pd
    import sklearn
    print("✅ Todas las dependencias disponibles")
except ImportError as e:
    print(f"❌ Faltan dependencias: {e}")
```

## Configuración Avanzada

### Para Desarrollo

Si planeas contribuir al desarrollo de IbioML:

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# O manualmente:
pip install pytest black flake8 mypy pre-commit
```

### Para Servidores/HPC

En clusters de computación o servidores:

```bash
# Instalación sin dependencias de visualización
pip install ibioml --no-deps
pip install numpy pandas scikit-learn torch scipy

# Para ambientes sin internet
pip download ibioml -d ./packages
pip install ./packages/*.whl --no-index --find-links ./packages
```

## Problemas Comunes

### Error: "No module named 'ibioml'"

```bash
# Verificar instalación
pip list | grep ibioml

# Reinstalar si es necesario
pip uninstall ibioml
pip install ibioml
```

### Error: "CUDA out of memory"

```python
# Reducir batch_size en la configuración
config = {
    "batch_size": 16,  # En lugar de 32 o 64
    # ... resto de configuración
}
```

### Error: "Permission denied"

```bash
# Instalar solo para el usuario actual
pip install --user ibioml
```

## Próximos Pasos

Una vez instalado IbioML exitosamente:

1. 📚 Lee la [guía de preprocesamiento](preprocessing.md)
2. 🚀 Ejecuta tu [primer experimento](experiments.md)
3. 📊 Explora las opciones de [visualización](visualization.md)
4. 📖 Consulta la [API reference](api/models.md) para uso avanzado

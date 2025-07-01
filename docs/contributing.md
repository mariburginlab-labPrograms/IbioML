# Contribuir a IBioML

¡Gracias por tu interés en contribuir a IBioML! Este proyecto se beneficia enormemente de las contribuciones de la comunidad.

## 🚀 Formas de Contribuir

### 🐛 Reportar Bugs
- Usa el [sistema de issues](https://github.com/tuusuario/IBioML/issues) de GitHub
- Incluye información detallada sobre el error
- Proporciona un ejemplo mínimo reproducible

### 💡 Sugerir Nuevas Características
- Abre un issue describiendo la característica
- Explica el caso de uso y beneficios
- Discute la implementación propuesta

### 📝 Mejorar Documentación
- Corregir errores tipográficos
- Agregar ejemplos o aclaraciones
- Traducir contenido

### 🔧 Contribuir Código
- Implementar nuevas características
- Corregir bugs
- Mejorar rendimiento
- Agregar tests

## 🛠️ Configuración del Entorno de Desarrollo

### 1. Fork y Clone

```bash
# Fork el repositorio en GitHub, luego:
git clone https://github.com/tu-usuario/IBioML.git
cd IBioML
```

### 2. Configurar Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar en modo desarrollo
pip install -e ".[dev]"
```

### 3. Instalar Dependencias de Desarrollo

```bash
pip install pytest black flake8 mypy pre-commit mkdocs-material
```

### 4. Configurar Pre-commit Hooks

```bash
pre-commit install
```

## 📋 Guías de Contribución

### Estilo de Código

IBioML sigue las convenciones de Python (PEP 8) con algunas extensiones:

```python
# Usar type hints
def preprocess_data(
    file_path: str,
    file_name_to_save: str,
    bins_before: int = 5,
    bins_after: int = 5
) -> None:
    """
    Preprocesa datos neuronales.
    
    Args:
        file_path: Ruta al archivo .mat
        file_name_to_save: Nombre base para archivos de salida
        bins_before: Ventana temporal hacia atrás
        bins_after: Ventana temporal hacia adelante
    """
    pass

# Nombres descriptivos
def calculate_r2_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calcula el coeficiente de determinación R²."""
    pass

# Documentación clara en español para funciones públicas
class MLPModel(nn.Module):
    """
    Perceptrón multicapa para neurodecodificación.
    
    Esta clase implementa una red neuronal feedforward con múltiples
    capas ocultas y dropout para regularización.
    """
    pass
```

### Formateo Automático

```bash
# Formatear código con black
black ibioml/

# Verificar estilo con flake8
flake8 ibioml/

# Verificar tipos con mypy
mypy ibioml/
```

### Estructura de Commits

Usa el formato de [Conventional Commits](https://www.conventionalcommits.org/):

```
tipo(ámbito): descripción breve

Descripción más detallada si es necesario.

Fixes #123
```

Tipos principales:
- `feat`: Nueva característica
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Cambios de formato (sin afectar lógica)
- `refactor`: Refactorización de código
- `test`: Agregar o modificar tests
- `chore`: Tareas de mantenimiento

Ejemplos:
```
feat(models): agregar soporte para modelos transformer

docs(preprocessing): mejorar ejemplos de uso

fix(tuner): corregir error en validación cruzada anidada
```

## 🧪 Testing

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_models.py

# Con cobertura
pytest --cov=ibioml
```

### Escribir Tests

```python
import pytest
import numpy as np
from ibioml.models import MLPModel

class TestMLPModel:
    """Tests para el modelo MLP."""
    
    def test_model_creation(self):
        """Test básico de creación del modelo."""
        model = MLPModel(
            input_size=100,
            hidden_size=50,
            output_size=1,
            num_layers=2,
            dropout=0.1
        )
        assert model.input_size == 100
        assert model.output_size == 1
    
    def test_forward_pass(self):
        """Test del forward pass."""
        model = MLPModel(100, 50, 1, 2, 0.1)
        x = torch.randn(10, 100)
        output = model(x)
        assert output.shape == (10, 1)
    
    @pytest.mark.parametrize("batch_size", [1, 16, 32])
    def test_different_batch_sizes(self, batch_size):
        """Test con diferentes tamaños de lote."""
        model = MLPModel(100, 50, 1, 2, 0.1)
        x = torch.randn(batch_size, 100)
        output = model(x)
        assert output.shape == (batch_size, 1)
```

## 📚 Documentación

### Estructura de Documentación

```
docs/
├── index.md              # Página principal
├── installation.md       # Guía de instalación
├── preprocessing.md       # Preprocesamiento
├── experiments.md         # Configuración de experimentos
├── visualization.md       # Visualización
├── api/                   # Documentación de API
│   ├── models.md
│   ├── preprocessing.md
│   └── training.md
└── examples/              # Tutoriales y ejemplos
    ├── basic_tutorial.md
    └── advanced_usage.md
```

### Escribir Documentación

```markdown
# Título de la Sección

Descripción breve y clara de qué hace esta funcionalidad.

## Uso Básico

```python
# Ejemplo de código simple
from ibioml.models import MLPModel

model = MLPModel(input_size=100, hidden_size=50, output_size=1)
```

## Parámetros

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `input_size` | int | Número de características de entrada |
| `hidden_size` | int | Neuronas en capas ocultas |

!!! tip "Recomendación"
    Para mejores resultados, usa `hidden_size` entre 64 y 512.

!!! warning "Advertencia"
    Valores muy altos de `dropout` pueden degradar el rendimiento.
```

### Generar Documentación Localmente

```bash
# Instalar dependencias
pip install mkdocs-material mkdocstrings[python]

# Servir documentación localmente
mkdocs serve

# Compilar documentación
mkdocs build
```

## 🔄 Proceso de Pull Request

### 1. Preparar el PR

```bash
# Crear rama para tu característica
git checkout -b feat/nueva-caracteristica

# Hacer cambios y commits
git add .
git commit -m "feat(models): agregar modelo transformer"

# Push a tu fork
git push origin feat/nueva-caracteristica
```

### 2. Crear Pull Request

1. Ve a GitHub y crea un PR desde tu rama
2. Usa la plantilla de PR (si existe)
3. Describe claramente los cambios
4. Relaciona con issues relevantes

### 3. Plantilla de PR

```markdown
## Descripción

Descripción breve de los cambios realizados.

## Tipo de cambio

- [ ] Bug fix (cambio que corrige un issue)
- [ ] Nueva característica (cambio que agrega funcionalidad)
- [ ] Breaking change (cambio que rompe compatibilidad)
- [ ] Documentación

## Testing

- [ ] Tests existentes pasan
- [ ] Agregué tests para nuevos cambios
- [ ] Tests cubren casos edge

## Checklist

- [ ] Mi código sigue el estilo del proyecto
- [ ] Agregué documentación para nuevas características
- [ ] Los tests pasan localmente
- [ ] Actualicé CHANGELOG.md (si aplica)

## Issues relacionados

Fixes #123
```

### 4. Revisión de Código

- Responde constructivamente a los comentarios
- Haz los cambios solicitados
- Mantén la discusión profesional y enfocada

## 🏗️ Arquitectura del Proyecto

### Estructura de Carpetas

```
ibioml/
├── __init__.py
├── models.py              # Modelos de ML
├── preprocessing.py       # Preprocesamiento de datos
├── trainer.py            # Lógica de entrenamiento
├── tuner.py              # Optimización de hiperparámetros
├── plots.py              # Funciones de visualización
├── utils/                # Utilidades
│   ├── __init__.py
│   ├── data_scaler.py
│   ├── evaluators.py
│   ├── model_factory.py
│   └── preprocessing_funcs.py
└── results/              # Gestión de resultados (nueva)
    ├── __init__.py
    ├── experiment_results.py
    └── visualizer.py
```

### Principios de Diseño

1. **Modularidad**: Cada módulo tiene una responsabilidad clara
2. **Extensibilidad**: Fácil agregar nuevos modelos y funcionalidades
3. **Usabilidad**: API simple e intuitiva
4. **Robustez**: Manejo de errores y validación de entrada
5. **Rendimiento**: Optimizado para datasets grandes

## 🎯 Áreas que Necesitan Contribuciones

### Alta Prioridad

- [ ] Soporte para modelos Transformer
- [ ] Mejoras en visualización interactiva
- [ ] Optimización de memoria para datasets grandes
- [ ] Integración con MLflow/Weights & Biases
- [ ] Tests adicionales (especialmente integration tests)

### Media Prioridad

- [ ] Soporte para más formatos de datos (HDF5, Parquet)
- [ ] Análisis estadísticos avanzados
- [ ] Exportación de modelos a ONNX
- [ ] Paralelización de experimentos
- [ ] Documentación en inglés

### Baja Prioridad

- [ ] Interfaz gráfica web
- [ ] Soporte para modelos probabilísticos
- [ ] Integración con cloud providers
- [ ] Mobile/edge deployment

## 🤝 Comunicación

### Canales de Comunicación

- **Issues de GitHub**: Para bugs y feature requests
- **Discussions**: Para preguntas generales y discusión
- **Email**: [jiponce@ibioba-mpsp-conicet.gov.ar](mailto:jiponce@ibioba-mpsp-conicet.gov.ar)

### Código de Conducta

- Sé respetuoso y constructivo
- Acepta feedback de manera positiva
- Ayuda a otros contribuidores
- Mantén discusiones técnicas enfocadas

## 📜 Licencia

Al contribuir a IBioML, aceptas que tus contribuciones sean licenciadas bajo la misma licencia MIT del proyecto.

---

¡Gracias por contribuir a IBioML! 🚀

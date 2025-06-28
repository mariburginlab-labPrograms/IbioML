# Documentación de IbioML

Esta carpeta contiene toda la documentación para el proyecto IbioML, construida con [MkDocs](https://www.mkdocs.org/) y el tema [Material](https://squidfunk.github.io/mkdocs-material/).

## 🚀 Inicio Rápido

### Configuración del entorno

```bash
# Desde la raíz del proyecto
./docs.sh setup
```

### Servir localmente

```bash
./docs.sh serve
```

La documentación estará disponible en http://localhost:8000

### Construir para producción

```bash
./docs.sh build
```

## 📁 Estructura de la documentación

```
docs/
├── index.md                    # Página principal
├── installation.md             # Guía de instalación
├── preprocessing.md             # Documentación de preprocesamiento
├── experiments.md               # Configuración de experimentos
├── visualization.md             # Guía de visualización
├── contributing.md              # Guía de contribución
├── api/                        # Documentación de API automática
│   ├── models.md               # Documentación de modelos
│   ├── preprocessing.md        # API de preprocesamiento
│   ├── training.md             # API de entrenamiento
│   └── results.md              # API de resultados
├── examples/                   # Ejemplos y tutoriales
│   ├── basic_tutorial.md       # Tutorial básico
│   └── full_experiment.md      # Experimento completo
├── images/                     # Imágenes para la documentación
├── stylesheets/                # Estilos CSS personalizados
│   └── extra.css               # Estilos adicionales
├── DEVELOPMENT.md              # Guía para desarrolladores de docs
└── README.md                   # Este archivo
```

## 🛠️ Comandos Útiles

### Script de gestión

El script `docs.sh` en la raíz del proyecto proporciona comandos útiles:

```bash
./docs.sh setup     # Instalar dependencias
./docs.sh serve     # Servir localmente
./docs.sh build     # Construir documentación
./docs.sh deploy    # Desplegar a GitHub Pages
./docs.sh check     # Verificar construcción
./docs.sh clean     # Limpiar archivos temporales
./docs.sh status    # Mostrar estado del sistema
./docs.sh help      # Mostrar ayuda
```

### Comandos MkDocs directos

```bash
# Servir con auto-reload
mkdocs serve

# Construir sitio estático
mkdocs build

# Desplegar a GitHub Pages
mkdocs gh-deploy

# Verificar construcción estricta
mkdocs build --strict
```

## 📝 Escribiendo Documentación

### Docstrings en Python

Usa el estilo Google para los docstrings:

```python
def mi_funcion(param: str, otro_param: int = 10) -> bool:
    """Descripción breve de la función.
    
    Descripción más detallada de lo que hace la función,
    incluyendo casos de uso y consideraciones importantes.
    
    Args:
        param: Descripción del parámetro string.
        otro_param: Descripción del parámetro entero. Por defecto 10.
        
    Returns:
        True si la operación fue exitosa, False en caso contrario.
        
    Raises:
        ValueError: Si param está vacío.
        TypeError: Si param no es un string.
        
    Example:
        Ejemplo básico de uso:
        
        >>> mi_funcion("test", 5)
        True
        
        Ejemplo con valores por defecto:
        
        >>> mi_funcion("test")
        True
    """
    if not isinstance(param, str):
        raise TypeError("param debe ser un string")
    if not param:
        raise ValueError("param no puede estar vacío")
    return True
```

### Admoniciones (Cajas de advertencia)

```markdown
!!! note "Nota"
    Información adicional útil.

!!! tip "Consejo"
    Sugerencia para el usuario.

!!! warning "Advertencia"
    Algo importante a tener en cuenta.

!!! danger "Peligro"
    Información crítica de seguridad.

!!! example "Ejemplo"
    Ejemplo de código o uso.
```

### Bloques de código

```markdown
```python title="ejemplo.py"
import ibioml
from ibioml.models import MLPModel

# Código de ejemplo con título
model = MLPModel(input_size=100, hidden_size=64)
```

### Pestañas

```markdown
=== "Python"
    ```python
    import ibioml
    ```

=== "Bash"
    ```bash
    pip install ibioml
    ```
```

### Tablas

```markdown
| Columna 1 | Columna 2 | Columna 3 |
|-----------|-----------|-----------|
| Valor 1   | Valor 2   | Valor 3   |
| Valor 4   | Valor 5   | Valor 6   |
```

## 🔧 Configuración

### mkdocs.yml

El archivo de configuración principal está en la raíz del proyecto. Las secciones importantes:

- **nav**: Define la navegación del sitio
- **theme**: Configuración del tema Material
- **plugins**: Extensiones como mkdocstrings para API docs
- **markdown_extensions**: Funcionalidades adicionales de Markdown

### Personalización de estilos

Los estilos personalizados están en `docs/stylesheets/extra.css`. Puedes modificar:

- Colores del tema
- Estilos de código
- Apariencia de tablas y botones
- Animaciones

### Variables de configuración

```yaml
# En mkdocs.yml
extra:
  version:
    provider: mike
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/mariburginlab-labPrograms/IbioML
```

## 🚀 Despliegue

### GitHub Pages (Automático)

La documentación se despliega automáticamente via GitHub Actions cuando se hace push a la rama `main`. El workflow está en `.github/workflows/docs.yml`.

### Despliegue Manual

```bash
# Desde la raíz del proyecto
./docs.sh deploy
```

O directamente:

```bash
mkdocs gh-deploy
```

## 🐛 Troubleshooting

### Errores comunes

**Module not found al generar API docs:**
```bash
# Instalar el paquete en modo editable
pip install -e .
```

**Links rotos:**
```bash
# Verificar con construcción estricta
mkdocs build --strict
```

**Problemas con el tema:**
```bash
# Reinstalar dependencias
pip install --upgrade mkdocs-material mkdocstrings[python]
```

### Debugging

1. Verificar que el paquete esté instalado: `pip list | grep ibioml`
2. Probar importaciones: `python -c "import ibioml"`
3. Revisar logs de construcción: `mkdocs build --verbose`
4. Verificar configuración: `mkdocs config`

## 📚 Recursos

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material Theme](https://squidfunk.github.io/mkdocs-material/)
- [MkDocstrings](https://mkdocstrings.github.io/)
- [Python Markdown Extensions](https://python-markdown.github.io/extensions/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)

## 🤝 Contribuir

Para contribuir a la documentación:

1. **Fork** el repositorio
2. **Clona** tu fork localmente
3. **Crea** una rama para tus cambios: `git checkout -b docs/mejora-api`
4. **Hace** tus cambios en los archivos de documentación
5. **Prueba** localmente: `./docs.sh serve`
6. **Hace** commit y push: `git commit -m "Mejorar docs de API"`
7. **Abre** un Pull Request

### Checklist para contribuciones

- [ ] Los cambios se ven correctamente en el servidor local
- [ ] Los enlaces funcionan correctamente
- [ ] El código de ejemplo es válido y ejecutable
- [ ] Los docstrings siguen el estilo Google
- [ ] Se agregaron ejemplos cuando es apropiado
- [ ] La construcción estricta pasa: `./docs.sh check`

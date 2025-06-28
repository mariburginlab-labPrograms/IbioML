# Comandos útiles para desarrollar la documentación

## Instalar dependencias de documentación
pip install -r requirements-docs.txt
pip install -e .

## Servir la documentación localmente (con auto-reload)
mkdocs serve

## Construir la documentación estática
mkdocs build

## Desplegar la documentación a GitHub Pages (manual)
mkdocs gh-deploy

## Verificar enlaces rotos
mkdocs build --strict

## Construir con modo verbose para debugging
mkdocs build --verbose

## Limpiar archivos de construcción
mkdocs build --clean

## Variables de entorno útiles
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

## Estructura de archivos de documentación
"""
docs/
├── index.md                    # Página principal
├── installation.md             # Guía de instalación
├── preprocessing.md             # Documentación de preprocesamiento
├── experiments.md               # Configuración de experimentos
├── visualization.md             # Guía de visualización
├── contributing.md              # Guía de contribución
├── api/                        # Documentación de API
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
└── ...
"""

## Tips para escribir documentación

1. **Usa docstrings estilo Google** en tu código Python
2. **Incluye ejemplos** en los docstrings
3. **Agrega type hints** para mejor documentación automática
4. **Usa admoniciones** para destacar información importante:
   - !!! note "Nota"
   - !!! warning "Advertencia"
   - !!! tip "Consejo"
   - !!! danger "Peligro"

5. **Ejemplos de código** con syntax highlighting:
   ```python
   def mi_funcion(param: str) -> int:
       """Ejemplo de función documentada.
       
       Args:
           param: Descripción del parámetro
           
       Returns:
           Descripción del valor de retorno
           
       Example:
           >>> mi_funcion("test")
           42
       """
       return 42
   ```

## Troubleshooting

### Error: Module not found
- Asegúrate de que `pip install -e .` se ejecutó correctamente
- Verifica que el PYTHONPATH incluya el directorio del proyecto

### Error: mkdocstrings no encuentra los módulos
- Verifica que los paths en mkdocs.yml sean correctos
- Asegúrate de que los __init__.py existan en los directorios de módulos

### Links rotos
- Usa `mkdocs build --strict` para encontrar enlaces rotos
- Verifica las rutas relativas en los archivos markdown

### Problemas con el tema Material
- Verifica la versión de mkdocs-material
- Revisa la sintaxis de las extensiones en mkdocs.yml

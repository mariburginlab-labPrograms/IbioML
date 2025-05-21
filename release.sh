#!/bin/bash

set -e  # si algo falla, se corta

# Parámetros
VERSION=$1   # ejemplo: ./release.sh 0.1.2 pypi
REPO=${2:-testpypi}  # si no se pasa, usa testpypi por defecto

if [ -z "$VERSION" ]; then
  echo "⚠️  Tenés que pasar una versión. Ej: ./release.sh 0.1.4 pypi"
  exit 1
fi

echo "📦 Actualizando a la versión $VERSION"

# Actualiza la versión en pyproject.toml o setup.py
if [ -f "pyproject.toml" ]; then
  sed -i '' "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
elif [ -f "setup.py" ]; then
  sed -i '' "s/version=.*,/version=\"$VERSION\",/" setup.py
else
  echo "❌ No encontré ni setup.py ni pyproject.toml"
  exit 1
fi

# Limpia versiones viejas
rm -rf dist/ build/ *.egg-info/

# Build
echo "⚙️  Generando paquete..."
python3 -m build

# Upload
echo "☁️  Subiendo a $REPO"
twine upload --repository $REPO dist/*

# (Opcional) Git
echo "📤 Haciendo commit y push"
git add .
git commit -m "Release v$VERSION"
git tag "v$VERSION"
git push && git push --tags

echo "✅ ¡Listo! Versión $VERSION subida a $REPO"

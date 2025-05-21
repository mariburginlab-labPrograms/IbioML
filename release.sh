#!/bin/bash

set -e  # si algo falla, se corta

# Parámetros
REPO="testpypi"  # o "testpypi" si querés probar
VERSION=$1   # pasás la versión como argumento (ej: ./release.sh 0.1.4)

if [ -z "$VERSION" ]; then
  echo "⚠️  Tenés que pasar una versión. Ej: ./release.sh 0.1.4"
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
if [ "$REPO" = "testpypi" ]; then
  twine upload --repository testpypi dist/*
else
  twine upload dist/*
fi

# (Opcional) Git
echo "📤 Haciendo commit y push"
git add .
git commit -m "Release v$VERSION"
git tag "v$VERSION"
git push && git push --tags

echo "✅ ¡Listo! Versión $VERSION subida a $REPO"

#!/bin/bash

# Script para gestionar la documentación de IbioML
# Uso: ./docs.sh [comando]

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo -e "${BLUE}IbioML Documentation Manager${NC}"
    echo ""
    echo "Uso: ./docs.sh [comando]"
    echo ""
    echo "Comandos disponibles:"
    echo "  setup     - Instalar dependencias de documentación"
    echo "  serve     - Servir documentación localmente (puerto 8000)"
    echo "  build     - Construir documentación estática"
    echo "  deploy    - Desplegar a GitHub Pages"
    echo "  check     - Verificar enlaces y construcción"
    echo "  clean     - Limpiar archivos de construcción"
    echo "  help      - Mostrar esta ayuda"
    echo ""
}

# Función para instalar dependencias
setup_docs() {
    echo -e "${BLUE}📦 Instalando dependencias de documentación...${NC}"
    pip install -r requirements-docs.txt
    pip install -e .
    echo -e "${GREEN}✅ Dependencias instaladas correctamente${NC}"
}

# Función para servir localmente
serve_docs() {
    echo -e "${BLUE}🚀 Sirviendo documentación en http://localhost:8000${NC}"
    echo -e "${YELLOW}💡 Presiona Ctrl+C para detener${NC}"
    mkdocs serve
}

# Función para construir
build_docs() {
    echo -e "${BLUE}🔨 Construyendo documentación...${NC}"
    mkdocs build
    echo -e "${GREEN}✅ Documentación construida en ./site/${NC}"
}

# Función para desplegar
deploy_docs() {
    echo -e "${BLUE}🚀 Desplegando a GitHub Pages...${NC}"
    echo -e "${YELLOW}⚠️  Esto sobrescribirá la documentación actual en GitHub Pages${NC}"
    read -p "¿Continuar? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdocs gh-deploy
        echo -e "${GREEN}✅ Documentación desplegada correctamente${NC}"
        echo -e "${BLUE}🌐 Disponible en: https://mariburginlab-labprograms.github.io/IbioML/${NC}"
    else
        echo -e "${YELLOW}❌ Despliegue cancelado${NC}"
    fi
}

# Función para verificar
check_docs() {
    echo -e "${BLUE}🔍 Verificando documentación...${NC}"
    
    echo "  ➤ Verificando construcción estricta..."
    if mkdocs build --strict; then
        echo -e "${GREEN}  ✅ Construcción estricta exitosa${NC}"
    else
        echo -e "${RED}  ❌ Errores encontrados en la construcción${NC}"
        return 1
    fi
    
    echo "  ➤ Verificando importaciones de Python..."
    if python -c "import ibioml; print('✅ Importación exitosa')"; then
        echo -e "${GREEN}  ✅ Módulos de Python importables${NC}"
    else
        echo -e "${RED}  ❌ Error al importar módulos${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Todas las verificaciones pasaron${NC}"
}

# Función para limpiar
clean_docs() {
    echo -e "${BLUE}🧹 Limpiando archivos de construcción...${NC}"
    
    if [ -d "site" ]; then
        rm -rf site
        echo -e "${GREEN}  ✅ Directorio site/ eliminado${NC}"
    fi
    
    if [ -d "docs/build" ]; then
        rm -rf docs/build
        echo -e "${GREEN}  ✅ Directorio docs/build/ eliminado${NC}"
    fi
    
    # Limpiar archivos temporales
    find . -name "*.tmp.md" -delete 2>/dev/null || true
    
    echo -e "${GREEN}✅ Limpieza completada${NC}"
}

# Función para mostrar estado
show_status() {
    echo -e "${BLUE}📊 Estado de la documentación:${NC}"
    echo ""
    
    # Verificar si las dependencias están instaladas
    if python -c "import mkdocs" 2>/dev/null; then
        echo -e "${GREEN}  ✅ MkDocs instalado${NC}"
    else
        echo -e "${RED}  ❌ MkDocs no encontrado${NC}"
    fi
    
    if python -c "import material" 2>/dev/null; then
        echo -e "${GREEN}  ✅ Material theme instalado${NC}"
    else
        echo -e "${RED}  ❌ Material theme no encontrado${NC}"
    fi
    
    if python -c "import mkdocstrings" 2>/dev/null; then
        echo -e "${GREEN}  ✅ MkDocstrings instalado${NC}"
    else
        echo -e "${RED}  ❌ MkDocstrings no encontrado${NC}"
    fi
    
    # Verificar archivos de configuración
    if [ -f "mkdocs.yml" ]; then
        echo -e "${GREEN}  ✅ mkdocs.yml encontrado${NC}"
    else
        echo -e "${RED}  ❌ mkdocs.yml no encontrado${NC}"
    fi
    
    if [ -f "requirements-docs.txt" ]; then
        echo -e "${GREEN}  ✅ requirements-docs.txt encontrado${NC}"
    else
        echo -e "${RED}  ❌ requirements-docs.txt no encontrado${NC}"
    fi
    
    # Verificar directorio docs
    if [ -d "docs" ]; then
        doc_count=$(find docs -name "*.md" | wc -l)
        echo -e "${GREEN}  ✅ Directorio docs/ encontrado (${doc_count} archivos .md)${NC}"
    else
        echo -e "${RED}  ❌ Directorio docs/ no encontrado${NC}"
    fi
    
    echo ""
}

# Procesamiento de comandos
case "${1:-help}" in
    setup)
        setup_docs
        ;;
    serve)
        serve_docs
        ;;
    build)
        build_docs
        ;;
    deploy)
        deploy_docs
        ;;
    check)
        check_docs
        ;;
    clean)
        clean_docs
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Comando desconocido: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

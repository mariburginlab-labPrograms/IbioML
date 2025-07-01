#!/usr/bin/env python3
"""
Script de verificación para asegurar que todos los módulos se importan correctamente
antes del despliegue de la documentación.
"""

def test_imports():
    """Prueba todas las importaciones críticas."""
    
    print("🧪 Verificando importaciones de IbioML...")
    
    try:
        # Importaciones principales
        import ibioml
        print("✅ ibioml")
        
        import ibioml.models
        print("✅ ibioml.models")
        
        import ibioml.trainer
        print("✅ ibioml.trainer")
        
        import ibioml.tuner
        print("✅ ibioml.tuner")
        
        import ibioml.plots
        print("✅ ibioml.plots")
        
        import ibioml.preprocess_data
        print("✅ ibioml.preprocess_data")
        
        # Importaciones utils
        import ibioml.utils.trainer_funcs
        print("✅ ibioml.utils.trainer_funcs")
        
        import ibioml.utils.tuner_funcs
        print("✅ ibioml.utils.tuner_funcs")
        
        import ibioml.utils.preprocessing_funcs
        print("✅ ibioml.utils.preprocessing_funcs")
        
        import ibioml.utils.data_scaler
        print("✅ ibioml.utils.data_scaler")
        
        import ibioml.utils.evaluators
        print("✅ ibioml.utils.evaluators")
        
        import ibioml.utils.pipeline_utils
        print("✅ ibioml.utils.pipeline_utils")
        
        import ibioml.utils.plot_functions
        print("✅ ibioml.utils.plot_functions")
        
        import ibioml.utils.plot_styles
        print("✅ ibioml.utils.plot_styles")
        
        import ibioml.utils.splitters
        print("✅ ibioml.utils.splitters")
        
        # Importación problemática (model_factory) - debe importarse directamente
        import ibioml.utils.model_factory
        print("✅ ibioml.utils.model_factory")
        
        print("\n🎉 ¡Todas las importaciones exitosas!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en importación: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_functions():
    """Prueba que las funciones principales estén disponibles."""
    
    print("\n🔍 Verificando funciones principales...")
    
    try:
        # Verificar que las clases principales estén disponibles
        from ibioml.models import MLPModel, RNNModel, LSTMModel, GRUModel
        print("✅ Modelos principales importados")
        
        from ibioml.utils.trainer_funcs import initialize_weights, create_dataloaders, EarlyStopping
        print("✅ Funciones de entrenamiento importadas")
        
        from ibioml.utils.preprocessing_funcs import get_spikes_with_history
        print("✅ Funciones de preprocesamiento importadas")
        
        from ibioml.preprocess_data import preprocess_data
        print("✅ Función principal de preprocesamiento importada")
        
        print("\n🎉 ¡Todas las funciones principales disponibles!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en funciones principales: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VERIFICACIÓN DE IMPORTACIONES PARA DOCUMENTACIÓN")
    print("=" * 60)
    
    imports_ok = test_imports()
    functions_ok = test_main_functions()
    
    print("\n" + "=" * 60)
    if imports_ok and functions_ok:
        print("✅ TODAS LAS VERIFICACIONES PASARON")
        print("🚀 La documentación debería construirse correctamente")
        exit(0)
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print("🛠️  Revisa los errores antes de hacer el PR")
        exit(1)

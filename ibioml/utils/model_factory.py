from ibioml.utils.models.models import (
    MLPModel, RNNModel, GRUModel, LSTMModel,
    DualOutputMLPModel, DualOutputRNNModel, DualOutputGRUModel, DualOutputLSTMModel
)

def create_model_class(base_model_class, y_dim):
    """
    Factory function que selecciona automáticamente entre versiones
    de salida única o dual de un modelo basado en la dimensión de y.
    
    Args:
        base_model_class: Clase base del modelo (MLPModel, RNNModel, etc.)
        y_dim: Dimensión de salida (1 o 2)
        
    Returns:
        La clase de modelo apropiada
    """
    model_mapping = {
        # Mapeo de modelos base a sus versiones dual output
        MLPModel: DualOutputMLPModel,
        RNNModel: DualOutputRNNModel,
        GRUModel: DualOutputGRUModel,
        LSTMModel: DualOutputLSTMModel
    }
    
    if y_dim == 2:
        # Si y tiene dos dimensiones (posición y velocidad), usar dual output
        if base_model_class in model_mapping:
            return model_mapping[base_model_class]
        else:
            raise ValueError(f"No existe versión DualOutput para {base_model_class.__name__}")
    else:
        # Para una dimensión, usar el modelo base
        return base_model_class
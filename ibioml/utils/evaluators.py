from abc import ABC, abstractmethod
from sklearn.metrics import r2_score
import numpy as np

class ModelEvaluator(ABC):
    """Clase abstracta base para evaluadores de modelos."""
    
    def __init__(self, device):
        self.device = device
    
    @abstractmethod
    def predict(self, model, X_tensor):
        """Realiza predicciones con el modelo."""
        pass
    
    @abstractmethod
    def calculate_r2(self, y_true_scaled, predictions):
        """Calcula el score R2."""
        pass
    
    @abstractmethod
    def rescale_predictions(self, scaler, predictions):
        """Re-escala las predicciones a valores originales."""
        pass
    
    @abstractmethod
    def prepare_results(self, predictions_rescaled, y_true):
        """Prepara los resultados para guardar."""
        pass
    
    @abstractmethod
    def initialize_results(self):
        """Inicializa el diccionario de resultados."""
        pass
    
    @abstractmethod
    def update_results(self, results, fold_results):
        """Actualiza el diccionario de resultados con los resultados del fold actual."""
        pass
    
    @abstractmethod
    def print_summary(self, results):
        """Imprime un resumen de los resultados."""
        pass
    
    @abstractmethod
    def prepare_evaluation_data(self, scaler, y_true):
        """Prepara los datos para evaluación."""
        pass


class SingleOutputEvaluator(ModelEvaluator):
    """Evaluador para modelos de salida única."""
    
    def predict(self, model, X_tensor):
        """Realiza predicciones con el modelo."""
        if hasattr(model.__class__, "__name__") and model.__class__.__name__ == "MLPModel":
            y_pred = model(X_tensor).cpu().numpy()
        else:
            y_pred = model(X_tensor).squeeze().cpu().numpy()
        return y_pred
    
    def calculate_r2(self, y_true_scaled, predictions):
        """Calcula el score R2."""
        return r2_score(y_true_scaled, predictions), None, None
    
    def rescale_predictions(self, scaler, predictions):
        """Re-escala las predicciones a valores originales."""
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return scaler.target_scaler.inverse_transform(predictions).ravel().tolist()
    
    def prepare_results(self, predictions_rescaled, y_true):
        """Prepara los resultados para guardar."""
        return predictions_rescaled, y_true.ravel().tolist()
    
    def initialize_results(self):
        """Inicializa el diccionario de resultados."""
        return {
            "test_r2_scores": [],
            "best_config_per_fold": [],
            "predictions_per_fold": [],
            "true_values_per_fold": []
        }
    
    def update_results(self, results, fold_results):
        """Actualiza el diccionario de resultados con los resultados del fold actual."""
        fold_r2, _, _ = fold_results["r2_scores"]
        results["test_r2_scores"].append(fold_r2)
        results["predictions_per_fold"].append(fold_results["predictions"])
        results["true_values_per_fold"].append(fold_results["true_values"])
        return results
    
    def print_summary(self, results):
        """Imprime un resumen de los resultados."""
        print(f"Mean R² Score: {np.mean(results['test_r2_scores']):.4f} ± {np.std(results['test_r2_scores']):.4f}")
    
    def prepare_evaluation_data(self, scaler, y_true):
        """Prepara los datos para evaluación."""
        return scaler.target_scaler.transform(y_true.reshape(-1, 1))


class DualOutputEvaluator(ModelEvaluator):
    """Evaluador para modelos de salida dual."""
    
    def predict(self, model, X_tensor):
        """Realiza predicciones con el modelo."""
        pos_pred, vel_pred = model(X_tensor)
        pos_pred = pos_pred.cpu().numpy()
        vel_pred = vel_pred.cpu().numpy()
        return (pos_pred, vel_pred)
    
    def calculate_r2(self, y_true_scaled, predictions):
        """Calcula el score R2 para salidas duales."""
        pos_pred, vel_pred = predictions
        r2_pos = r2_score(y_true_scaled[:, 0].reshape(-1, 1), pos_pred)
        r2_vel = r2_score(y_true_scaled[:, 1].reshape(-1, 1), vel_pred)
        # Devolver R2 promedio y componentes
        return (r2_pos + r2_vel) / 2, r2_pos, r2_vel
    
    def rescale_predictions(self, scaler, predictions):
        """Re-escala las predicciones a valores originales."""
        pos_pred, vel_pred = predictions
        pos_pred_rescaled = scaler.inverse_transform_pos(pos_pred).ravel().tolist()
        vel_pred_rescaled = scaler.inverse_transform_vel(vel_pred).ravel().tolist()
        return (pos_pred_rescaled, vel_pred_rescaled)
    
    def prepare_results(self, predictions_rescaled, y_true):
        """Prepara los resultados para guardar."""
        pos_pred_rescaled, vel_pred_rescaled = predictions_rescaled
        return [pos_pred_rescaled, vel_pred_rescaled], [y_true[:, 0].tolist(), y_true[:, 1].tolist()]
    
    def initialize_results(self):
        """Inicializa el diccionario de resultados con la nueva estructura."""
        results = {
            "test_r2_scores": [],
            "test_r2_scores_pos": [],
            "test_r2_scores_vel": [],
            "best_config_per_fold": [],
            "predictions_per_fold": [],
            "true_values_per_fold": []
        }
        return results
    
    def update_results(self, results, fold_results):
        """Actualiza el diccionario de resultados con los resultados del fold actual."""
        fold_r2, r2_pos, r2_vel = fold_results["r2_scores"]
        results["test_r2_scores"].append(fold_r2)
        results["test_r2_scores_pos"].append(r2_pos)
        results["test_r2_scores_vel"].append(r2_vel)
        
        # Reorganizar las predicciones en formato diccionario 
        pos_pred, vel_pred = fold_results["predictions"] 
        predictions_dict = {
            "position": pos_pred,
            "velocity": vel_pred
        }
        results["predictions_per_fold"].append(predictions_dict)
        
        # Reorganizar los valores verdaderos en formato diccionario
        pos_true, vel_true = fold_results["true_values"]
        true_values_dict = {
            "position": pos_true,
            "velocity": vel_true
        }
        results["true_values_per_fold"].append(true_values_dict)
        
        return results
    
    def print_summary(self, results):
        """Imprime un resumen de los resultados."""
        print(f"Mean R² Score: {np.mean(results['test_r2_scores']):.4f} ± {np.std(results['test_r2_scores']):.4f}")
        print(f"Mean Position R² Score: {np.mean(results['test_r2_scores_pos']):.4f} ± {np.std(results['test_r2_scores_pos']):.4f}")
        print(f"Mean Velocity R² Score: {np.mean(results['test_r2_scores_vel']):.4f} ± {np.std(results['test_r2_scores_vel']):.4f}")
    
    def prepare_evaluation_data(self, scaler, y_true):
        """Prepara los datos para evaluación."""
        return scaler.target_scaler.transform(y_true)


def create_evaluator(target_dim, device):
    """Crea el evaluador apropiado según el tipo de modelo."""
    if target_dim == 1:
        return SingleOutputEvaluator(device)
    elif target_dim == 2:
        return DualOutputEvaluator(device)
    else:
        raise ValueError("target_dim debe ser 1 o 2.")
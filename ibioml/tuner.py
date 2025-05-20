import numpy as np
import torch
import datetime
import optuna
import os
import json
import copy
from .trainer import train_model
from .utils.trainer_funcs import create_dataloaders
from .utils.data_scaler import scale_data
from .utils.evaluators import create_evaluator, ModelEvaluator
from .utils.splitters import TrialKFold, trial_train_test_split
from sklearn.model_selection import KFold, train_test_split

class Tuner:
    """
    Afinador de hiperparámetros con validación cruzada anidada.
    
    Esta clase maneja la optimización de hiperparámetros usando Optuna y CV anidada.
    Soporta modelos de salida única y dual.
    """
    
    def __init__(self, model_space=None, evaluator: ModelEvaluator = None):
        """Inicializa el afinador con una configuración de modelo opcional y un evaluador."""
        self.model_space = model_space
        self.evaluator = evaluator
        self.save_path = "results"
        self.study_name = None
        self.num_trials = 5
        self.outer_folds = 5
        self.inner_folds = 5
        self.search_alg = "bayes"
        self.search_space = None
    
    def set_model_space(self, model_space):
        """Establece la configuración del espacio de modelos."""
        self.model_space = model_space
        return self
    
    def set_study_params(self, save_path="results", study_name=None, num_trials=5, 
                        outer_folds=5, inner_folds=5, search_alg="bayes", 
                        search_space=None):
        """Establece todos los parámetros del estudio a la vez."""
        self.save_path = save_path
        self.study_name = study_name
        self.num_trials = num_trials
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.search_alg = search_alg
        self.search_space = search_space
        return self
    
    def set_evaluator(self, evaluator: ModelEvaluator):
        """Establece la instancia del evaluador."""
        self.evaluator = evaluator
        return self
    
    def run(self, X, y, T):
        """Ejecuta la optimización de hiperparámetros con CV anidada."""
        # Setup directories
        experiment_dir, training_results_dir = self._setup_directories()
        
        # Initialize results storage
        results = self._initialize_results()
        
        # Run outer CV loop
        self._run_outer_cv_loop(X, y, T, results, training_results_dir)
        
        # Save final results
        self._save_final_results(results, experiment_dir)
        
        # Print summary
        self._print_cv_summary(results)
        
        return results
    
    def _setup_directories(self):
        """Crea la estructura de directorios para los resultados del experimento."""
        if self.study_name:
            experiment_dir = os.path.join(self.save_path, self.study_name)
        else:    
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_dir = os.path.join(self.save_path, f"study_{timestamp}")
        
        training_results_dir = os.path.join(experiment_dir, "training_results")
        
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(training_results_dir, exist_ok=True)
        
        return experiment_dir, training_results_dir
    
    def _initialize_results(self):
        """Inicializa el diccionario de resultados usando el evaluador."""
        return self.evaluator.initialize_results()
    
    def _run_outer_cv_loop(self, X, y, T, results, training_results_dir):
        """Ejecuta el bucle externo de validación cruzada."""
        outer_cv = TrialKFold(n_splits=self.outer_folds, shuffle=True)
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(trial_markers=T)):
            print(f"\n===== Outer Fold {fold_idx+1}/{self.outer_folds} =====")
            
            # Divide los datos para este fold
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            T_train_val = T[train_val_idx]
            
            # Crea un directorio para el fold
            fold_dir = os.path.join(training_results_dir, f"fold_{fold_idx+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Divide datos para validación interna
            X_train, X_val, y_train, y_val = trial_train_test_split(
                X_train_val, y_train_val, T_train_val, train_size=0.8
            )
            
            # Optimiza para este fold
            best_model, final_config, study, scaler = self._optimize_fold(
                fold_idx, fold_dir, X_train, y_train, X_val, y_val
            )
            
            # Registra la mejor configuración
            results["best_config_per_fold"].append(make_serializable(final_config))
            
            # Evalúa modelo
            fold_results = self._evaluate_model(
                best_model, scaler, X_test, y_test, X_val, y_val, study
            )
            
            # Actualiza resultados
            self._update_results(results, fold_results)
            
            # Guarda resultados del fold
            self._save_fold_results(fold_dir, fold_results, study)
    
    def _optimize_fold(self, fold_idx, fold_dir, X_train, y_train, X_val, y_val):
        """Optimiza hiperparámetros para un solo fold."""
        # Inicializa configuración y obtiene escalador
        config, scaler = initialize_config(
            self.model_space, X_train, y_train, X_val, y_val, get_scaler=True
        )
        
        # Define función objetivo
        def objective_fn(trial):
            return self._objective_function(trial, config, fold_dir)
        
        # Crea y ejecuta estudio de Optuna
        study = self._create_study()
        study.optimize(objective_fn, n_trials=self.num_trials)
        
        # Imprime resultados
        self._print_study_results(study)
        
        # Carga el mejor modelo
        best_model_path = os.path.join(fold_dir, "best_model.pt")
        best_model = torch.load(best_model_path, weights_only=False)
        
        # Crea configuración final
        final_config = copy.deepcopy(config)
        for key, value in study.best_params.items():
            final_config[key] = value
        
        return best_model, final_config, study, scaler
    
    def _objective_function(self, trial, base_config, fold_dir):
        """Función objetivo para la optimización de Optuna."""
        # Crea una copia de la configuración base
        trial_config = copy.deepcopy(base_config)
        
        # Define hiperparámetros a optimizar
        trial_config["hidden_size"] = trial.suggest_int("hidden_size", 128, 512, step=64)
        trial_config["num_layers"] = trial.suggest_int("num_layers", 1, 3)
        trial_config["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        trial_config["lr"] = trial.suggest_float("lr", 1e-8, 1e-3, log=True)
        
        # Maneja casos especiales para modelos RNN
        if hasattr(trial_config["model_class"], "__name__") and "RNN" in trial_config["model_class"].__name__:
            if trial_config["num_layers"] == 1:
                trial_config["dropout"] = 0.0  # Avoid PyTorch warning for single-layer RNNs
        
        # Crea directorio para esta prueba
        model_dir = os.path.join(fold_dir, f"trial_{trial.number}")
        os.makedirs(model_dir, exist_ok=True)
        trial_config["results_dir"] = model_dir
        
        # Entrena el modelo
        results_dict = train_model(trial_config)
        current_r2 = results_dict["r2_score"]
        
        # Guarda el modelo si es el mejor hasta ahora
        if trial.number == 0 or current_r2 > trial.study.best_value:
            print(f"New best model found! R² = {current_r2:.4f}")
            torch.save(results_dict["model"], os.path.join(fold_dir, "best_model.pt"))
        
        return current_r2
    
    def _create_study(self):
        """Crea un estudio de Optuna con el algoritmo de búsqueda especificado."""
        if self.search_alg == "bayes":
            return optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler()
            )
        elif self.search_alg == "grid":
            return optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.GridSampler(self.search_space)
            )
    
    def _print_study_results(self, study):
        """Imprime los resultados de un estudio de Optuna."""
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best Validation R² score: {study.best_value:.4f}")
    
    def _evaluate_model(self, model, scaler, X_test, y_test, X_val, y_val, study):
        """Evalúa un modelo en datos de prueba y validación usando el evaluador."""
        # Escala datos de prueba
        X_test_scaled = scaler.feature_scaler.transform(X_test)
        X_val_scaled = scaler.feature_scaler.transform(X_val)
        
        # Inicializa diccionario de resultados
        fold_results = {}
        
        # Evalúa en datos de prueba
        model.eval()
        with torch.no_grad():
            # Configura tensores
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.model_space["device"])
            
            # Obtiene predicciones de prueba y métricas
            y_test_scaled = self.evaluator.prepare_evaluation_data(scaler, y_test)
            predictions = self.evaluator.predict(model, X_test_tensor)
            r2_scores = self.evaluator.calculate_r2(y_test_scaled, predictions)
            predictions_rescaled = self.evaluator.rescale_predictions(scaler, predictions)
            fold_results["predictions"], fold_results["true_values"] = self.evaluator.prepare_results(predictions_rescaled, y_test)
            fold_results["r2_scores"] = r2_scores
            
            # Imprime resultados
            print(f"Test R² Score: {r2_scores[0]:.4f}")
            if len(r2_scores) > 1:
                print(f"Additional R² Scores: {r2_scores[1:]}")
            
            # Evalúa en datos de validación (comprobación de coherencia)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.model_space["device"])
            y_val_scaled = self.evaluator.prepare_evaluation_data(scaler, y_val)
            val_predictions = self.evaluator.predict(model, X_val_tensor)
            val_r2_scores = self.evaluator.calculate_r2(y_val_scaled, val_predictions)
            
            # Añade métricas de validación a los resultados
            fold_results["val_r2"] = val_r2_scores[0]
            if len(val_r2_scores) > 1:
                fold_results["val_r2_scores"] = val_r2_scores[1:]
            
            # Añade información del estudio
            fold_results["study"] = {
                "best_trial_number": study.best_trial.number,
                        "best_validation_r2_score": val_r2_scores[0]
            }
        
        return fold_results
    
    def _update_results(self, results, fold_results):
        """Actualiza los resultados globales usando el evaluador."""
        return self.evaluator.update_results(results, fold_results)
    
    def _save_fold_results(self, fold_dir, fold_results, study):
        """Guarda los resultados de un fold en un archivo JSON."""
        outer_fold_results = {
            "best_trial_number": study.best_trial.number,
            "best_validation_r2_score": fold_results["val_r2"],
            "test_r2_score": fold_results["r2_scores"][0],
        }
        
        if len(fold_results["r2_scores"]) > 1:
            outer_fold_results["additional_test_r2_scores"] = fold_results["r2_scores"][1:]
            outer_fold_results["additional_val_r2_scores"] = fold_results.get("val_r2_scores", [])
        
        with open(f"{fold_dir}/results.json", "w") as f:
            json.dump(outer_fold_results, f, indent=4, default=str)
    
    def _save_final_results(self, results, experiment_dir):
        """Guarda los resultados finales en un archivo JSON."""
        # Asegurarse de que los resultados sean serializables
        json_results = {}
        
        # Copiar todos los campos existentes
        for key, value in results.items():
            json_results[key] = make_serializable(value)
        
        # Guardar en formato JSON
        with open(f"{experiment_dir}/final_results.json", "w") as f:
            json.dump(json_results, f, default=str)
    
    def _print_cv_summary(self, results):
        """Imprime un resumen de los resultados de validación cruzada usando el evaluador."""
        print("\n===== Nested CV Summary =====")
        self.evaluator.print_summary(results)


# Keep original functions for backward compatibility
def initialize_config(config, X_train, y_train, X_val, y_val, get_scaler=False):
    """
    Inicializa la configuración para el entrenamiento.
    Maneja automáticamente modelos de salida única y dual.
    """
    # Detectar automáticamente la dimensión de salida
    if y_train.ndim > 1 and y_train.shape[1] == 2:
        y_dim = 2
    else:
        y_dim = 1
    
    # Escalar datos
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler = scale_data(
        X_train, y_train, X_val, y_val, return_scaler=True
    )
    
    # Crear DataLoaders
    train_loader, val_loader = create_dataloaders(
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, config["batch_size"]
    )
    
    # Configurar modelo y otros parámetros
    train_config = copy.deepcopy(config)
    train_config["input_size"] = X_train.shape[X_train.ndim-1]
    train_config["train_loader"] = train_loader
    train_config["val_loader"] = val_loader
    train_config["scaler"] = scaler  # Store scaler in config
    
    # Configuración específica según la dimensión de salida
    if y_dim == 2:
        train_config["output_size"] = 1  # Cada cabeza predice un valor
    else:
        train_config["y_scaler"] = scaler.target_scaler
    
    if get_scaler:
        return train_config, scaler
    else:
        return train_config


def make_serializable(obj):
    """Convierte un objeto complejo a una forma serializable en JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items() 
                if not callable(v) and k not in ['device', 'model_class', 'train_loader', 'val_loader', 'y_scaler', 'scaler']}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    else:
        return obj


def run_study(X, y, T, model_space, num_trials=5, outer_folds=5, inner_folds=5, save_path="results", search_alg="bayes", search_space=None, study_name=None):
    """
    Validación cruzada anidada con Optuna para optimización de hiperparámetros.
    Utiliza el evaluador proporcionado para la evaluación del modelo.
    """
    # Inicializar el evaluador
    evaluator = create_evaluator(y.shape[1], model_space["device"])
    
    # Crear instancia del afinador
    tuner = Tuner(model_space, evaluator)
    
    # Configurar tuner
    tuner.set_study_params(
        save_path=save_path,
        study_name=study_name,
        num_trials=num_trials,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        search_alg=search_alg,
        search_space=search_space
    )
    
    # Ejecutar optimización
    tuner.run(X, y, T)
    return True
import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

class ExperimentResults:
    """
    Gestiona la carga y acceso a los resultados de un experimento.
    Implementa carga perezosa (lazy loading) para eficiencia.
    """
    def __init__(self, experiment_path: str):
        """
        Inicializa el gestor de resultados del experimento.

        Args:
            experiment_path (str): Ruta a la carpeta raíz del experimento.
        
        Raises:
            FileNotFoundError: Si la ruta del experimento o archivos/carpetas clave no existen.
        """
        self.base_path = Path(experiment_path)
        self.training_results_path = self.base_path / "training_results"
        self.final_results_file = self.base_path / "final_results.json"

        self._validate_paths()

        # Cachés para carga perezosa
        self._final_results_cache: Optional[Dict[str, Any]] = None
        self._fold_summaries_cache: Dict[str, Dict[str, Any]] = {}
        self._fold_best_models_cache: Dict[str, Any] = {} # Debería ser torch.nn.Module, pero Any por ahora

    def _validate_paths(self):
        """Verifica que las rutas y archivos esenciales existan."""
        if not self.base_path.is_dir():
            raise FileNotFoundError(f"La ruta del experimento no existe o no es un directorio: {self.base_path}")
        if not self.final_results_file.is_file():
            raise FileNotFoundError(f"El archivo 'final_results.json' no se encontró en: {self.base_path}")
        if not self.training_results_path.is_dir():
            raise FileNotFoundError(f"La carpeta 'training_results' no se encontró en: {self.base_path}")

    @property
    def final_results(self) -> Dict[str, Any]:
        """
        Retorna el contenido de 'final_results.json'.
        Los datos se cargan una vez y se cachean.
        """
        if self._final_results_cache is None:
            with open(self.final_results_file, 'r') as f:
                self._final_results_cache = json.load(f)
        return self._final_results_cache

    def get_fold_names(self) -> List[str]:
        """
        Retorna una lista con los nombres de las carpetas de los folds.
        Ej: ['fold_0', 'fold_1', ...]
        """
        if not self.training_results_path.exists():
            return []
        return sorted([p.name for p in self.training_results_path.iterdir() if p.is_dir()])

    def get_fold_summary(self, fold_name: str) -> Dict[str, Any]:
        """
        Retorna el contenido del 'results.json' para un fold específico.
        Los datos se cargan una vez por fold y se cachean.

        Args:
            fold_name (str): Nombre de la carpeta del fold (ej: 'fold_0').

        Raises:
            FileNotFoundError: Si el 'results.json' del fold no existe.
        """
        if fold_name not in self._fold_summaries_cache:
            fold_path = self.training_results_path / fold_name
            summary_file = fold_path / "results.json"
            if not summary_file.is_file():
                raise FileNotFoundError(f"El archivo 'results.json' no se encontró en el fold: {fold_path}")
            with open(summary_file, 'r') as f:
                self._fold_summaries_cache[fold_name] = json.load(f)
        return self._fold_summaries_cache[fold_name]

    def get_fold_best_model(self, fold_name: str, map_location: Optional[str] = 'cpu') -> Any: # Debería ser torch.nn.Module
        """
        Carga y retorna el mejor modelo (.pt) para un fold específico.
        El modelo se carga una vez por fold y se cachea.

        Args:
            fold_name (str): Nombre de la carpeta del fold (ej: 'fold_0').
            map_location (str, optional): Dispositivo donde cargar el modelo (ej: 'cpu', 'cuda').
                                          Por defecto 'cpu' para portabilidad.

        Raises:
            FileNotFoundError: Si no se encuentra un archivo .pt en la carpeta del fold.
        """
        if fold_name not in self._fold_best_models_cache:
            fold_path = self.training_results_path / fold_name
            model_files = list(fold_path.glob("*.pt")) # Asume que el modelo es un archivo .pt
            
            if not model_files:
                raise FileNotFoundError(f"No se encontró un archivo de modelo (.pt) en el fold: {fold_path}")
            if len(model_files) > 1:
                # Podrías tener una lógica más sofisticada si hay múltiples .pt,
                # por ahora toma el primero o lanza una advertencia/error.
                print(f"Advertencia: Se encontraron múltiples archivos .pt en {fold_path}. Usando {model_files[0]}.")

            model_path = model_files[0]
            self._fold_best_models_cache[fold_name] = torch.load(model_path, map_location=map_location)
        return self._fold_best_models_cache[fold_name]

    def get_all_fold_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Retorna un diccionario con los resúmenes de todos los folds."""
        all_summaries = {}
        for fold_name in self.get_fold_names():
            all_summaries[fold_name] = self.get_fold_summary(fold_name)
        return all_summaries

    def clear_cache(self):
        """Limpia todas las cachés para liberar memoria."""
        self._final_results_cache = None
        self._fold_summaries_cache.clear()
        self._fold_best_models_cache.clear()
        print("Caché de ExperimentResults limpiada.")

    def __repr__(self) -> str:
        return f"<ExperimentResults path='{self.base_path.name}'>"
import torch
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler

# ----------------------- ESCALADORES DE CARACTERÍSTICAS (X) -----------------------

class FeatureScaler(ABC):
    """Clase base abstracta para escaladores de características."""
    
    def __init__(self):
        self.x_scaler = StandardScaler()
    
    @abstractmethod
    def fit_transform(self, X_train):
        """Ajusta el escalador y transforma los datos de entrenamiento."""
        pass
    
    @abstractmethod
    def transform(self, X_unscaled):
        """Transforma los datos no escalados."""
        pass


class TwoDFeatureScaler(FeatureScaler):
    """Escalador para características 2D."""
    
    def fit_transform(self, X_train):
        return self.x_scaler.fit_transform(X_train)
    
    def transform(self, X_unscaled):
        return self.x_scaler.transform(X_unscaled)


class ThreeDFeatureScaler(FeatureScaler):
    """Escalador para características 3D (secuencias)."""
    
    def fit_transform(self, X_train):
        # Si los datos son 3D (samples, timesteps, features):
        n_samples, n_timesteps, n_features = X_train.shape
        
        # Aplanamos para escalar los features
        X_train_flat = X_train.reshape(-1, n_features)
        X_train_scaled = self.x_scaler.fit_transform(X_train_flat)
        
        # Volvemos a dar forma
        return X_train_scaled.reshape(n_samples, n_timesteps, n_features)
    
    def transform(self, X_unscaled):
        # Si los datos son 3D (samples, timesteps, features):
        n_samples, n_timesteps, n_features = X_unscaled.shape
        
        # Aplanamos para escalar
        X_flat = X_unscaled.reshape(-1, n_features)
        X_scaled = self.x_scaler.transform(X_flat)
        
        # Volvemos a dar forma
        return X_scaled.reshape(n_samples, n_timesteps, n_features)


# ----------------------- ESCALADORES DE OBJETIVOS (Y) -----------------------

class TargetScaler(ABC):
    """Clase base abstracta para escaladores de objetivos."""
    
    @abstractmethod
    def fit_transform(self, y_train):
        """Ajusta el escalador y transforma los objetivos de entrenamiento."""
        pass
    
    @abstractmethod
    def transform(self, y):
        """Transforma los objetivos usando el escalador ajustado."""
        pass
    
    @abstractmethod
    def inverse_transform(self, y_scaled):
        """Invierte la transformación para recuperar los valores originales."""
        pass


class SingleTargetScaler(TargetScaler):
    """Escalador para un único objetivo."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit_transform(self, y_train):
        # Asegurar que y es 2D para StandardScaler
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        return self.scaler.fit_transform(y_train)
    
    def transform(self, y):
        # Asegurar que y es 2D para StandardScaler
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return self.scaler.transform(y)
    
    def inverse_transform(self, y_scaled):
        # Asegurar que y es 2D para StandardScaler
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.scaler.inverse_transform(y_scaled)


class DualTargetScaler(TargetScaler):
    """Escalador para dos objetivos (posición y velocidad)."""
    
    def __init__(self):
        # Crear escaladores independientes para posición y velocidad
        self.pos_scaler = StandardScaler()
        self.vel_scaler = StandardScaler()
    
    def fit_transform(self, y_train):
        """
        Ajusta ambos escaladores y transforma los objetivos duales.
        
        Args:
            y_train: Array de forma (n_samples, 2) con posición en [:, 0] y velocidad en [:, 1]
            
        Returns:
            Array escalado de la misma forma
        """
        # Extraer y escalar posición (primera columna)
        pos = y_train[:, 0].reshape(-1, 1)
        pos_scaled = self.pos_scaler.fit_transform(pos)
        
        # Extraer y escalar velocidad (segunda columna)
        vel = y_train[:, 1].reshape(-1, 1)
        vel_scaled = self.vel_scaler.fit_transform(vel)
        
        # Combinar resultados
        return np.hstack([pos_scaled, vel_scaled])
    
    def transform(self, y):
        """Transforma los datos de posición y velocidad."""
        # Extraer y escalar posición y velocidad
        pos = y[:, 0].reshape(-1, 1)
        vel = y[:, 1].reshape(-1, 1)
        
        pos_scaled = self.pos_scaler.transform(pos)
        vel_scaled = self.vel_scaler.transform(vel)
        
        return np.hstack([pos_scaled, vel_scaled])
    
    def inverse_transform(self, y_scaled):
        """Invierte la transformación para ambos objetivos."""
        # Si es una lista de dos arrays separados
        if isinstance(y_scaled, list) and len(y_scaled) == 2:
            pos_scaled, vel_scaled = y_scaled
            
            if pos_scaled.ndim == 1:
                pos_scaled = pos_scaled.reshape(-1, 1)
            if vel_scaled.ndim == 1:
                vel_scaled = vel_scaled.reshape(-1, 1)
                
            pos = self.pos_scaler.inverse_transform(pos_scaled)
            vel = self.vel_scaler.inverse_transform(vel_scaled)
            
            return np.hstack([pos, vel])
        
        # Si es un único array con ambas columnas
        pos_scaled = y_scaled[:, 0].reshape(-1, 1)
        vel_scaled = y_scaled[:, 1].reshape(-1, 1)
        
        pos = self.pos_scaler.inverse_transform(pos_scaled)
        vel = self.vel_scaler.inverse_transform(vel_scaled)
        
        return np.hstack([pos, vel])
    
    # Métodos específicos para transformar cada objetivo individualmente
    def inverse_transform_pos(self, pos_scaled):
        """Invierte la transformación sólo para posición."""
        if pos_scaled.ndim == 1:
            pos_scaled = pos_scaled.reshape(-1, 1)
        return self.pos_scaler.inverse_transform(pos_scaled)
    
    def inverse_transform_vel(self, vel_scaled):
        """Invierte la transformación sólo para velocidad."""
        if vel_scaled.ndim == 1:
            vel_scaled = vel_scaled.reshape(-1, 1)
        return self.vel_scaler.inverse_transform(vel_scaled)


# ----------------------- FACTORIES Y SISTEMA DE ESCALADO COMPLETO -----------------------

class DataScaler:
    """
    Escalador completo que combina escaladores de características y objetivos.
    
    Esta clase utiliza el patrón de composición para delegar el escalado
    a las clases especializadas según las características de los datos.
    """
    
    def __init__(self, feature_scaler, target_scaler):
        """
        Args:
            feature_scaler: Escalador para características (X)
            target_scaler: Escalador para objetivos (y)
        """
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
    
    def standardize(self, X_train, y_train, X_test, y_test):
        """
        Estandariza características y objetivos de entrenamiento y prueba.
        
        Returns:
            tuple: (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)
        """
        # Escalar características
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Escalar objetivos
        y_train_scaled = self.target_scaler.fit_transform(y_train)
        y_test_scaled = self.target_scaler.transform(y_test)
        
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
    
    def inverse_transform(self, y_scaled):
        """Invierte la transformación para recuperar valores originales."""
        return self.target_scaler.inverse_transform(y_scaled)
    
    # Delegación de métodos específicos de DualTargetScaler
    def inverse_transform_pos(self, pos_scaled):
        """Invierte la transformación para posición (sólo si es DualTargetScaler)."""
        if hasattr(self.target_scaler, 'inverse_transform_pos'):
            return self.target_scaler.inverse_transform_pos(pos_scaled)
        raise AttributeError("El escalador no soporta inverse_transform_pos")
    
    def inverse_transform_vel(self, vel_scaled):
        """Invierte la transformación para velocidad (sólo si es DualTargetScaler)."""
        if hasattr(self.target_scaler, 'inverse_transform_vel'):
            return self.target_scaler.inverse_transform_vel(vel_scaled)
        raise AttributeError("El escalador no soporta inverse_transform_vel")
    
    @property
    def is_dual(self):
        """Indica si el escalador de objetivos es dual."""
        return isinstance(self.target_scaler, DualTargetScaler)


def create_scaler(X, y):
    """
    Crea un escalador adecuado basado en las características de los datos.
    
    Args:
        X: Datos de características
        y: Datos de objetivos
        
    Returns:
        DataScaler: Escalador configurado para los datos
    """
    # Determinar el tipo de escalador de características
    if X.ndim == 3:
        feature_scaler = ThreeDFeatureScaler()
    else:
        feature_scaler = TwoDFeatureScaler()
    
    # Determinar el tipo de escalador de objetivos
    if y.ndim > 1 and y.shape[1] == 2:
        target_scaler = DualTargetScaler()
    else:
        target_scaler = SingleTargetScaler()
    
    # Crear y devolver el escalador completo
    return DataScaler(feature_scaler, target_scaler)


def scale_data(X_train, y_train, X_test, y_test, return_scaler=False):
    """
    Escala los datos de forma automática según sus características.
    
    Args:
        X_train, y_train, X_test, y_test: Datos a escalar
        return_scaler: Si True, devuelve también el escalador
        
    Returns:
        tuple: Datos escalados y opcionalmente el escalador
    """
    # Crear el escalador adecuado
    scaler = create_scaler(X_train, y_train)
    
    # Escalar los datos
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = scaler.standardize(
        X_train, y_train, X_test, y_test
    )
    
    if return_scaler:
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
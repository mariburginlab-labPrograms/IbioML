import numpy as np
from sklearn.utils import check_random_state

class TrialKFold():
    """
    Implementación de K-Fold cross-validation que respeta los límites de trials.
    
    Esta clase asegura que todos los time bins pertenecientes a un mismo trial
    permanezcan en el mismo conjunto (entrenamiento o prueba) durante el proceso
    de validación cruzada.
    
    Parámetros:
    -----------
    n_splits : int
        Número de folds. Debe ser al menos 2.
    shuffle : bool, default=False
        Si es True, los trials son mezclados antes de dividirlos.
    random_state : int, RandomState, default=None
        Controla la aleatoriedad de la mezcla.
    """
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, trial_markers):
        """
        Genera índices para dividir los datos en conjuntos de entrenamiento y prueba.
        
        Parámetros:
        -----------
        X : array-like
            Datos de entrenamiento.
        y : array-like, opcional
            Etiquetas.
        trial_markers : array-like, obligatorio
            Array que indica a qué trial pertenece cada muestra.
        
        Retorna:
        --------
        train_indices : array
            Índices de las muestras en el conjunto de entrenamiento.
        test_indices : array
            Índices de las muestras en el conjunto de prueba.
        """
        # Obtenemos los IDs de trials únicos
        unique_trials = np.unique(trial_markers)
        n_trials = len(unique_trials)
        
        if n_trials < self.n_splits:
            raise ValueError(
                f"El número de trials ({n_trials}) debe ser al menos "
                f"igual al número de folds ({self.n_splits})"
            )
        
        # Mezclamos los trials si shuffle=True
        if self.shuffle:
            random_state = check_random_state(self.random_state)
            shuffled_trials = random_state.permutation(unique_trials)
        else:
            shuffled_trials = unique_trials
        
        # Dividimos los trials en n_splits grupos aproximadamente iguales
        trial_folds = np.array_split(shuffled_trials, self.n_splits)
        
        # Generamos los índices para cada fold
        for i in range(self.n_splits):
            # Los trials para el conjunto de prueba son los del fold actual
            test_trials = trial_folds[i]
            # Creamos una máscara para los índices de prueba
            test_mask = np.isin(trial_markers, test_trials)
            test_indices = np.where(test_mask)[0]
            
            # Los trials para entrenamiento son todos los demás
            train_mask = ~test_mask
            train_indices = np.where(train_mask)[0]
            
            yield train_indices, test_indices


def trial_train_test_split(X, y, trial_markers, train_size=0.8, random_state=None, shuffle=True, return_mask=False):
    """
    Dividir los datos en conjuntos de entrenamiento y prueba manteniendo los trials intactos.
    
    Parámetros:
    -----------
    X : array-like
        Datos de entrenamiento.
    y : array-like
        Etiquetas.
    trial_markers : array-like
        Array que indica a qué trial pertenece cada muestra.
    train_size : float, default=0.8
        Proporción de datos para el conjunto de entrenamiento.
    random_state : int, RandomState, default=None
        Controla la aleatoriedad de la mezcla.
    shuffle : bool, default=True
        Si es True, los trials son mezclados antes de dividirlos.
    return_mask : bool, default=False
        Si es True, también retorna las máscaras de índices de entrenamiento y prueba.
    
    Retorna:
    --------
    X_train : array-like
        Datos de entrenamiento.
    X_test : array-like
        Datos de prueba.
    y_train : array-like
        Etiquetas de entrenamiento. 
    y_test : array-like
        Etiquetas de prueba.
    train_mask : array-like, opcional
        Máscara de índices de entrenamiento. Solo se retorna si return_mask=True.
    test_mask : array-like, opcional
        Máscara de índices de prueba. Solo se retorna si return_mask=True.
    """
    # Verificar que train_size esté entre 0 y 1
    if not 0 < train_size < 1:
        raise ValueError("train_size debe estar entre 0 y 1")

    # Obtenemos los IDs de trials únicos
    unique_trials = np.unique(trial_markers)
    n_trials = len(unique_trials)
    
    # Mezclamos los trials si shuffle=True
    if shuffle:
        random_state = check_random_state(random_state)
        shuffled_trials = random_state.permutation(unique_trials)
    else:
        shuffled_trials = unique_trials 
        
    # Calcular número de trials para entrenamiento
    n_train_trials = int(np.ceil(n_trials * train_size))
    
    # Dividir trials en train y test
    train_trials = shuffled_trials[:n_train_trials]
    test_trials = shuffled_trials[n_train_trials:]
    
    # Crear máscaras para los índices
    train_mask = np.isin(trial_markers, train_trials)
    test_mask = np.isin(trial_markers, test_trials)
    
    # Dividir X e y usando las máscaras
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    if return_mask:
        return X_train, X_test, y_train, y_test, train_mask, test_mask
    else:
        return X_train, X_test, y_train, y_test
    
    
        
    
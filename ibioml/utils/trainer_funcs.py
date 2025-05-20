import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
import numpy as np
import copy

#------------------- Initialization of Weights -------------------#
def initialize_weights(model, init_type="he_normal"):
    """
    Initialize the weights of the model.
    
    Args:
        model (nn.Module): The model to initialize.
        init_type (str): The type of initialization to use. Options are "he_normal", "he_uniform", "xavier_normal", "xavier_uniform".
    """
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            match init_type:
                case "he_normal":
                    init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                case "he_uniform":  
                    init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                case "xavier_normal":
                    init.xavier_normal_(layer.weight)
                case "xavier_uniform":
                    init.xavier_uniform_(layer.weight)
                case _:
                    raise ValueError(f"Unknown initialization type: {init_type}")
            
            if layer.bias is not None:
                init.zeros_(layer.bias)


#------------------- Training and Prediction for DL Models -------------------#  


def create_dataloaders(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, batch_size):
    """
    Create the PyTorch DataLoader objects for training and validation.
    Works with both single-output and dual-output targets.
    
    Args:
        X_train_scaled: The scaled training features.
        y_train_scaled: The scaled training target(s). Can be (n_samples, 1) or (n_samples, 2).
        X_val_scaled: The scaled validation features.
        y_val_scaled: The scaled validation target(s).
        batch_size (int): The batch size to use.
        
    Returns:
        tuple: The training and validation DataLoader objects.
    """
    # Asegurar que los datos son tensores PyTorch
    if not isinstance(X_train_scaled, torch.Tensor):
        X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
    else:
        X_train_scaled = X_train_scaled.float()
        
    if not isinstance(y_train_scaled, torch.Tensor):
        y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32)
    else:
        y_train_scaled = y_train_scaled.float()
        
    if not isinstance(X_val_scaled, torch.Tensor):
        X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32)
    else:
        X_val_scaled = X_val_scaled.float()
        
    if not isinstance(y_val_scaled, torch.Tensor):
        y_val_scaled = torch.tensor(y_val_scaled, dtype=torch.float32)
    else:
        y_val_scaled = y_val_scaled.float()

    # Crear los DataLoaders (ya no usamos clone().detach())
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_scaled, y_train_scaled),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Asegurarse de que el último batch no se queda sin datos
    )

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_scaled, y_val_scaled),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def configure_train_config(config, X_train, train_loader, val_loader, split_idx, scaler):
    """
    Configure the training parameters for the model.
    
    Args:
        config (dict): The configuration dictionary.
        X_train (np.ndarray): The training features.
        train_loader (DataLoader): The training DataLoader.
        val_loader (DataLoader): The validation DataLoader.
        split_idx (np.ndarray): The indices of the training and validation split.
        scaler (StandardScaler): The StandardScaler object.
        
    Returns:
        dict: The updated configuration dictionary.
    """
    train_config = config.copy()
    train_config["input_size"] = X_train.shape[X_train.ndim-1]
    train_config["train_loader"] = train_loader
    train_config["test_loader"] = val_loader
    train_config["split_idx"] = split_idx
    train_config["y_scaler"] = scaler.y_scaler
    
    return train_config


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    
    Args:
        patience (int): How many epochs to wait before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        restore_best_weights (bool): Whether to restore the best model weights when stopping.
    """
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
"""
Modelo PINN (Physics-Informed Neural Network) para prever perda de propagação
Modelo prevê Ld (perda por difração) e usa física FSPL para calcular PL = FSPL + Ld
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class PINNPropagationLoss(nn.Module):
    """
    Physics-Informed Neural Network para prever perda de propagação
    
    O modelo prevê Ld (perda por difração) e incorpora física através da loss:
    L_total = L_data + λ * L_physics
    onde L_data = MSE(Ld_real, Ld_pred)
    e L_physics = MSE(PL_real, FSPL + Ld_pred)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 3,
        neurons_per_layer: int = 64,
        learning_rate: float = 0.001,
        lambda_physics: float = 1.0,
        activation: str = 'relu'
    ):
        """
        Args:
            input_dim: Dimensão das features de entrada
            hidden_layers: Número de camadas ocultas
            neurons_per_layer: Número de neurônios por camada oculta
            learning_rate: Taxa de aprendizado
            lambda_physics: Peso do termo físico na loss
            activation: Função de ativação ('relu', 'tanh', 'gelu')
        """
        super(PINNPropagationLoss, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.lambda_physics = lambda_physics
        self.learning_rate = learning_rate
        
        # Definir função de ativação
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        # Construir rede neural
        layers = []
        
        # Camada de entrada
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(self.activation)
        
        # Camadas ocultas
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self.activation)
            layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(nn.Dropout(0.1))
        
        # Camada de saída (prevê Ld - perda por difração)
        layers.append(nn.Linear(neurons_per_layer, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Otimizador será criado no método train
        self.optimizer = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: prevê Ld
        
        Args:
            x: Features normalizadas (batch_size, input_dim)
        
        Returns:
            Ld predito (batch_size, 1)
        """
        return self.network(x)
    
    def physics_loss(
        self,
        ld_pred: torch.Tensor,
        ld_real: torch.Tensor,
        fspl: torch.Tensor,
        pl_real: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula loss física combinada
        
        Args:
            ld_pred: Ld predito pelo modelo
            ld_real: Ld real (PL - FSPL)
            fspl: FSPL calculado
            pl_real: PL real
        
        Returns:
            loss_data: Loss de dados (MSE entre Ld_real e Ld_pred)
            loss_physics: Loss física (MSE entre PL_real e FSPL + Ld_pred)
        """
        # Loss de dados: erro entre Ld real e predito
        loss_data = nn.functional.mse_loss(ld_pred.squeeze(), ld_real)
        
        # Loss física: garante que PL = FSPL + Ld
        pl_pred = fspl + ld_pred.squeeze()
        loss_physics = nn.functional.mse_loss(pl_pred, pl_real)
        
        return loss_data, loss_physics
    
    def train_model(
        self,
        X_train: np.ndarray,
        ld_train: np.ndarray,
        fspl_train: np.ndarray,
        pl_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        ld_val: Optional[np.ndarray] = None,
        fspl_val: Optional[np.ndarray] = None,
        pl_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 1024,
        verbose: bool = True
    ) -> dict:
        """
        Treina o modelo PINN
        
        Args:
            X_train: Features de treino
            ld_train: Ld real de treino
            fspl_train: FSPL de treino
            pl_train: PL real de treino
            X_val: Features de validação (opcional)
            ld_val: Ld real de validação (opcional)
            fspl_val: FSPL de validação (opcional)
            pl_val: PL real de validação (opcional)
            epochs: Número de épocas
            batch_size: Tamanho do batch
            verbose: Se True, imprime progresso
        
        Returns:
            history: Dicionário com histórico de treinamento
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Criar otimizador
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate if hasattr(self, 'learning_rate') else 0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
        # Converter para tensores
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        ld_train_tensor = torch.FloatTensor(ld_train).to(device)
        fspl_train_tensor = torch.FloatTensor(fspl_train).to(device)
        pl_train_tensor = torch.FloatTensor(pl_train).to(device)
        
        history = {
            'train_loss': [],
            'train_loss_data': [],
            'train_loss_physics': [],
            'val_loss': [],
            'val_loss_data': [],
            'val_loss_physics': []
        }
        
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            epoch_loss_data = 0.0
            epoch_loss_physics = 0.0
            
            # Shuffle dados
            indices = torch.randperm(n_samples)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train_tensor[batch_indices]
                ld_batch = ld_train_tensor[batch_indices]
                fspl_batch = fspl_train_tensor[batch_indices]
                pl_batch = pl_train_tensor[batch_indices]
                
                # Forward pass
                ld_pred = self.forward(X_batch)
                
                # Calcular loss
                loss_data, loss_physics = self.physics_loss(
                    ld_pred, ld_batch, fspl_batch, pl_batch
                )
                
                loss_total = loss_data + self.lambda_physics * loss_physics
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()
                
                epoch_loss += loss_total.item()
                epoch_loss_data += loss_data.item()
                epoch_loss_physics += loss_physics.item()
            
            # Média das losses
            epoch_loss /= n_batches
            epoch_loss_data /= n_batches
            epoch_loss_physics /= n_batches
            
            history['train_loss'].append(epoch_loss)
            history['train_loss_data'].append(epoch_loss_data)
            history['train_loss_physics'].append(epoch_loss_physics)
            
            # Validação
            if X_val is not None:
                val_loss, val_loss_data, val_loss_physics = self.evaluate_loss(
                    X_val, ld_val, fspl_val, pl_val, batch_size, device
                )
                history['val_loss'].append(val_loss)
                history['val_loss_data'].append(val_loss_data)
                history['val_loss_physics'].append(val_loss_physics)
                
                scheduler.step(val_loss)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} "
                          f"(Data: {epoch_loss_data:.4f}, Physics: {epoch_loss_physics:.4f}) "
                          f"- Val Loss: {val_loss:.4f}")
            else:
                scheduler.step(epoch_loss)
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} "
                          f"(Data: {epoch_loss_data:.4f}, Physics: {epoch_loss_physics:.4f})")
        
        return history
    
    def evaluate_loss(
        self,
        X: np.ndarray,
        ld_real: np.ndarray,
        fspl: np.ndarray,
        pl_real: np.ndarray,
        batch_size: int = 1024,
        device: Optional[torch.device] = None
    ) -> Tuple[float, float, float]:
        """
        Avalia loss sem atualizar gradientes
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.eval()
        total_loss = 0.0
        total_loss_data = 0.0
        total_loss_physics = 0.0
        
        X_tensor = torch.FloatTensor(X).to(device)
        ld_tensor = torch.FloatTensor(ld_real).to(device)
        fspl_tensor = torch.FloatTensor(fspl).to(device)
        pl_tensor = torch.FloatTensor(pl_real).to(device)
        
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_tensor[start_idx:end_idx]
                ld_batch = ld_tensor[start_idx:end_idx]
                fspl_batch = fspl_tensor[start_idx:end_idx]
                pl_batch = pl_tensor[start_idx:end_idx]
                
                ld_pred = self.forward(X_batch)
                loss_data, loss_physics = self.physics_loss(
                    ld_pred, ld_batch, fspl_batch, pl_batch
                )
                
                loss_total = loss_data + self.lambda_physics * loss_physics
                
                total_loss += loss_total.item()
                total_loss_data += loss_data.item()
                total_loss_physics += loss_physics.item()
        
        return (
            total_loss / n_batches,
            total_loss_data / n_batches,
            total_loss_physics / n_batches
        )
    
    def predict(self, X: np.ndarray, fspl: np.ndarray, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz Ld e PL
        
        Args:
            X: Features normalizadas
            fspl: FSPL calculado
            batch_size: Tamanho do batch
        
        Returns:
            ld_pred: Ld predito
            pl_pred: PL predito (FSPL + Ld_pred)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        X_tensor = torch.FloatTensor(X).to(device)
        fspl_tensor = torch.FloatTensor(fspl).to(device)
        
        ld_preds = []
        
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_tensor[start_idx:end_idx]
                ld_pred_batch = self.forward(X_batch)
                ld_preds.append(ld_pred_batch.cpu().numpy())
        
        ld_pred = np.concatenate(ld_preds, axis=0).squeeze()
        pl_pred = fspl_tensor.cpu().numpy() + ld_pred
        
        return ld_pred, pl_pred
    
    def evaluate(
        self,
        X: np.ndarray,
        pl_real: np.ndarray,
        fspl: np.ndarray,
        batch_size: int = 1024
    ) -> dict:
        """
        Avalia modelo e retorna métricas
        
        Args:
            X: Features normalizadas
            pl_real: PL real
            fspl: FSPL calculado
            batch_size: Tamanho do batch
        
        Returns:
            Dicionário com métricas (RMSE, MAE, R², MAPE)
        """
        ld_pred, pl_pred = self.predict(X, fspl, batch_size)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(pl_real, pl_pred))
        mae = mean_absolute_error(pl_real, pl_pred)
        r2 = r2_score(pl_real, pl_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((pl_real - pl_pred) / pl_real)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
    
    def feature_importance_analysis(
        self,
        X: np.ndarray,
        fspl: np.ndarray,
        feature_names: List[str],
        n_samples: int = 1000
    ) -> dict:
        """
        Análise de importância de features através de sensibilidade
        
        Args:
            X: Features normalizadas
            fspl: FSPL calculado
            feature_names: Nomes das features
            n_samples: Número de amostras para análise
        
        Returns:
            Dicionário com importância de cada feature
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        # Amostrar dados
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
            fspl_sample = fspl[indices]
        else:
            X_sample = X
            fspl_sample = fspl
        
        # Predição base
        ld_pred_base, _ = self.predict(X_sample, fspl_sample)
        
        feature_importance = {}
        
        for i, feature_name in enumerate(feature_names):
            # Perturbar feature
            X_perturbed = X_sample.copy()
            std_dev = np.std(X_sample[:, i])
            X_perturbed[:, i] += 0.1 * std_dev  # Perturbar 10% do desvio padrão
            
            # Predição perturbada
            ld_pred_perturbed, _ = self.predict(X_perturbed, fspl_sample)
            
            # Calcular diferença
            sensitivity = np.mean(np.abs(ld_pred_perturbed - ld_pred_base))
            feature_importance[feature_name] = sensitivity
        
        # Normalizar importância
        max_importance = max(feature_importance.values())
        if max_importance > 0:
            feature_importance = {k: v / max_importance for k, v in feature_importance.items()}
        
        return feature_importance


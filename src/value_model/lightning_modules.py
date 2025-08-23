from typing import Optional, Tuple
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from lightning.pytorch import LightningModule
from torchmetrics import Precision, Recall, F1Score, Accuracy

from src.value_model.model import ValueHeadModel


class H5ValueDataset(Dataset):
    """
    Expects HDF5 with:
      - latent_vectors: (N, num_latent_steps, latent_dim)
      - accuracy_rewards: (N,)
    Returns:
      x: (latent_dim,)
      y: (1,) in [0,1]
    """
    def __init__(self, h5_path: str):
        super().__init__()
        self.h5_path = h5_path
        
        # Preprocess all data in init
        with h5py.File(self.h5_path, "r") as hf:
            latent_vectors = hf["latent_vectors"][:]  # (N, num_latent_steps, latent_dim)
            accuracy_rewards = hf["accuracy_rewards"][:]  # (N,)
            
            # Reshape to treat each embedding as separate sample
            N, num_steps, latent_dim = latent_vectors.shape
            self.latent_dim = latent_dim
            
            # Reshape: (N, num_steps, latent_dim) -> (N*num_steps, latent_dim)
            self.x_data = torch.from_numpy(latent_vectors.reshape(-1, latent_dim)).float()
            
            # Repeat rewards for each step: (N,) -> (N*num_steps,)
            self.y_data = torch.from_numpy(accuracy_rewards.repeat(num_steps)).float().unsqueeze(1)
            
            self._length = self.x_data.shape[0]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class LigthningValueHeadModel(LightningModule):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float, learning_rate: float):
        super().__init__()
        self.model = ValueHeadModel(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
        )
        self.loss_fn = FocalLoss(gamma=2.0)
        self.learning_rate = learning_rate
        
        # Initialize metrics
        self.precision = Precision(task='binary')
        self.recall = Recall(task='binary')
        self.f1 = F1Score(task='binary')
        self.accuracy = Accuracy(task='binary')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Convert logits to probabilities and then to binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Calculate metrics
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        accuracy = self.accuracy(preds, y)

        # Log all metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_precision", precision, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_recall", recall, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_f1", f1, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
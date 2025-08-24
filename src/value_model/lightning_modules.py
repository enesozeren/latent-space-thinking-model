from typing import Optional, Tuple
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from lightning.pytorch import LightningModule
from torchmetrics import Precision, Recall, F1Score, Accuracy, AUROC

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


class LigthningValueHeadModel(LightningModule):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float, learning_rate: float):
        super().__init__()
        self.model = ValueHeadModel(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        
        # Initialize metrics - these will accumulate across batches
        self.train_precision = Precision(task='binary')
        self.train_recall = Recall(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.train_accuracy = Accuracy(task='binary')
        self.train_auroc = AUROC(task='binary')
        
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.val_auroc = AUROC(task='binary')
        
        self.test_precision = Precision(task='binary')
        self.test_recall = Recall(task='binary')
        self.test_f1 = F1Score(task='binary')
        self.test_accuracy = Accuracy(task='binary')
        self.test_auroc = AUROC(task='binary')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Convert logits to probabilities and then to binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        
        # Update metrics (accumulates across batches)
        if stage == "train":
            self.train_precision.update(preds, y.int())
            self.train_recall.update(preds, y.int())
            self.train_f1.update(preds, y.int())
            self.train_accuracy.update(preds, y.int())
            self.train_auroc.update(probs, y.int())
        elif stage == "val":
            self.val_precision.update(preds, y.int())
            self.val_recall.update(preds, y.int())
            self.val_f1.update(preds, y.int())
            self.val_accuracy.update(preds, y.int())
            self.val_auroc.update(probs, y.int())
        elif stage == "test":
            self.test_precision.update(preds, y.int())
            self.test_recall.update(preds, y.int())
            self.test_f1.update(preds, y.int())
            self.test_accuracy.update(preds, y.int())
            self.test_auroc.update(probs, y.int())

        # Log only loss per step (metrics will be logged at epoch end)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        # Compute and log accumulated metrics at epoch end
        self.log("train_precision", self.train_precision.compute(), prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_accuracy", self.train_accuracy.compute(), prog_bar=True)
        self.log("train_roc_auc", self.train_auroc.compute(), prog_bar=True)
        
        # Reset metrics for next epoch
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        self.train_accuracy.reset()
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        # Compute and log accumulated metrics at epoch end
        self.log("val_precision", self.val_precision.compute(), prog_bar=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_accuracy", self.val_accuracy.compute(), prog_bar=True)
        self.log("val_roc_auc", self.val_auroc.compute(), prog_bar=True)
        
        # Reset metrics for next epoch
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_accuracy.reset()
        self.val_auroc.reset()

    def on_test_epoch_end(self):
        # Compute and log accumulated metrics at epoch end
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_roc_auc", self.test_auroc.compute(), prog_bar=True)
        
        # Reset metrics for next epoch
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_accuracy.reset()
        self.test_auroc.reset()
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
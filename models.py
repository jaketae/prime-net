import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score
from torch import nn

from dataset import get_pos_weight


class PriSTM(pl.LightningModule):
    def __init__(self, embed_dim, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.embed = nn.Embedding(2, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        in_features = hidden_size * (bidirectional + 1)
        self.fc = nn.Linear(in_features, 1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weight())

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.flatten(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y_true)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        recall = recall_score(
            y_true.cpu().to(int).numpy(),
            (y_pred > 0.5).cpu().to(int).numpy(),
            zero_division=0,
        )
        self.log("recall", recall, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

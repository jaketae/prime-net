import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

from dataset import get_pos_weight, make_loader


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

    def get_metrics(self, batch):
        x, y_true = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y_true)
        f1 = f1_score(
            y_true.to(int).numpy(), (y_pred > 0.5).to(int).numpy(), zero_division=0
        )
        return loss, f1

    def training_step(self, batch, batch_idx):
        loss, _ = self.get_metrics(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        _, f1 = self.get_metrics(batch)
        self.log("f1", f1)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def train_dataloader(self):
        return make_loader("train")

    def val_dataloader(self):
        return make_loader("val")

    def test_dataloader(self):
        return make_loader("test")
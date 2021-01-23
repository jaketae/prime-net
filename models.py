import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score
from torch import nn

from dataset import get_pos_weight


class PriSTM(pl.LightningModule):
    def __init__(self, embed_dim, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.embed = nn.Embedding(3, embed_dim)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
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
        batch_size = x.size(0)
        x = self.embed(x)
        _, (hidden, _) = self.lstm(x)
        # hidden.shape = (batch, num layers * num directions, hidden_size)
        hidden = hidden.transpose(0, 1)
        # hidden.shape = (num_layers * num_directions, batch, hidden_size)
        hidden = hidden.reshape(self.num_layers, -1, batch_size, self.hidden_size)
        # hidden.shape = (num_layers, num_directions, batch, hidden_Size)
        last_hidden = hidden[-1].transpose(0, 1).reshape(batch_size, -1)
        # last_hidden.shape = (batch, num_directions * hidden_size)
        x = self.fc(last_hidden)
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
        self.log("recall", recall, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

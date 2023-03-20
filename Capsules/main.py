import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.loggers import NeptuneLogger

neptune_logger = NeptuneLogger(
    project="kaori/Capsule",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjZiMDA2YS02MDM3LTQxZjQtOTE4YS1jODZkMTJjNGJlMDYifQ==",
    log_model_checkpoints=False,
)

PARAMS = {
    "batch_size": 32,
    "lr": 0.007,
    "max_epochs": 10,
}

neptune_logger.log_hyperparams(params=PARAMS)


class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("metrics/batch/loss", loss, prog_bar=False)

        y_true = y.tolist()
        y_pred = y_hat.argmax(axis=1).tolist()
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/batch/acc", acc)

        self.training_step_outputs.append({"loss": loss.item(), "y_true": y_true, "y_pred": y_pred})
        return loss

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.log("metrics/epoch/loss", loss.mean())
        self.log("metrics/epoch/acc", acc)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_true = y.tolist()
        y_pred = y_hat.argmax(axis=1).tolist()
        self.validation_step_outputs.append({"loss": loss.item(), "y_true": y_true, "y_pred": y_pred})
    

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)
        self.logger.experiment["val/loss"] = loss.mean()
        self.logger.experiment["val/acc"] = acc
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=PARAMS["lr"])


mnist_model = MNISTModel()

train_ds = MNIST("data", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=PARAMS["batch_size"])

trainer = Trainer(
    logger=neptune_logger,
    max_epochs=PARAMS["max_epochs"],
)
trainer.fit(mnist_model, train_loader)

val_ds = MNIST("data", train=False, download=True, transform=transforms.ToTensor())
val_loader = DataLoader(val_ds, batch_size=PARAMS["batch_size"])
trainer.validate(mnist_model, val_loader)

import numpy as np
import yaml
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from Model import *
from ReadDataset import *
from pytorch_lightning import LightningModule, Trainer


with open("Capsules/config.yaml", 'r') as stream:
    try:
        PARAMS = yaml.safe_load(stream)
        PARAMS = PARAMS['affNist']
    except yaml.YAMLError as exc:
        print(exc)

class CapsuleModel(LightningModule):
    def __init__(self):
        super().__init__()

        if(PARAMS['architect_settings']['shortcut']):
            self.model = CoreArchitect(model_configs=PARAMS['architect_settings'])
        else:
            self.model = CapNets(model_configs=PARAMS['architect_settings'])
        self.loss = SpreadLoss(num_classes=PARAMS['architect_settings']['n_cls'])

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
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
       
        loss = self.loss(y_hat, y)

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

        self.log("val/epoch/loss", loss.mean())
        self.log("val/epoch/acc", acc)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        y_true = y.tolist()
        y_pred = y_hat.argmax(axis=1).tolist()
        self.test_step_outputs.append({"loss": loss.item(), "y_true": y_true, "y_pred": y_pred})
    

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        loss = np.array([])
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            loss = np.append(loss, results_dict["loss"])
            y_true = np.append(y_true, results_dict["y_true"])
            y_pred = np.append(y_pred, results_dict["y_pred"])
        acc = accuracy_score(y_true, y_pred)

        self.logger.experiment["test/loss"] = loss.mean()
        self.logger.experiment["test/acc"] = acc
        self.test_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
      
        y_pred = y_hat.argmax(axis=1).tolist()
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS['training_settings']['lr'])
        # optimizer = torch.optim.SGD(self.parameters(), lr=PARAMS['training_settings']['lr'], 
        #                             weight_decay=PARAMS['training_settings']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=PARAMS['training_settings']['lr_step'], 
                                                    gamma=PARAMS['training_settings']['lr_decay'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
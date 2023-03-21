import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from Model import *
from ReadDataset import *
from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.loggers import NeptuneLogger

neptune_logger = NeptuneLogger(
    project="kaori/Capsule",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjZiMDA2YS02MDM3LTQxZjQtOTE4YS1jODZkMTJjNGJlMDYifQ==",
    log_model_checkpoints=False,
)

PARAMS = {
    "architect_settings": 
    {
        "shortcut": True,
        "n_cls": 10,
        "n_conv": 2,
        "Conv1": {"in": 1,
                "out": 64,
                "k": 5,
                "s": 2,
                "p": 0},
        "Conv2": {"in": 64,
                "out": 128,
                "k": 5,
                "s": 1,
                "p": 0},
        "Conv3": {"in": 128,
                "out": 128,
                "k": 5,
                "s": 2,
                "p": 0},
        "PrimayCaps": {"in": 128,
                    "out":32,
                    "k": 1,
                    "s": 1,
                    "p": 0},
        "n_caps": 2,
        "cap_dims": 4,
        "n_routing": 3,
        "Caps1": {"in": 32,
                "out": 32,
                "k": (3, 3),
                "s": (2, 2)},
        "Caps2": {"in": 32,
                "out": 10,
                "k": (3, 3),
                "s": (1, 1)},
        "Caps3": {"in": 32,
                "out": 10,
                "k": (3, 3),
                "s": (1, 1)},
        "routing": {"type": "dynamic",
                    "params" : [3]}
    },

    "training_settings":
    {
        "lr": 0.001,
        "lr_step": 20,
        "lr_decay": 0.8,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "n_epoch": 2,
        "n_batch": 32,
        "dataset": "Mnist",
        "ckpt_path": "ckpt"
    }
}

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS['training_settings']['lr'])
        # optimizer = torch.optim.SGD(_model.parameters(), lr=training_settings['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=PARAMS['training_settings']['lr_step'], 
                                                    gamma=PARAMS['training_settings']['lr_decay'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":

    ##Data Loader
    if(PARAMS['training_settings']['dataset'] == 'Mnist'):


        Train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
        
            ])
            
        Test_transform = transforms.Compose([
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

        Train_data = Mnistread(mode='train', data_path="data", transform=Train_transform)
        Val_data = Mnistread(mode='val', data_path="data", transform=Test_transform)
        Test_data = Mnistread(mode='test', data_path="data", transform=Test_transform)

    elif(PARAMS['training_settings']['dataset'] == 'FMnist'):
        
        Train_transform = transforms.Compose([
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
            ])
            
        Test_transform = transforms.Compose([
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

        Train_data = FashionMnistread(mode='train', data_path="data", transform=Train_transform)
        Val_data = FashionMnistread(mode='val', data_path="data", transform=Test_transform)
        Test_data = FashionMnistread(mode='test', data_path="data", transform=Test_transform)

    elif(PARAMS['training_settings']['dataset'] == 'affNist'):
        Train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])
            
        Test_transform = transforms.Compose([
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

        Train_data = affNistread(mode="train", data_path="affMnist", transform=Train_transform)
        Val_data =  affNistread(mode="val", data_path="affMnist", transform=Test_transform)
        Test_data = affNistread(mode="test", data_path="affMnist", transform=Test_transform)

    elif(PARAMS['training_settings']['dataset'] == 'SVHN'):

        Train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
        Test_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        Train_data = SVHNread(mode='train', data_path='data', transform=Train_transform)
        Val_data = SVHNread(mode='val', data_path='data', transform=Test_transform)
        Test_data = SVHNread(mode='test', data_path='data', transform=Test_transform)

    elif(PARAMS['training_settings']['dataset'] == 'smallNorb'):

        Train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
            ])
            
        Test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(32),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

        Train_data = SmallNorbread(mode="train", data_path="smallNorb", transform=Train_transform)
        Val_data = SmallNorbread(mode="val", data_path="smallNorb", transform=Train_transform)
        Test_data = SmallNorbread(mode="test", data_path="smallNorb", transform=Test_transform)


    Train_dataloader = DataLoader(dataset=Train_data, batch_size = PARAMS['training_settings']['n_batch'], shuffle=True, num_workers=4)
    Val_dataloader = DataLoader(dataset=Val_data, batch_size = PARAMS['training_settings']['n_batch'], shuffle=False, num_workers=4)
    Test_dataloader = DataLoader(dataset=Test_data, batch_size = PARAMS['training_settings']['n_batch'], shuffle=False, num_workers=4)


    from pytorch_lightning.callbacks import ModelCheckpoint
    torch.set_float32_matmul_precision('medium')

    model = CapsuleModel()

    # (neptune) log hyper-parameters
    neptune_logger.log_hyperparams(params=PARAMS)

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("model", PARAMS['training_settings']["ckpt_path"]), save_top_k=1, monitor='val/epoch/acc', mode="max")

    # (neptune) initialize a trainer and pass neptune_logger
    trainer = Trainer(
        # logger=neptune_logger,
        max_epochs=PARAMS['training_settings']["n_epoch"],
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback]
    )

    # train the model log metadata to the Neptune run
    trainer.fit(model=model, train_dataloaders=Train_dataloader, val_dataloaders=Val_dataloader)

    # test
    trainer.test(model, dataloaders=Test_dataloader)
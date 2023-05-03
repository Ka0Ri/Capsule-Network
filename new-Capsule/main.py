import numpy as np
import yaml
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from Model import *
from ReadDataset import *
from pytorch_lightning import LightningModule, Trainer
from neptune.types import File
from pytorch_lightning.loggers import NeptuneLogger



class CapsuleModel(LightningModule):
    def __init__(self, PARAMS):
        super().__init__()

        self.reconstructed = False
        self.segment = False

        if(PARAMS['architect_settings']['model'] == "convolution"):
            self.model = ConvNeuralNet(model_configs=PARAMS['architect_settings'])
        elif(PARAMS['architect_settings']['model'] == "eff"):
            self.model = EffCapNets(model_configs=PARAMS['architect_settings'])
        elif(PARAMS['architect_settings']['model'] == "shortcut"):
            self.model = ShortcutCapsNet(model_configs=PARAMS['architect_settings'])
        elif(PARAMS['architect_settings']['model'] == "base"):
            self.model = CapNets(model_configs=PARAMS['architect_settings'])
        elif(PARAMS['architect_settings']['model'] == "segment-caps"):
            self.segment = True
            self.model = CapConvUNet(model_configs=PARAMS['architect_settings'])
        elif(PARAMS['architect_settings']['model'] == "segment-conv"):
            self.segment = True
            self.model = ConvUNet(model_configs=PARAMS['architect_settings'])
        else:
            print("Model is not implemented yet")


        if(PARAMS['training_settings']['loss'] == "ce"):                                                                                                                
            self.loss = CrossEntropyLoss(num_classes=PARAMS['architect_settings']['n_cls'])
        elif(PARAMS['training_settings']['loss'] == "spread"):
            self.loss = SpreadLoss(num_classes=PARAMS['architect_settings']['n_cls'])
        elif(PARAMS['training_settings']['loss'] == "margin"):
            self.loss = MarginLoss(num_classes=PARAMS['architect_settings']['n_cls'])
        elif(PARAMS['training_settings']['loss'] == "bce"):
            self.loss = BCE()
        elif(PARAMS['training_settings']['loss'] == "mse"):
            self.loss = MSE()
        elif(PARAMS['training_settings']['loss'] == "dice"):
            self.loss = DiceLoss()
        else:
            print("Loss is not implemented yet")

        if(PARAMS['architect_settings']['reconstructed']):
            self.reconstructed = True
            self.reconstruct_loss = MSE()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if(PARAMS['training_settings']['CAM']):
            
            self.features_blobs = []
            # for CAM visualization
            if(PARAMS['architect_settings']['model'] == "convolution"):
                def hook_feature(module, input, output):
                    self.features_blobs.append(np.array(output.tolist()))
                self.model.caps_layers[0].register_forward_hook(hook_feature)
            else:
                def hook_feature(module, input, output):
                    self.features_blobs.append(np.array(output[0].tolist()))
                self.model.caps_layers[0].register_forward_hook(hook_feature)
                self.model.caps_layers[1].register_forward_hook(hook_feature)

    def forward(self, x, y=None):
        return self.model(x, y)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        if(self.segment):
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            y_true = y.cpu().detach()
            y_pred = y_hat.cpu().detach()
            
        else:
            if(self.reconstructed):
                y_hat, reconstructions = self(x, y)
                loss = self.loss(y_hat, y) + 0.5 * self.reconstruct_loss(reconstructions, x)
            else:
                y_hat = self(x)
                loss = self.loss(y_hat, y)
            
            y_true = y.tolist()
            y_pred = y_hat.argmax(axis=1).tolist()
            self.log("metrics/batch/acc", accuracy_score(y_true, y_pred), prog_bar=False)

        self.log("metrics/batch/loss", loss, prog_bar=False)
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
       
        self.log("metrics/epoch/loss", loss.mean())
        if(self.segment == False):
            self.log("metrics/epoch/acc", accuracy_score(y_true, y_pred))
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
       
        if(self.segment):
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            y_true = y.cpu().detach()
            y_pred = torch.sigmoid(y_hat).cpu().detach()
            self.validation_step_outputs.append({"loss": loss.item(), "y_true": y_true, "y_pred": y_pred})

        else:
            if(self.reconstructed):
                y_hat, reconstructions = self(x, y)
                loss = self.loss(y_hat, y) + 0.1 * self.reconstruct_loss(reconstructions, x)
                y_true = y.tolist()
                y_pred = y_hat.argmax(axis=1).tolist()
                self.validation_step_outputs.append({"loss": loss.item(), "y_true": y_true, "y_pred": y_pred, "reconstructions": reconstructions})

            else:
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
        self.log("val/epoch/loss", loss.mean())
        if(self.segment):
            y_true = make_grid(outputs[0]["y_true"], nrow=int(PARAMS['training_settings']["n_batch"] ** 0.5))
            y_pred = make_grid(outputs[0]["y_pred"], nrow=int(PARAMS['training_settings']["n_batch"] ** 0.5))
            y_true = y_true.numpy().transpose(1, 2, 0)
            y_pred = y_pred.numpy().transpose(1, 2, 0)
            
            self.logger.experiment["val/gt_images"].append(File.as_image(y_true))
            self.logger.experiment["val/outputs"].append(File.as_image(y_pred))
        else:
            self.log("val/epoch/acc", accuracy_score(y_true, y_pred))
            if(self.reconstructed):
                reconstructions = make_grid(outputs[0]["reconstructions"], nrow=int(PARAMS['training_settings']["n_batch"] ** 0.5))
                reconstructions = reconstructions.cpu().numpy().transpose(1, 2, 0)
                self.logger.experiment["val/reconstructions"].append(File.as_image(reconstructions))

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if(self.reconstructed):
            y_hat, reconstructions = self(x, y)
            loss = self.loss(y_hat, y) + 0.1 * self.reconstruct_loss(reconstructions, x)
        else:
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
    
    # def predict_step(self, batch, batch_idx):
    #     x = batch
    #     y_hat = self(x)
      
    #     # y_pred = y_hat.argmax(axis=1)
    #     y_pred = torch.softmax(y_hat, dim=1)
    #     return y_pred.tolist()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=PARAMS['training_settings']['lr'])
        # optimizer = torch.optim.SGD(self.parameters(), lr=PARAMS['training_settings']['lr'], 
        #                             weight_decay=PARAMS['training_settings']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=PARAMS['training_settings']['lr_step'], 
                                                    gamma=PARAMS['training_settings']['lr_decay'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}





if __name__ == "__main__":

    with open("Capsules/config.yaml", 'r') as stream:
        PARAMS = yaml.safe_load(stream)
        PARAMS = PARAMS['config-segment']
        print(PARAMS)
       

    neptune_logger = NeptuneLogger(
        project="kaori/Capsule",
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZjZiMDA2YS02MDM3LTQxZjQtOTE4YS1jODZkMTJjNGJlMDYifQ==",
        log_model_checkpoints=False,
    )

    ##Data Loader
    if(PARAMS['training_settings']['dataset'] == 'Mnist'):


        Train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
            ])
            
        Test_transform = transforms.Compose([
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

        Train_data = Mnistread(mode='train', data_path="data", transform=Train_transform)
        Val_data = Mnistread(mode='val', data_path="data", transform=Test_transform)
        Test_data = Mnistread(mode='test', data_path="data", transform=Test_transform)

    elif(PARAMS['training_settings']['dataset'] == 'FMnist'):
        
        Train_transform = transforms.Compose([
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
            ])
            
        Test_transform = transforms.Compose([
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

        Train_data = FashionMnistread(mode='train', data_path="data", transform=Train_transform)
        Val_data = FashionMnistread(mode='val', data_path="data", transform=Test_transform)
        Test_data = FashionMnistread(mode='test', data_path="data", transform=Test_transform)

    elif(PARAMS['training_settings']['dataset'] == 'affNist'):
        Train_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])
    
        Train_data = affNistread(mode="train", data_path="data/affMnist", transform=Train_transform)
        Val_data =  affNistread(mode="val", data_path="data/affMnist", transform=Train_transform)
        Test_data = affNistread(mode="test", data_path="data/affMnist", transform=Train_transform)
        
    elif(PARAMS['training_settings']['dataset'] == "centerMnist"):

        Train_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])
            
        Test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

        Train_data = affNistread(mode="train", data_path="data/centerMnist", aff=False, transform=Train_transform)
        Val_data =  affNistread(mode="val", data_path="data/centerMnist", aff=False, transform=Test_transform)
        Test_data = affNistread(mode="test", data_path="data/centerMnist", aff=False, transform=Test_transform)
       

    elif(PARAMS['training_settings']['dataset'] == 'SVHN'):

        Train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
        Test_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        Train_data = SVHNread(mode='train', data_path='data', transform=Train_transform)
        Val_data = SVHNread(mode='val', data_path='data', transform=Test_transform)
        Test_data = SVHNread(mode='test', data_path='data', transform=Test_transform)

    elif(PARAMS['training_settings']['dataset'] == 'smallNorb'):

        Train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ColorJitter(brightness=0.5),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
            ])
            
        Test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(32),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize(mean = (0.5,), std = (0.5,))
        ])

        Train_data = SmallNorbread(mode="train", data_path="data/smallNorb", transform=Train_transform)
        Val_data = SmallNorbread(mode="val", data_path="data/smallNorb", transform=Test_transform)
        Test_data = SmallNorbread(mode="test", data_path="data/smallNorb", transform=Test_transform)

    elif(PARAMS['training_settings']['dataset'] == 'CT-scan-dataset'):
        Train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
            # transforms.Normalize((0.5), (0.5))
        ])

        Test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
                    transforms.ToTensor(),# default : range [0, 255] -> [0.0,1.0]
                    # transforms.Normalize((0.5), (0.5))
                ])

        train_ds = LungCTscan(data_dir="data/CT-scan-dataset", transform=Train_transform)
        val_ds = LungCTscan(data_dir="data/CT-scan-dataset", transform=Train_transform)
        Train_data, _ = random_split(train_ds, [200, 67])
        _, Val_data = random_split(val_ds, [200, 67])
        _, Test_data = random_split(val_ds, [200, 67])

    Train_dataloader = DataLoader(dataset=Train_data, batch_size = PARAMS['training_settings']['n_batch'], shuffle=True, num_workers=4)
    Val_dataloader = DataLoader(dataset=Val_data, batch_size = PARAMS['training_settings']['n_batch'], shuffle=False, num_workers=4)
    Test_dataloader = DataLoader(dataset=Test_data, batch_size = PARAMS['training_settings']['n_batch'], shuffle=False, num_workers=4)


    from pytorch_lightning.callbacks import ModelCheckpoint
    torch.set_float32_matmul_precision('medium')

    model = CapsuleModel(PARAMS=PARAMS)

    # (neptune) log hyper-parameters
    neptune_logger.log_hyperparams(params=PARAMS)

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("model", PARAMS['training_settings']["ckpt_path"]), save_top_k=1, monitor='metrics/batch/loss', mode="min")

    # (neptune) initialize a trainer and pass neptune_logger
    trainer = Trainer(
        logger=neptune_logger,
        max_epochs=PARAMS['training_settings']["n_epoch"],
        accelerator="gpu",
        devices=[1],
        callbacks=[checkpoint_callback]
    )

    # train the model log metadata to the Neptune run
    trainer.fit(model=model, train_dataloaders=Train_dataloader, val_dataloaders=Val_dataloader)

    # test
    # trainer.test(model, dataloaders=Test_dataloader)
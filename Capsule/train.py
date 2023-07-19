import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks
import torch.nn.functional as F


from ultis import *
from model import CapsuleWrappingClassifier, CapsuleWrappingSegment

from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from neptune.types import File
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

seed_everything(SEED=42)
torch.set_float32_matmul_precision('medium')

def get_lr_scheduler_config(optimizer, settings):
    '''
    set up learning rate scheduler
    Args:
        optimizer: optimizer
        settings: settings hyperparameters
    Returns:
        lr_scheduler_config: [learning rate scheduler, configuration]
    '''
    if settings['lr_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=settings['lr_step'], gamma=settings['lr_decay'])
    elif settings['lr_scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=settings['lr_step'], gamma=settings['lr_decay'])
    elif settings['lr_scheduler'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001)
    else:
        raise NotImplementedError

    return {
            'scheduler': scheduler,
            'monitor': f'metrics/epoch/val_{settings["metric"]}',
            'interval': 'epoch',
            'frequency': 1,
        }

def get_optimizer(parameters, settings):
    '''
    set up learning optimizer
    Args:
        parameters: model's parameters
        settings: settings hyperparameters
    Returns:
        optimizer: optimizer
    '''
    if settings['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=settings['lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            parameters, lr=settings['lr'], weight_decay=settings['weight_decay'], momentum=settings['momentum'])
    else:
        raise NotImplementedError()

    return optimizer

def get_loss_function(type):
    '''
    set up loss function
    Args:
        settings: settings hyperparameters,
    Returns:
        loss: loss function
    '''
    if type == "ce": 
        loss = nn.CrossEntropyLoss()
    elif type == "nll": 
        loss = nn.NLLLoss()
    elif type == "bce": 
        loss = nn.BCELoss()
    elif type == "mse": 
        loss = nn.MSELoss()
    elif type == "margin":
        loss = MarginLoss()
    else: 
        raise NotImplementedError()

    return loss

def get_gpu_settings(gpu_ids, n_gpu):
    '''
    Get gpu settings for pytorch-lightning trainer:
    Args:
        gpu_ids (list[int])
        n_gpu (int)
    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    '''
    if not torch.cuda.is_available():
        return "cpu", None, None

    if gpu_ids is not None:
        devices = gpu_ids
        strategy = "ddp" if len(gpu_ids) > 1 else 'auto'
    elif n_gpu is not None:
        devices = n_gpu
        strategy = "ddp" if n_gpu > 1 else 'auto'
    else:
        devices = 1
        strategy = 'auto'

    return "gpu", devices, strategy

def get_basic_callbacks(settings):
    '''
    Get basic callbacks for pytorch-lightning trainer:
    Args: 
        settings
    Returns:
        last ckpt, best ckpt, lr callback, early stopping callback
    '''
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    last_ckpt_callback = ModelCheckpoint(
        filename='last_model_{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=None,
    )
    best_ckpt_calllback = ModelCheckpoint(
        filename='best_model_{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=f'metrics/epoch/val_{settings["metric"]}',
        mode='max',
        verbose=True
    )
    if settings['early_stopping']:
        early_stopping_callback = EarlyStopping(
            monitor=f'metrics/epoch/val_{settings["metric"]}',  # Metric to monitor for improvement
            mode='max',  # Choose 'min' or 'max' depending on the metric (e.g., 'min' for loss, 'max' for accuracy)
            patience=10,  # Number of epochs with no improvement before stopping
        )
        return [last_ckpt_callback, best_ckpt_calllback, lr_callback, early_stopping_callback]
    else: 
        return [last_ckpt_callback, best_ckpt_calllback, lr_callback]

def get_trainer(settings, logger) -> Trainer:
    '''
    Get trainer and logging for pytorch-lightning trainer:
    Args: 
        settings: hyperparameter settings
        task: task to run training
    Returns:
        trainer: trainer object
        logger: neptune logger object
    '''
    callbacks = get_basic_callbacks(settings)
    accelerator, devices, strategy = get_gpu_settings(settings['gpu_ids'], settings['n_gpu'])

    trainer = Trainer(
        logger=[logger],
        max_epochs=settings['n_epoch'],
        default_root_dir=settings['ckpt_path'],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
    )
    return trainer

class DataModule(LightningDataModule):
    '''
    Data Module for Train/Val/Test data loadding
    Args: 
        data_settings, training_settings: hyperparameter settings
        transform: data augmentation
    Returns:
        Train/Test/Val data loader
    '''
    def __init__(self, data_settings, training_settings, transform=[None, None]):
        super().__init__()

        self.dataset = data_settings['name']
        self.root_dir = data_settings['path']
        self.img_size = data_settings['img_size']
        self.batch_size = training_settings['n_batch']
        self.num_workers = training_settings['num_workers']

        # assert self.dataset in ["CIFAR10", "LungCT-Scan", "CUB2011", "PennFudan"], "Dataset not supported"
        self.data_class = {
            "CIFAR10": CIFAR10read,
            "CIFAR100": CIFAR100read,
            "LungCT-Scan": LungCTscan,
            "CIFAR10-feats": CIFAR10_feats,
            "CIFAR100-feats": CIFAR100_feats,
            "Caltech101": Caltech101,
            "VOC2012": VOC2012read,
        }
        self.class_list = None
        self.collate_fn = None
        self.train_transform, self.val_transform = transform
        
    def setup(self, stage: str):

        if stage == "fit":
            self.Train_dataset = self.data_class[self.dataset](mode="train", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.train_transform)
            self.Val_dataset = self.data_class[self.dataset](mode="val", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.val_transform)
                
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.Test_dataset = self.data_class[self.dataset](mode="test", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.val_transform)
           
    def train_dataloader(self):
        return DataLoader(self.Train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.Val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.Test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
    
class Model(LightningModule):
    def __init__(self, PARAMS):
        super().__init__()
        self.save_hyperparameters()

        self.architect_settings = PARAMS['architect_settings']
        self.train_settings = PARAMS['training_settings']
        self.dataset_settings = PARAMS['dataset_settings']
        self.num_classes = self.architect_settings['head']['n_cls']
        self.task = PARAMS['task']
        # Model selection
        if(self.task == 'classification'):
            self.model = CapsuleWrappingClassifier(model_configs=self.architect_settings)
            self.train_metrics = torchmetrics.Accuracy(task='multiclass', num_classes= self.num_classes)
            self.valid_metrics = torchmetrics.Accuracy(task='multiclass', num_classes= self.num_classes)
        elif(self.task == 'segmentation'):
            self.model = CapsuleWrappingSegment(model_configs=self.architect_settings)
            self.train_metrics = torchmetrics.Dice(num_classes= self.num_classes)
            self.valid_metrics = torchmetrics.Dice(num_classes= self.num_classes)
        else:
            raise NotImplementedError()
        self.metrics_name = self.train_settings['metric']

    def setup(self, stage: str):
        if stage == "fit":
            # Loss selection
            self.loss = get_loss_function(self.train_settings['loss'])
            self.train_step_outputs = []
            self.validation_step_outputs = []
        elif stage == "test":
            self.loss = get_loss_function(self.train_settings['loss'])
            self.test_step_outputs = []
        elif stage == "predict":
            self.pred_step_outputs = []
    
    def forward(self, x, y=None):
        return self.model(x, y)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
       
        y = y.long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_pred = torch.softmax(y_hat, dim=1)
        self.train_metrics.update(y_pred.cpu(), y.cpu())

        self.log("metrics/batch/train_loss", loss, prog_bar=False)

        return loss

    def on_train_epoch_end(self):

        metrics = self.train_metrics.compute()
        self.log(f"metrics/epoch/train_{self.metrics_name}", metrics)
        self.train_metrics.reset()

    def _shared_eval_step(self, batch, batch_idx):
        x, y, images = batch
        y_hat = self(x)
    
        y = y.long()
        loss = self.loss(y_hat, y)
        y_pred = torch.softmax(y_hat, dim=1)
        images = (images * 255).to(torch.uint8)
            
        return images.cpu(), y_pred.cpu(), y.cpu(), loss.item()

    def validation_step(self, batch, batch_idx):
    
        images, y_pred, targets, loss = self._shared_eval_step(batch, batch_idx)
        self.valid_metrics.update(y_pred, targets)
        self.validation_step_outputs.append({"image": images, "predictions": y_pred, "targets": targets, "loss": loss})
        self.log('metrics/batch/val_loss', loss)

    def on_validation_epoch_end(self):
        loss =[outputs['loss'] for outputs in self.validation_step_outputs]
        self.log('metrics/epoch/val_loss', sum(loss) / len(loss))
        self.log(f"metrics/epoch/val_{self.metrics_name}", self.valid_metrics.compute())

        outputs = self.validation_step_outputs[0]
        images, predictions, targets = outputs["image"], outputs["predictions"], outputs["targets"]
        if(self.task == 'classification'):
            reconstructions = images
        elif(self.task == 'segmentation'):
            classes_masks = predictions.argmax(1) == torch.arange(predictions.shape[1])[:, None, None, None]
            reconstructions = [draw_segmentation_masks(image, masks=mask, alpha=.8)
                               for image, mask in zip(images, classes_masks.swapaxes(0, 1))]
            reconstructions = torch.stack([F.interpolate(img.unsqueeze(0), size=(128, 128))
                                            for img in reconstructions]).squeeze(1)
          
        reconstructions = make_grid(reconstructions, nrow= int(self.train_settings['n_batch'] ** 0.5))
        reconstructions = reconstructions.numpy().transpose(1, 2, 0)
        reconstructions = normalize_image(reconstructions)
        self.logger.experiment["val/reconstructions"].append(File.as_image(reconstructions))

        self.validation_step_outputs.clear()
        self.valid_metrics.reset()


    def test_step(self, batch, batch_idx):

        images, y_pred, targets, loss = self._shared_eval_step(batch, batch_idx)
        self.valid_metrics.update(y_pred, targets) # can reuse valid metrics 
        self.test_step_outputs.append({"image": images, "predictions": y_pred, "targets": targets, "loss": loss})
    
    def on_test_epoch_end(self):
        loss =[outputs['loss'] for outputs in self.test_step_outputs]

        self.log('metrics/epoch/test_loss', sum(loss) / len(loss))
        self.log(f"metrics/epoch/test_{self.metrics_name}", self.valid_metrics.compute())    
     
        self.test_step_outputs.clear()
        self.valid_metrics.reset()
        
        
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.train_settings)
        lr_scheduler_config = get_lr_scheduler_config(optimizer, self.train_settings)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
if __name__ == "__main__":
    
    import yaml
    from pytorch_lightning.loggers import NeptuneLogger
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', '-c', type=str, required=True, help='Config file'
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        PARAMS = yaml.safe_load(stream)
        print(PARAMS)

    neptune_logger = NeptuneLogger(
            project=PARAMS['logger']['project'],
            # with_id="AIS-113",
            # api_key=PARAMS['logger']['api_key'],
            tags=PARAMS['logger']['tags'],
            log_model_checkpoints=False
        )
    neptune_logger.log_hyperparams(params=PARAMS)
    
    #load data
    data = DataModule(PARAMS['dataset_settings'], PARAMS['training_settings'])

    # create model
    model = Model(PARAMS=PARAMS)
    trainer = get_trainer(PARAMS['training_settings'], neptune_logger)
    # train
    trainer.fit(model, data)
    # test
    trainer.test(model, data)
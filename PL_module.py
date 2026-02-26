import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import Precision,Recall
class PLModule(pl.LightningModule):
    def __init__(self,model:torch.nn.Module,lr:float)->None:
        super(PLModule,self).__init__()
        self.save_hyperparameters(ignore=["model","val_dataloader","train_dataloader"])
        self.model=model
        self.loss_func=torch.nn.CrossEntropyLoss()
        self.f1_score=F1Score(task="multiclass",num_classes=3)
        self.accuracy=Accuracy(task="multiclass",num_classes=3)
        self.precision=Precision(task="multiclass",num_classes=3)
        self.recall=Recall(task="multiclass",num_classes=3)
        self.lr=lr
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.trainer.train_dataloader
    def val_dataloader(self):
        return self.trainer.val_dataloaders
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.trainer.test_dataloaders
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.model(x)
    def common_computations(self,pred:torch.Tensor,labels:torch.Tensor)->dict:
        loss=self.loss_func(pred,labels)
        recall=self.recall(pred,labels)
        accuracy=self.accuracy(pred,labels)
        precision=self.precision(pred,labels)
        f1_score=self.f1_score(pred,labels)
        return dict(loss=loss,recall=recall,accuracy=accuracy,precision=precision,f1_score=f1_score)
    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0)->dict:
        images, labels = batch["image"], batch["label"]
        pred = self(images)
        metrics=self.common_computations(pred,labels)
        self.log(name="loss",value=metrics["loss"],prog_bar=True,on_step=True,on_epoch=True)
        for k,v in metrics.items():
            self.log(name=f"train/{k}",value=v,prog_bar=True,on_step=True,on_epoch=True)
        return metrics
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0)->dict:
        images, labels = batch["image"], batch["label"]
        pred = self(images)
        metrics=self.common_computations(pred,labels)
        for k,v in metrics.items():
            self.log(name=f"val/{k}",value=v,prog_bar=True,on_step=True,on_epoch=True)
        return metrics
    def test_step(self,  batch, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT:
        images, labels = batch["image"], batch["label"]
        pred = self(images)
        metrics=self.common_computations(pred,labels)
        for k,v in metrics.items():
            self.log(name=f"val/{k}",value=v,prog_bar=True,on_step=True,on_epoch=True)
        return metrics
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        images = batch["image"]
        logits = self(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions
        }
    def configure_optimizers(self):
        optimizer=torch.optim.Adam(params=self.model.parameters(),lr=self.lr)
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=5)
        return [optimizer],[lr_scheduler]

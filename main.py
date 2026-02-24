import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from DataSet import FootBallDataSet, transforms
from Model import Model
from PL_module import PLModule

# Constants
DATA_DIR = "/home/muhammad/Downloads/football_match_dataset"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20
NUM_CLASSES = 3
NUM_WORKERS = min(4, os.cpu_count() or 1)

class TrainingModule(PLModule):
    """
    Subclassing PLModule to handle the loss function correctly.
    The Model outputs Softmax probabilities, but PLModule uses CrossEntropyLoss (which expects logits).
    We override common_computations to use NLLLoss with log-probabilities.
    """
    def __init__(self, model, lr):
        super().__init__(model, lr)
        self.loss_func = torch.nn.NLLLoss()

    def common_computations(self, pred: torch.Tensor, labels: torch.Tensor) -> dict:
        # pred comes from Model, which applies Softmax.
        # NLLLoss expects log-probabilities.
        log_pred = torch.log(pred + 1e-7) 
        loss = self.loss_func(log_pred, labels)
        
        # Metrics
        recall = self.recall(pred, labels)
        accuracy = self.accuracy(pred, labels)
        precision = self.precision(pred, labels)
        f1_score = self.f1_score(pred, labels)
        
        return dict(loss=loss, recall=recall, accuracy=accuracy, precision=precision, f1_score=f1_score)

def main():
    pl.seed_everything(42)

    # 1. Prepare Dataset
    if not os.path.exists(DATA_DIR):
        print(f"Warning: Data directory {DATA_DIR} does not exist. Please check the path.")

    dataset = FootBallDataSet(images_dir=DATA_DIR, transforms=transforms)
    
    # 2. Split Dataset
    train_set_length = int(len(dataset) * 0.8)
    val_set_length = len(dataset) - train_set_length
    
    # Handle case where dataset might be empty if path is wrong
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    train_dataset, val_dataset = random_split(
        dataset=dataset, 
        lengths=[train_set_length, val_set_length],
        generator=torch.Generator().manual_seed(42)
    )

    # 3. DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 4. Model & Lightning Module
    model = Model(num_classes=NUM_CLASSES)
    pl_module = TrainingModule(model=model, lr=LEARNING_RATE)

    # 5. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoints',
        filename='football-action-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=5,
        mode='min'
    )

    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10
    )

    # 7. Start Training
    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()

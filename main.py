import os, torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from DataSet import FootBallDataSet
from Model import Model
from PL_module import PLModule
import albumentations as A

# Constants
DATA_DIR = "/home/muhammad/Downloads/football_match_dataset"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20
NUM_CLASSES = 3
NUM_WORKERS = min(4, os.cpu_count() or 1)
def main():
    pl.seed_everything(42)
    if not os.path.exists(DATA_DIR):
        print(f"Warning: Data directory {DATA_DIR} does not exist. Please check the path.")
    transforms=A.Compose(
        transforms=[
            A.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0],p=1.0,normalization="min_max_per_channel"),
            A.Resize(height=224,width=224,p=1.0),
            A.pytorch.ToTensorV2()
        ]
    )
    dataset = FootBallDataSet(images_dir=DATA_DIR, transforms=transforms)
    train_set_length = int(len(dataset) * 0.8)
    val_set_length = len(dataset) - train_set_length
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
    pl_module = PLModule(model=model, lr=LEARNING_RATE)

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
        log_every_n_steps=10,
        fast_dev_run=3
    )

    # 7. Start Training
    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()

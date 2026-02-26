import os, torch,argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from DataSet import FootBallDataSet
from Model import Model
from PL_module import PLModule
import albumentations as A
def main(DATA_DIR:str,BATCH_SIZE:int,LEARNING_RATE:float,MAX_EPOCHS:int,NUM_WORKERS:int,NUM_CLASSES:int=3):
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

    model = Model(num_classes=NUM_CLASSES)
    pl_module = PLModule(model=model, lr=LEARNING_RATE)
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="football_actions",
        default_hp_metric=False,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/val_loss',
        dirpath='checkpoints',
        filename='football-action-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    early_stopping = EarlyStopping(
        monitor='val/val_loss',
        patience=5,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices="auto",
        logger=logger,
        log_every_n_steps=10,
        fast_dev_run=0
    )

    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--bs",type=int,default=32)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--data_dir",type=str,default=None,required=True)
    args=parser.parse_args()
    main(DATA_DIR=args.data_dir,BATCH_SIZE=args.bs,LEARNING_RATE=args.lr,MAX_EPOCHS=args.epochs,NUM_WORKERS= args.num_workers)

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl

from data.dataloader import build_dataloader
from model.cvae import cVAE
from utils import GenerateCallback
from model.monet import MONet


def main():
    train_dl, val_dl = build_dataloader(
        batch_size=64, n_steps=1,
        num_workers=4,
        path="data/datasets/", dataset_class='vor'
    )

    # model = cVAE(out_dim=3)
    model = MONet(6, beta=0.5, gamma=0.5)

    sample_imgs = next(iter(train_dl))
    callbacks = [GenerateCallback(sample_imgs),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(monitor='loss/val'),
    ]
    # logger = WandbLogger(project='monet', log_model='all')
    # logger.watch(model)
    trainer = pl.Trainer(gpus=1, callbacks=callbacks, max_epochs=10000,
        # logger=logger,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
    )

    trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    main()

import torch
from modules.efficientPS import EfficientPS
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import Cityscapes
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser, Namespace

def main(hparams: Namespace):
    model = EfficientPS(num_classes=19, batch_size=hparams.batch_size)

    if hparams.checkpoint:
        trainer = pl.Trainer(resume_from_checkpoint="./checkpoints/"+hparams.checkpoint)
    
    else:
        logger = False
        if hparams.log_wandb:
            logger = WandbLogger(name='first_log',project='EfficientPS')

            # optional: log model topology
            # logger.watch(model, log_freq=100)

        trainer = pl.Trainer(
            gpus=hparams.gpus,
            logger=logger,
            max_epochs=hparams.epochs,
            gradient_clip_val=0.5,
        )

    
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="number of available GPUs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--log_wandb", action='store_true', help="log training on Weights & Biases")
    parser.add_argument("--checkpoint", help="checkpoint name if starting from checkpoint")

    hparams = parser.parse_args()

    main(hparams)
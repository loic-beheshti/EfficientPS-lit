import torch
from modules.efficientPS import EfficientPS
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import Cityscapes
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser, Namespace
#from datasets.cityscapes_transforms import cityscapesTransforms


def main(hparams: Namespace):
    model = EfficientPS(num_classes=19)
    #dataset = Cityscapes('./data/Cityscapes', split='train', mode='fine', target_type='semantic', transforms=cityscapesTransforms())
    #img, smnt = dataset[0]
    #print("img, labels", img, smnt)
    logger = False
    if hparams.log_wandb:
        logger = WandbLogger()

        # optional: log model topology
        logger.watch(model.net)


    trainer = pl.Trainer(
        gpus=hparams.gpus,
        logger=logger,
        max_epochs=hparams.epochs,
    )

    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="number of available GPUs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--log_wandb", action='store_true', help="log training on Weights & Biases")

    hparams = parser.parse_args()

    main(hparams)
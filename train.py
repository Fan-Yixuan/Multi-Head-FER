import argparse

import pytorch_lightning as pl
import timm
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from data.raf_db_basic import raf_db_basic

parser = argparse.ArgumentParser(description='FER')
parser.add_argument('--lr', type=float, help='base learning rate')
parser.add_argument('--gamma', type=float, help='gamma')
parser.add_argument('--bs', type=int, help='bs')
args = parser.parse_args()


class head(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(192, 192)
        self.act = nn.SiLU()
        self.cls = nn.Linear(192, 7)

    def forward(self, x):
        return self.cls(self.act(self.l1(x)))


class model_top(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.net = timm.create_model('convnext_base', pretrained=False, num_classes=7, drop_path_rate=0.25)
        self.moe_head = nn.ModuleList([head(), head(), head(), head()])
        self.lr = lr
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.loss = LabelSmoothingCrossEntropy()
        self.load_finetune_checkpoint('convnext-base-glint-mini.ckpt')

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x = self.net.forward_features(x)
        x = self.net.head[:-1](x)
        loss = 0
        preds = torch.zeros(x.shape[0], 7, device=self.device)
        for cnt_moe in range(4):
            pred = self.moe_head[cnt_moe](x[:, cnt_moe * 192:cnt_moe * 192 + 192])
            loss += self.loss(pred, y) / 4
            preds += pred
        preds = torch.argmax(preds, dim=1)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy(preds, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x = self.net.forward_features(x)
        x = self.net.head[:-1](x)
        loss = 0
        preds = torch.zeros(x.shape[0], 7, device=self.device)
        for cnt_moe in range(4):
            pred = self.moe_head[cnt_moe](x[:, cnt_moe * 192:cnt_moe * 192 + 192])
            loss += self.loss(pred, y) / 4
            preds += pred
        preds = torch.argmax(preds, dim=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy(preds, y), prog_bar=True)
        return 0

    def setup(self, stage=None):
        self.train_dataset = raf_db_basic(raf_path='datasets/raf_db_basic',
                                          phase='train',
                                          transform=T.Compose([
                                              T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
                                              T.RandomApply([T.RandomRotation(5)], p=0.2),
                                              create_transform(input_size=224,
                                                               is_training=True,
                                                               scale=(1.0, 1.0),
                                                               ratio=(1.0, 1.0),
                                                               hflip=0.5,
                                                               vflip=0.,
                                                               color_jitter=0.4,
                                                               auto_augment=None,
                                                               interpolation='bilinear',
                                                               re_prob=0.25,
                                                               re_mode='pixel',
                                                               re_count=1,
                                                               re_num_splits=0)
                                          ]))
        self.val_dataset = raf_db_basic(raf_path='datasets/raf_db_basic',
                                        phase='val',
                                        transform=T.Compose([
                                            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ]))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, num_workers=8, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            },
        }

    def load_finetune_checkpoint(self, path):
        m = torch.load(path)['state_dict']
        model_dict = self.state_dict()
        for k in m.keys():
            if m[k].shape != model_dict[k].shape:
                print('drop ' + str(k))
                continue
            if k in model_dict:
                model_dict[k] = m[k].clone().to(model_dict[k].device)
        self.load_state_dict(model_dict)


pl.seed_everything(42)
model = model_top(lr=args.lr)
trainer = pl.Trainer(
    gpus=1,
    # strategy="ddp",
    max_epochs=90,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    auto_lr_find=True,
    stochastic_weight_avg=False,
    accumulate_grad_batches=2,
    callbacks=[ModelCheckpoint(monitor='val_acc', mode='max', save_weights_only=True, save_top_k=1)],
    logger=TensorBoardLogger(save_dir='logs', name="experiment", version='0'))
trainer.fit(model)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import timm
import torch
import torchvision.transforms as T
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.raf_db_basic import raf_db_basic


class head(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(192, 192)
        self.act = nn.SiLU()
        self.cls = nn.Linear(192, 7)

    def forward(self, x):
        return self.cls(self.act(self.l1(x)))


class model_top(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        self.net = timm.create_model('convnext_base', pretrained=False, num_classes=7)
        self.moe_head = nn.ModuleList([head(), head(), head(), head()])
        self.load_finetune_checkpoint(weight_path)

    def forward(self, x):
        x = self.net.forward_features(x)
        x = self.net.head[:-1](x)
        preds = torch.zeros(x.shape[0], 7).cuda()
        pred_sep = torch.zeros(4, x.shape[0], 7).cuda()
        for cnt_moe in range(4):
            pred = self.moe_head[cnt_moe](x[:, cnt_moe * 192:cnt_moe * 192 + 192])
            preds += nn.Softmax(dim=-1)(pred)
            pred_sep[cnt_moe, :, :] = nn.Softmax(dim=-1)(pred)
        final_preds = torch.argmax(preds, dim=1)
        return final_preds, preds, pred_sep

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
        print('LOAD')


x_label = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
model = model_top('convnext-base-4expert-90.97.ckpt').cuda()
model.eval()
val_dataset = raf_db_basic(raf_path='datasets/raf_db_basic',
                           phase='val',
                           transform=T.Compose([
                               T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ]))
val_loader = DataLoader(val_dataset, batch_size=256, num_workers=8, pin_memory=True, shuffle=True)
total_pred = torch.Tensor([]).int().cuda()
total_y = torch.Tensor([]).int().cuda()
with torch.no_grad():
    for batch in tqdm(val_loader):
        x, y, img_path = batch
        x = x.cuda()
        y = y.cuda()
        pred_stu, pred_log_stu, pred_sep_stu = model(x)
        pred = torch.argmax(pred_log_stu, dim=1)
        total_pred = torch.cat((total_pred, pred))
        total_y = torch.cat((total_y, y))
        wrong_idx = (y != pred)
        # for cnt, cnt_path in enumerate(img_path):
        #     if wrong_idx[cnt]:
        #         plt.figure(figsize=(12, 5))
        #         plt.subplot(1, 3, 1)
        #         plt.imshow(mpimg.imread(cnt_path))
        #         plt.axis('off')
        #         plt.title(label_exp[int(y[cnt])], fontsize=16)
        #         plt.subplot(1, 3, 2)
        #         x = np.arange(7)
        #         width = 0.1
        #         plt.bar(x - 1.5 * width, pred_sep[0, cnt, :].cpu().numpy(), width, label='expert_1')
        #         plt.bar(x - 0.5 * width, pred_sep[1, cnt, :].cpu().numpy(), width, label='expert_2')
        #         plt.bar(x + 0.5 * width, pred_sep[2, cnt, :].cpu().numpy(), width, label='expert_3')
        #         plt.bar(x + 1.5 * width, pred_sep[3, cnt, :].cpu().numpy(), width, label='expert_4')
        #         plt.xticks(x, labels=x_label, rotation=45, fontsize=12)
        #         plt.legend()
        #         plt.subplot(1, 3, 3)
        #         x = np.arange(7)
        #         width = 0.2
        #         plt.bar(x, pred_log[cnt, :].cpu().numpy(), width)
        #         plt.xticks(x, labels=x_label, rotation=45, fontsize=12)
        #         plt.savefig('wrong_samples/' + cnt_path.split('/')[-1].replace('jpg', 'png'), dpi=300)
        #         plt.close()
print(sum(total_pred == total_y) / len(total_pred))
print(len(total_pred))

cm = confusion_matrix(y_true=total_y.cpu(), y_pred=total_pred.cpu())
df = pd.DataFrame(cm, index=x_label, columns=x_label)
sn.heatmap(df, annot=True, fmt='g', cmap='Purples', norm=LogNorm())
plt.tight_layout()
plt.savefig('cm.png', dpi=300)
plt.close()

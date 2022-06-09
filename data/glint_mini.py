import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class glint_mini(Dataset):
    def __init__(self, glint_path, phase, transform=None):
        self.transform = transform

        df = pd.read_csv(os.path.join(glint_path, 'Glint-Mini.list'), sep=' ', header=None, names=['name', 'label'])
        file_names = df.loc[:, 'name'].values
        labels = df.loc[:, 'label'].values

        _, sample_counts = np.unique(labels, return_counts=True)
        file_names = file_names[sample_counts[labels] > 10]
        labels = labels[sample_counts[labels] > 10]

        self.image = []
        self.label = []
        cur_label = 0
        cur_act_label = 0
        cnt_image = 0
        for cnt, f in tqdm(enumerate(file_names)):
            if labels[cnt] != cur_act_label:
                cur_act_label = labels[cnt]
                cur_label += 1
                cnt_image = 0
            cnt_image += 1
            if phase == 'train' and cnt_image <= 3:
                continue
            if phase == 'val' and cnt_image > 3:
                continue
            path = os.path.join(glint_path, f)
            # self.image.append(Image.open(path).convert('RGB'))
            self.image.append(path)
            self.label.append(cur_label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.image[idx]).convert('RGB')), self.label[idx]

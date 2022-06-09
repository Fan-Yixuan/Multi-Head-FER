import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class raf_db_basic(Dataset):
    def __init__(self, raf_path, phase, transform=None):
        self.transform = transform

        df = pd.read_csv(os.path.join(raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name', 'label'])
        if phase == 'train':
            data = df[df['name'].str.startswith('train')]
        else:
            data = df[df['name'].str.startswith('test')]

        file_names = data.loc[:, 'name'].values
        self.label = data.loc[:, 'label'].values - 1
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, sample_counts = np.unique(self.label, return_counts=True)
        print(f' distribution of {phase} samples: {sample_counts}')

        self.image = []
        self.image_path = []
        for f in tqdm(file_names):
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(raf_path, 'Image/aligned', f)
            self.image.append(Image.open(path).convert('RGB'))
            self.image_path.append(path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.transform(self.image[idx]), self.label[idx], self.image_path[idx]

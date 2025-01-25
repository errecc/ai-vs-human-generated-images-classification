import pytorch_lightning
import opendatasets as od
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torchvision.transforms as T


class AiVsHumanDataset(Dataset):
    def __init__(self, test = False, path = "ai-vs-human-generated-dataset"):
        self.test = test
        if test:
            file_path = os.path.join(path, "test.csv")
            data = pd.read_csv(file_path)
            self.paths = [os.path.join(path, p) for p in data["id"]]
        else:
            file_path = os.path.join(path, "train.csv")
            data = pd.read_csv(file_path)
            self.paths = [os.path.join(path, p) for p in data["file_name"]]
            self.labels = [i for i in data["label"]]
        # Define transform for images
        self.transform = T.Compose([T.Resize(512),
                                    T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        X = self.transform(image)
        if self.test:
            return X, None
        return X, self.labels[idx]


dataset = AiVsHumanDataset()
# Download the data
od.download("https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset")

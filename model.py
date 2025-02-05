import pytorch_lightning as pl
import opendatasets as od
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torchvision.transforms as T
import torchvision
from torchvision.models import ResNet18_Weights


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
        self.transform = T.Compose([T.CenterCrop(224),
                                    #T.Resize([64, 3]),
                                    T.Grayscale(num_output_channels = 3),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        X = self.transform(image)
        if self.test:
            return X, None
        return X, self.labels[idx]


class AiVsHumanClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.pretrainded_model = torchvision.models.resnet50(pretrained=True)
        # freeze layers
        self.pretrainded_model.eval()
        for param in self.pretrainded_model.parameters():
            param.requires_grad = False
            print(param.shape)
        self.pretrainded_model.fc = torch.nn.Linear(2048, 2)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.pretrainded_model.parameters())
        return optim

    def forward(self, x):
        return self.pretrainded_model(x)

    def training_step(self, batch, batch_idx):
        inputs, label = batch
        out = self(inputs)
        loss = self.loss_fn(out, label)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


# Download the data
od.download("https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset")
dataset = AiVsHumanDataset()
dataloader = DataLoader(dataset, shuffle=True)
model = AiVsHumanClassifier()
trainer = pl.Trainer()
trainer.fit(model, dataloader)

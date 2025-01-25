import pytorch_lightning
import opendatasets as od
import torch
from PIL import Image
import numpy as np



# Download the data
od.download("https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset")

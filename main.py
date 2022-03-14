import os
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import transformers
import requests
import torch.nn as nn
from feature_extractor import extract_feature
import pandas as pd
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
import cv2

#pd.options.mode.chained_assignment = None  # default='warn'
path = 'E:/pictures/glasses1.jpg'
preprocess = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
initial_model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = nn.Sequential(*list(initial_model.children())[0:-1])
new_feature = extract_feature(model= model, preprocess= preprocess, url= path)
dataset = pd.read_csv('path_feature_score.csv', converters={'feature': literal_eval})
for i in range(len(dataset['file_name'])):
    dataset.loc[i,('score')] = (np.array(dataset.loc[i,('feature')][0])* np.array(new_feature.detach())).sum()

dataset.sort_values('score', ascending=False, inplace=True)
dataset= dataset.reset_index(drop = True)

row, column = 3, 3
k = row* column
fig = plt.figure(figsize=(10, 7))
for i in range(k):
    fig.add_subplot(row, column, i+1)
    image = cv2.imread(dataset.loc[i, 'file_name'])
    plt.imshow(image)
    plt.axis('off')

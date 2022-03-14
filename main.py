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

#pd.options.mode.chained_assignment = None  # default='warn'
path = 'E:/pictures/woman1.jpg'
preprocess = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
initial_model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = nn.Sequential(*list(initial_model.children())[0:-1])
new_feature = extract_feature(model= model, preprocess= preprocess, url= path)
dataset = pd.read_csv('path_feature_score.csv', converters={'feature': literal_eval})
for i in range(len(dataset['file_name'])):
    dataset.loc[i,('score')] = (np.array(dataset.loc[i,('feature')][0])* np.array(new_feature.detach())).sum()


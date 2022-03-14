
import os
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import transformers
import requests
import torch.nn as nn
from feature_extractor import extract_feature
import pandas as pd


preprocess = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
initial_model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = nn.Sequential(*list(initial_model.children())[0:-1])

path = 'E:/pictures'
files = os.listdir(path)
df = pd.DataFrame(columns=['file_name', 'feature', 'score'])
for i, file in enumerate(files):
    url = path+ '/'+ file
    feature = extract_feature(model= model, preprocess= preprocess, url= url)
    df = df.append({'file_name': url, 'feature': feature.detach().numpy(), 'score': 0 }, ignore_index= True)

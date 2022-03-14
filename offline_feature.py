
import os
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import transformers
import requests
import torch.nn as nn
from feature_extractor import CBIR
import pandas as pd


cbir = CBIR()
path = 'E:/pictures'
files = os.listdir(path)
df = pd.DataFrame(columns=['file_name', 'feature', 'score'])

for i, file in enumerate(files):
    url = path+ '/'+ file
    feature = cbir.get_feature(path= url)
    df = df.append({'file_name': url, 'feature': feature.detach().numpy().tolist(), 'score': 0 }, ignore_index= True)

df.to_csv('E:/codes_py/CBIR_with_BEIT/path_feature_score.csv', index = False)

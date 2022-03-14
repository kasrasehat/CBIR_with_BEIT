from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import transformers
import requests
import torch.nn as nn
import torch


def extract_feature(model, preprocess, url):
    image = Image.open(url)
    inputs = preprocess(images=image, return_tensors="pt")
    output = model(inputs['pixel_values'])
    features = output['pooler_output']/torch.linalg.norm(output['pooler_output'])
    return features
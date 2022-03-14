from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import torch.nn as nn
import torch


class CBIR():
    def __init__(self):
        self.model = nn.Sequential(*list(BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k').children())[0:-1])
        self.preprocess = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

    def get_feature(self, path):
        image = Image.open(path)
        inputs = self.preprocess(images=image, return_tensors="pt")
        output = self.model(inputs['pixel_values'])
        features = output['pooler_output'] / torch.linalg.norm(output['pooler_output'])
        return features






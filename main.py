from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import transformers
import requests
import torch.nn as nn
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
new_model = nn.Sequential(*list(model.children())[0:-1])
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
outputs1 = new_model(inputs['pixel_values'])
features = outputs1['pooler_output']
#import things

import yaml
import os
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11s.pt')  # load a pretrained model
# Train the model
#model result are in run/detect/train 
results = model.train(data='Salmon-salmon-yo/data.yaml', epochs=30)
results = model.val()
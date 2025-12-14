#import things

import yaml
import os
from ultralytics import YOLO

# Load a model
model = YOLO('yolo11s.pt')  # load a pretrained model
# Train the model
#model result and model is saved in run/detect/train 
results = model.train(data='Salmon-salmon-yo/data.yaml', epochs=30)
results = model.val()



#testing sample image 
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
%matplotlib inline

# Load your trained YOLO model
model = YOLO('runs/detect/train7/weights/best.pt')

# Load test data
with open("Salmon-salmon-co/test/_annotations.coco.json", 'r') as f:
    test_data = json.load(f)

# Run inference on one image
for img_info in test_data['images'][1:10]:
    img_path = os.path.join("Salmon-salmon-co/test", img_info['file_name'])
    
    # Run inference
    results = model(img_path, conf=0.5)
    result = results[0]  # easier reference
    
    print(f"\n{img_info['file_name']}:")
    print(f"  Detected {len(result.boxes)} objects")
    
    # Manual plotting
    image = Image.open(img_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(
            x1, y1 - 5,
            f'Class {cls}: {conf:.2f}',
            bbox=dict(facecolor='red', alpha=0.5),
            fontsize=10, color='white'
        )

    ax.axis('off')
    plt.title(f"Predictions: {img_info['file_name']}")
    plt.tight_layout()
    plt.show()

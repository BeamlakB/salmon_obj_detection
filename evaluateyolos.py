
#Register dataset as torchvision CocoDetection
import torchvision
import os
from torch.utils.data import DataLoader

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super().__getitem__(idx)

        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

from transformers import AutoFeatureExtractor
def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['labels'] = labels
  return batch
#load dataset 
dataset="/umbc/rs/pi_bbekele1/users/bbekele1/CVyolo/Salmon-salmon-co"
feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=512, max_size=864)

test_dataset = CocoDetection(img_folder=(dataset + '/test'), feature_extractor=feature_extractor)
print("Number of training examples:", len(test_dataset))
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=4)

"""

#Check loading
import numpy as np
import os
from PIL import Image, ImageDraw
#proper laoding of image 
image_ids = test_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = test_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join(dataset + '/test', image['file_name']))

annotations = test_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = test_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
  box = annotation['bbox']
  class_idx = annotation['category_id']
  x,y,w,h = tuple(box)
  draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
  draw.text((x, y), id2label[class_idx], fill='white')

image
"""


#initalize model
import pytorch_lightning as pl
from transformers import DetrConfig, AutoModelForObjectDetection
import torch

#loding the yolos model 
class YoloS(pl.LightningModule):

     def __init__(self, lr, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small", 
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         self.lr = lr
         self.weight_decay = weight_decay
         self.save_hyperparameters()  # adding this will save the hyperparameters to W&B too

     def forward(self, pixel_values):
       outputs = self.model(pixel_values=pixel_values)

       return outputs
       #return self.model(**inputs)
     
     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("train/loss", loss)  # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace
        for k,v in loss_dict.items():
          self.log("train/" + k, v.item())  # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss) # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace
        for k,v in loss_dict.items():
          self.log("validation/" + k, v.item()) #  logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace

        return loss

     def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer
     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader




#load model
import torch
from tqdm import tqdm
from coco_eval import CocoEvaluator
from transformers import DetrFeatureExtractor  # or DetrImageProcessor
import supervision as sv
from transformers import AutoModelForObjectDetection
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
# Re-initialize the model
model = YoloS(lr=2.5e-5, weight_decay=1e-4)

# Load the saved state dict
model.load_state_dict(torch.load("yolos_lightning_model.ckpt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(DEVICE)




ANNOTATION_FILE_NAME = "_annotations.coco.json"


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor, 
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

#helper 
def coco_to_detections(coco_anns):
    boxes = []
    class_ids = []

    for ann in coco_anns:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])      # xyxy format
        class_ids.append(ann["category_id"])

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=np.int32)
    )





#inital test
import random
import cv2
import numpy as np
import supervision as sv

#one predection
# utils
categories = test_dataset.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()
CONFIDENCE_TRESHOLD=0.5
# select random image
image_ids = test_dataset.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# load image and annotatons 
image = test_dataset.coco.loadImgs(image_id)[0]
annotations = test_dataset.coco.imgToAnns[image_id]
image_path = os.path.join(test_dataset.root, image['file_name'])
image = cv2.imread(image_path)

# annotate
#detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
detections = coco_to_detections(annotations)
labels = [
    f"{id2label[class_id]}" 
    for _, _,_,class_id, _,_ 
    in detections
]
#labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
frame = box_annotator.annotate(scene=image.copy(), detections=detections)

print('ground truth')
get_ipython().run_line_magic('matplotlib', 'inline')
sv.plot_image(frame, (16, 16))

# inference
with torch.no_grad():

    # load image and predict
    inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
    outputs = model(**inputs)

    # post-process
    target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
    results = image_processor.post_process_object_detection(
        outputs=outputs, 
        threshold=CONFIDENCE_TRESHOLD, 
        target_sizes=target_sizes
    )[0]

# annotate
detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)
print (detections)
#print (id2label[16])
#labels = [f"{id2label[class_id]} {confidence:.2f}" for _,_, confidence, class_id, _,_ in detections]
frame = box_annotator.annotate(scene=image.copy(), detections=detections)

print("detections")
get_ipython().run_line_magic('matplotlib', 'inline')
sv.plot_image(frame, (16, 16))


# In[77]:


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results




import torch
from tqdm import tqdm
from coco_eval import CocoEvaluator
from transformers import DetrFeatureExtractor  # or DetrImageProcessor
import supervision as sv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(DEVICE)

# COCO categories
categories = test_dataset.coco.cats
id2label = {k: v['name'] for k, v in categories.items()}
print ("label", id2label)
# Initialize COCO evaluator

coco_gt = test_dataset.coco  # This is a pycocotools.COCO object
evaluator = CocoEvaluator(coco_gt=coco_gt, iou_types=["bbox"])

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for img_id, pred in predictions.items():
        if len(pred["boxes"]) == 0:
            continue
        boxes = convert_to_xywh(pred["boxes"]).tolist()
        scores = pred["scores"].tolist()
        labels = pred["labels"].tolist()
        coco_results.extend(
            [
                {"image_id": img_id, "category_id": labels[k], "bbox": box, "score": scores[k]}
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

print("Running evaluation...")

for batch in tqdm(test_dataloader):
    pixel_values = batch["pixel_values"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
    #print (labels)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    orig_sizes = torch.stack([t["orig_size"] for t in labels], dim=0)
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=orig_sizes
    )
    #print (results)
    predictions = {
        int(t["image_id"].item()): r
        for t, r in zip(labels, results)
    }
    #print (predictions)
    # for x in predictions:
    #     #print (predictions[x]["labels"])
    #     #predictions[x]["labels"] = t
    #     predictions[x]['labels'] = torch.full_like(predictions[x]['labels'], 1)
    coco_predictions = prepare_for_coco_detection(predictions)
    print (coco_predictions)
    if coco_predictions:  # only update if there are predictions
        evaluator.update(coco_predictions)
    #evaluator.update(coco_predictions)

# Finalize evaluation
evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()



map50 = evaluator.coco_eval['bbox'].stats[1]  # index 1 is mAP@0.5
print(f"mAP@0.5: {map50:.4f}")




#Each class evalution 

from pycocotools.cocoeval import COCOeval
import numpy as np

coco_eval = evaluator.coco_eval["bbox"]

# precision shape:
# [TxRxKxAxM]
# T = IoU thresholds
# R = recall thresholds
# K = number of classes
# A = area ranges
# M = max detections
precision = coco_eval.eval["precision"]

cat_ids = coco_eval.params.catIds  # [1, 2]
class_names = id2label
for idx, cat_id in enumerate(cat_ids):
    # AP averaged over IoUs, recalls, area=all, maxDets=100
    precision_k = precision[:, :, idx, 0, -1]
    precision_k = precision_k[precision_k > -1]

    ap = np.mean(precision_k) if precision_k.size else float("nan")

    print(f"AP for {class_names[cat_id]}: {ap:.4f}")


print(model.hparams)
import torch
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")


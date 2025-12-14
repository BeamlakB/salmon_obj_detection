#converted from notebook to py file so clean up the code cells

# In[8]:


#eval
import pytorch_lightning as pl
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor


# In[3]:


#loop to train 
#wil be using pytorch 
#train model 
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch

CHECKPOINT = 'facebook/detr-resnet-50'
class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        #intialize from loaded model
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        
        self.lr = lr
        #backbone has its own learning rate 
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # have bothe learning rate 
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER


# In[17]:


#Initial Model before training 
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)


# In[12]:


#Load Trained model 
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
checkpoint_path = "detr_model.ckpt"
loaded_model = Detr.load_from_checkpoint(checkpoint_path, 
                                         lr=1e-4, 
                                         lr_backbone=1e-5, 
                                         weight_decay=1e-4)
model = loaded_model


# In[10]:


import os
import torchvision
#load data 
dataset="/umbc/rs/pi_bbekele1/users/bbekele1/CVyolo/Salmon-salmon-co"
# settings
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")
VAL_DIRECTORY = os.path.join(dataset, "valid")
TEST_DIRECTORY = os.path.join(dataset, "test")


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


TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY, 
    image_processor=image_processor, 
    train=True)
VAL_DATASET = CocoDetection(
    image_directory_path=VAL_DIRECTORY, 
    image_processor=image_processor, 
    train=False)
TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY, 
    image_processor=image_processor, 
    train=False)

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))

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
from torch.utils.data import DataLoader

def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible 
    # to directly batch together images. Hence they pad the images to the biggest 
    # resolution in a given batch, and create a corresponding binary pixel_mask 
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)


# In[13]:


import random
import cv2
import numpy as np
import supervision as sv

#one predection
# utils
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
box_annotator = sv.BoxAnnotator()

# select random image
image_ids = TEST_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
print('Image #{}'.format(image_id))

# load image and annotatons 
image = TEST_DATASET.coco.loadImgs(image_id)[0]
annotations = TEST_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TEST_DATASET.root, image['file_name'])
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


# In[14]:


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


# In[22]:


import torch
from tqdm import tqdm
from coco_eval import CocoEvaluator
from transformers import DetrFeatureExtractor  # or DetrImageProcessor
import supervision as sv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(DEVICE)

image_processor = DetrFeatureExtractor()  # or DetrImageProcessor.from_pretrained(...)

# COCO categories
categories = TEST_DATASET.coco.cats
id2label = {k: v['name'] for k, v in categories.items()}
print ("label", id2label)
# Initialize COCO evaluator

coco_gt = TEST_DATASET.coco  # This is a pycocotools.COCO object
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

for batch in tqdm(TEST_DATALOADER):
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
    #print (labels)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

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

#result 
"""
[{'scores': tensor([0.9884, 0.9975], device='cuda:0'), 'labels': tensor([1, 1], device='cuda:0'), 'boxes': tensor([[139.1354, 190.4655, 341.4030, 315.1118],
        [112.8933, 233.1269, 302.6567, 381.9147]], device='cuda:0')}, {'scores': tensor([0.9528], device='cuda:0'), 'labels': tensor([2], device='cuda:0'), 'boxes': tensor([[202.9638,  52.6585, 430.3690, 240.3893]], device='cuda:0')}, {'scores': tensor([0.7845, 0.9968, 0.8196, 0.8963], device='cuda:0'), 'labels': tensor([1, 1, 1, 1], device='cuda:0'), 'boxes': tensor([[117.0479, 198.9505, 322.6935, 326.7004],
        [107.1030, 232.7864, 299.6960, 381.0735],
        [125.3642, 196.0194, 319.6397, 326.0384],
        [124.4109, 198.1457, 319.9150, 328.5719]], device='cuda:0')}, {'scores': tensor([0.9802], device='cuda:0'), 'labels': tensor([2], device='cuda:0'), 'boxes': tensor([[ 32.5562, 307.9153, 508.2642, 511.3649]], device='cuda:0')}]
"""

#predection
"""

{0: {'scores': tensor([0.9884, 0.9975], device='cuda:0'), 'labels': tensor([1, 1], device='cuda:0'), 'boxes': tensor([[139.1354, 190.4655, 341.4030, 315.1118],
        [112.8933, 233.1269, 302.6567, 381.9147]], device='cuda:0')}, 1: {'scores': tensor([0.9528], device='cuda:0'), 'labels': tensor([2], device='cuda:0'), 'boxes': tensor([[202.9638,  52.6585, 430.3690, 240.3893]], device='cuda:0')}, 2: {'scores': tensor([0.7845, 0.9968, 0.8196, 0.8963], device='cuda:0'), 'labels': tensor([1, 1, 1, 1], device='cuda:0'), 'boxes': tensor([[117.0479, 198.9505, 322.6935, 326.7004],
        [107.1030, 232.7864, 299.6960, 381.0735],
        [125.3642, 196.0194, 319.6397, 326.0384],
        [124.4109, 198.1457, 319.9150, 328.5719]], device='cuda:0')}, 3: {'scores': tensor([0.9802], device='cuda:0'), 'labels': tensor([2], device='cuda:0'), 'boxes': tensor([[ 32.5562, 307.9153, 508.2642, 511.3649]], device='cuda:0')}}
"""

#labels

"""
[{'size': tensor([800, 800], device='cuda:0'), 'image_id': tensor([0], device='cuda:0'), 'class_labels': tensor([1, 1], device='cuda:0'), 'boxes': tensor([[0.4077, 0.6177, 0.3740, 0.3096],
        [0.4580, 0.5068, 0.3965, 0.2480]], device='cuda:0'), 'area': tensor([74103.3906, 62941.8945], device='cuda:0'), 'iscrowd': tensor([0, 0], device='cuda:0'), 'orig_size': tensor([512, 512], device='cuda:0')}, {'size': tensor([800, 800], device='cuda:0'), 'image_id': tensor([1], device='cuda:0'), 'class_labels': tensor([2], device='cuda:0'), 'boxes': tensor([[0.6270, 0.2852, 0.4297, 0.3789]], device='cuda:0'), 'area': tensor([104199.2188], device='cuda:0'), 'iscrowd': tensor([0], device='cuda:0'), 'orig_size': tensor([512, 512], device='cuda:0')}, {'size': tensor([800, 800], device='cuda:0'), 'image_id': tensor([2], device='cuda:0'), 'class_labels': tensor([1, 1], device='cuda:0'), 'boxes': tensor([[0.4009, 0.6157, 0.3760, 0.3057],
        [0.4341, 0.5127, 0.3955, 0.2480]], device='cuda:0'), 'area': tensor([73550.4141, 62786.8672], device='cuda:0'), 'iscrowd': tensor([0, 0], device='cuda:0'), 'orig_size': tensor([512, 512], device='cuda:0')}, {'size': tensor([800, 800], device='cuda:0'), 'image_id': tensor([3], device='cuda:0'), 'class_labels': tensor([2], device='cuda:0'), 'boxes': tensor([[0.5342, 0.7998, 0.9316, 0.4004]], device='cuda:0'), 'area': tensor([238732.9062], device='cuda:0'), 'iscrowd': tensor([0], device='cuda:0'), 'orig_size': tensor([512, 512], device='cuda:0')}]
"""

#cocopred
"""
[{'image_id': 0, 'category_id': 1, 'bbox': [139.13540649414062, 190.46551513671875, 202.26760864257812, 124.64630126953125], 'score': 0.9884489178657532}, 
 {'image_id': 0, 'category_id': 1, 'bbox': [112.89329528808594, 233.1268768310547, 189.7633819580078, 148.7877960205078], 'score': 0.997468113899231},
 {'image_id': 1, 'category_id': 2, 'bbox': [202.9637908935547, 52.658531188964844, 227.4052276611328, 187.73080444335938], 'score': 0.9527735710144043}, 
 {'image_id': 2, 'category_id': 1, 'bbox': [117.04785919189453, 198.9505157470703, 205.64559936523438, 127.74986267089844], 'score': 0.7844830751419067}, 
 {'image_id': 2, 'category_id': 1, 'bbox': [107.1030044555664, 232.78640747070312, 192.59304809570312, 148.287109375], 'score': 0.9968112111091614}, 
 {'image_id': 2, 'category_id': 1, 'bbox': [125.36421966552734, 196.0194091796875, 194.27545166015625, 130.01898193359375], 'score': 0.819608211517334}, 
 {'image_id': 2, 'category_id': 1, 'bbox': [124.41085052490234, 198.14566040039062, 195.50411987304688, 130.42620849609375], 
   """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            'score': 0.8962547779083252}, {'image_id': 3, 'category_id': 2, 'bbox': [32.556182861328125, 307.915283203125, 475.7080078125, 203.4495849609375], 'score': 0.980189323425293}]


# In[23]:


map50 = evaluator.coco_eval['bbox'].stats[1]  # index 1 is mAP@0.5
print(f"mAP@0.5: {map50:.4f}")
#mAP@0.5: 0.4283


# In[24]:


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


# In[25]:


total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")


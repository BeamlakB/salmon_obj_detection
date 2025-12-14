
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("cv-project-gydhs").project("salmon-salmon")
version = project.version(1)
dataset = version.download("yolov11") #or coco
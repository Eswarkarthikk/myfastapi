import os
import torch
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import json
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

# Disable Tkinter-related warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Use a non-interactive Matplotlib backend
import matplotlib
matplotlib.use('Agg')

app = FastAPI()

# Paths to local model file
MODEL_PATH = "faster_rcnn_state.pth"

# Number of labels (directly hardcoded)
num_labels = 20

# Create model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_labels)

# Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dictionary for class labels (adjust based on your model's classes)
classes = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
           11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
           19: 'train', 20: 'tvmonitor'}

def obj_detector(image_file):
    img = np.array(Image.open(io.BytesIO(image_file)).convert("RGB"))
    img = img.astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    
    model.eval()
    detection_threshold = 0.70
    
    img = list(im.to(device) for im in img)
    output = model(img)

    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()
    labels = output[0]['labels'].data.cpu().numpy()

    labels = labels[scores >= detection_threshold]
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    scores = scores[scores >= detection_threshold]

    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img[0].permute(1, 2, 0).cpu().numpy())
    
    for i, box in enumerate(boxes):
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, classes[labels[i]], color='white', backgroundcolor='red', fontsize=10)
    
    buffered = io.BytesIO()
    fig.savefig(buffered, format="JPEG")
    plt.close()
    
    labeled_image = Image.open(buffered).convert("RGB")
    
    return labeled_image

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_file = await file.read()
        labeled_image = obj_detector(image_file)
        
        original_image_str = base64.b64encode(image_file).decode("utf-8")
        
        buffered = io.BytesIO()
        labeled_image.save(buffered, format="JPEG")
        labeled_image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        buffered.truncate(0)
        buffered.seek(0)
        
        return JSONResponse(content={
            'original_image': original_image_str,
            'labeled_image': labeled_image_str
        })
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"Hello": "World"}

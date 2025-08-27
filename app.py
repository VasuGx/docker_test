from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import cv2
import numpy as np

app = FastAPI()

# Load model
model = YOLO("best.pt")

@app.get("/")
def read_root():
    return {"status": "YOLO model is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    results = model(temp_file)
    predictions = results[0].boxes.xyxy.tolist()  # bounding boxes
    classes = results[0].boxes.cls.tolist()       # class IDs
    scores = results[0].boxes.conf.tolist()       # confidence

    return {"predictions": [
        {"bbox": bbox, "class": int(cls), "score": float(score)}
        for bbox, cls, score in zip(predictions, classes, scores)
    ]}

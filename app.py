from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import shutil
import cv2
import numpy as np
import io
import os
import uvicorn

app = FastAPI()

# Load YOLO model
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

    # Read image for drawing
    img = cv2.imread(temp_file)

    # Draw bounding boxes
    for bbox, cls, score in zip(predictions, classes, scores):
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{int(cls)} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Convert image to JPEG in memory
    _, img_encoded = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

# ✅ Needed for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

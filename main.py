from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI(title="Face Mask AI API")

app.add_middleware(
    CORSMiddleware,
        allow_origins=[
        "http://localhost:3000",
        "https://face-mask-ai-web-seven.vercel.app",
        "https://face-mask-ai-web.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

print("Model loaded successfully.")
print("Classes:", model.names)


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Face Mask AI backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "face_mask_detector_v1_best.pt",
        "classes": model.names
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    contents = await file.read()

    try:
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )

    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.40,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": model.names[class_id],
                "confidence": round(confidence, 4),
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

    annotated = result.plot()

    annotated_image = Image.fromarray(
        annotated[:, :, ::-1]
    )

    buffer = io.BytesIO()

    annotated_image.save(
        buffer,
        format="JPEG",
        quality=90
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "success": True,
        "detections": detections,
        "image_width": image.width,
        "image_height": image.height,
        "annotated_image": (
            f"data:image/jpeg;base64,{encoded_image}"
        )
    }
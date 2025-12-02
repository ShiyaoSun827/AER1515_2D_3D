import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from main import YOLOPv2 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLOPv2("yolopv2_Nx3x480x640.onnx", confThreshold=0.5)

class ImageRequest(BaseModel):
    image_base64: str
    # Geometric parameters from frontend
    cam_height: float = 1.3     # Camera height in meters
    pitch: float = 0.0          # Camera pitch in radians
    fov_v: float = 55.0         # Vertical FOV in degrees
    fcw_threshold: float = 15.0 # Forward Collision Warning threshold in meters

class ImageResponse(BaseModel):
    image_base64: str

@app.post("/infer", response_model=ImageResponse)
def infer(req: ImageRequest):
    try:
        # Decode image
        img_data = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Failed to decode image"}

        # Pack camera config
        camera_cfg = {
            'cam_height': req.cam_height,
            'pitch': req.pitch,
            'fov_v': req.fov_v
        }

        # Run detection with geometry distance and warning threshold
        out = model.detect(img, camera_cfg=camera_cfg, fcw_threshold=req.fcw_threshold)

        # Encode result
        _, buf = cv2.imencode(".jpg", out)
        out_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        
        return ImageResponse(image_base64=out_b64)
        
    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
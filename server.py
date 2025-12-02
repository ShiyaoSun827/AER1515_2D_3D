# server.py
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from main import YOLOPv2   # 这里引用你现有的 main.py 里的类

app = FastAPI()

# 允许 RN 客户端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 开发阶段先放开，后面可以收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 你的 yolopv2_Nx3x480x640.onnx 放在 onnx/ 目录下
model = YOLOPv2("yolopv2_Nx3x480x640.onnx", confThreshold=0.5)

class ImageRequest(BaseModel):
    image_base64: str  # RN 传过来的 base64 图片字符串

class ImageResponse(BaseModel):
    image_base64: str  # 处理后的图片（叠加了车道分割）

@app.post("/infer", response_model=ImageResponse)
def infer(req: ImageRequest):
    # 1. base64 → OpenCV 图像
    import base64
    img_data = base64.b64decode(req.image_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. YOLOPv2 推理（会在图像上画 drivable area + lane line）
    out = model.detect(img)   # detect 函数里已经画好了蓝色车道线、绿色可通行区域

    # 3. 再编码成 JPEG + base64 返回给前端
    _, buf = cv2.imencode(".jpg", out)
    out_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return ImageResponse(image_base64=out_b64)

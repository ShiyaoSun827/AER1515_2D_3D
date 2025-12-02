# server.py
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from main import YOLOPv2   # 引用修改后的 main.py

app = FastAPI()

# 允许 RN 客户端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 开发阶段先放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型
# 请确保 onnx 模型路径正确
model = YOLOPv2("onnx/yolopv2_Nx3x480x640.onnx", confThreshold=0.5)

class ImageRequest(BaseModel):
    image_base64: str          # 图片数据
    cam_height: float = 1.3    # [新增] 相机离地高度，默认 1.3米
    pitch: float = 0.0         # [新增] 相机俯仰角(弧度)，默认 0
    fov_v: float = 55.0        # [新增] 垂直视场角，默认 55度

class ImageResponse(BaseModel):
    image_base64: str          # 处理后的图片

@app.post("/infer", response_model=ImageResponse)
def infer(req: ImageRequest):
    # 1. base64 -> OpenCV 图像
    try:
        img_data = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return {"error": "Invalid image data"}

    # 2. 构造几何配置参数
    camera_cfg = {
        'cam_height': req.cam_height,
        'pitch': req.pitch,
        'fov_v': req.fov_v
    }

    # 3. YOLOPv2 推理 (传入 camera_cfg 以启用测距)
    # detect 函数会返回绘制了检测框、车道线、可行驶区域以及距离数值的图像
    out = model.detect(img, camera_cfg=camera_cfg)

    # 4. 编码回 JPEG + base64 返回给前端
    _, buf = cv2.imencode(".jpg", out)
    out_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    
    return ImageResponse(image_base64=out_b64)

if __name__ == "__main__":
    import uvicorn
    # 启动服务：host="0.0.0.0" 允许局域网访问
    uvicorn.run(app, host="0.0.0.0", port=8000)
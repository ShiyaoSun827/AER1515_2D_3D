import cv2
from main import YOLOPv2

# 1. 配置路径
model_path = "yolopv2_Nx3x480x640.onnx"   # 你的模型
input_video_path = "input.mp4"            # 下载好的原始视频
output_video_path = "output_yolop.mp4"    # 输出视频名

# 2. 初始化模型（用 OpenCV dnn 版 YOLOPv2）
yolop = YOLOPv2(model_path, confThreshold=0.5)

# 3. 打开输入视频
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {input_video_path}")

# 读出原视频的宽、高、FPS，用来创建输出视频
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # 或者 "avc1"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLOPv2 推理 + 画框/车道线/可通行区域
    result = yolop.detect(frame)   # detect 会返回画好东西的 frame

    # 5. 写入输出视频
    out.write(result)

    # （可选）实时看看效果
    cv2.imshow("YOLOPv2 Video", result)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 提前退出
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"处理完成，共 {frame_idx} 帧，输出保存到: {output_video_path}")

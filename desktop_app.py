import cv2
import time
import numpy as np
from main import YOLOPv2  # 直接复用你现有的核心逻辑

def nothing(x):
    pass

def run_desktop_fcw():
    # --- 1. 初始化模型 ---
    print("正在加载模型...")
    # 请确保路径正确，和你之前的 server.py 保持一致
    model_path = "yolopv2_Nx3x480x640.onnx" 
    try:
        model = YOLOPv2(model_path, confThreshold=0.5)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # --- 2. 初始化摄像头 ---
    # 0 通常是 Mac 自带摄像头，1 通常是外接 USB 摄像头。
    # 如果打不开，请尝试修改这个索引为 0 或 1。
    cap = cv2.VideoCapture(0) 
    
    # 设置摄像头分辨率 (可选，取决于你的摄像头支持)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("无法打开摄像头，请检查连接或权限。")
        return

    # --- 3. 创建控制面板 (GUI) ---
    window_name = "MacBook FCW System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 创建滑块 (Trackbars) 用来调节参数
    # OpenCV 滑块只支持整数，所以部分小数参数我们需要在读取时除以 100
    cv2.createTrackbar("Height (cm)", window_name, 130, 250, nothing)    # 默认 1.3m (130cm)
    cv2.createTrackbar("Pitch (deg)", window_name, 50, 100, nothing)     # 默认 0度，范围设为 -50 到 +50 (基准50)
    cv2.createTrackbar("Warn Dist (m)", window_name, 15, 100, nothing)   # 默认 15m

    print("系统启动成功！按 'q' 键退出程序。")

    # 用于计算 FPS
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收画面 (stream end?). Exiting ...")
            break

        # --- 4. 获取当前滑块的参数 ---
        # 高度：滑块值 130 -> 1.3m
        cam_height_cm = cv2.getTrackbarPos("Height (cm)", window_name)
        cam_height = cam_height_cm / 100.0

        # 俯仰角：滑块值 50 为 0度。 <50 为低头，>50 为抬头。
        # 范围：-5.0度 到 +5.0度 (通常摄像头安装误差就在这之间)
        # 逻辑：(Value - 50) / 10.0 -> result in degrees -> convert to radians
        pitch_val_trackbar = cv2.getTrackbarPos("Pitch (deg)", window_name)
        pitch_deg = (pitch_val_trackbar - 50) / 2.0  # 调整精度，每一格 0.5 度
        pitch_rad = np.radians(pitch_deg)

        # 预警阈值
        fcw_threshold = cv2.getTrackbarPos("Warn Dist (m)", window_name)

        # 构造配置字典
        camera_cfg = {
            'cam_height': cam_height,
            'pitch': pitch_rad,
            'fov_v': 55.0 # 很多 Webcam 的垂直视场角约为 50-60度，可根据实际摄像头微调
        }

        # --- 5. 执行推理 ---
        # 直接调用 main.py 中的 detect，它会处理画框、画车道线、测距逻辑
        res_frame = model.detect(frame, camera_cfg=camera_cfg, fcw_threshold=fcw_threshold)

        # --- 6. 绘制 UI 辅助信息 (FPS, 地平线等) ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # 显示 FPS 和 参数
        #info_text = f"FPS: {fps:.1f} | H: {cam_height}m | P: {pitch_deg:.1f} deg"
        info_text = f"Height: {cam_height}m | P: {pitch_deg:.1f} deg"
        cv2.putText(res_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)

        # 画一条地平线辅助线 (绿色虚线)，帮助你物理调节摄像头角度
        # 当 pitch=0 时，地平线应该在画面正中心 (如果光轴居中)
        h, w = res_frame.shape[:2]
        # 简单的地平线估算： y = h/2 + tan(pitch) * f_v (这里简化处理，只画中心线作为参考)
        center_y = int(h / 2)
        cv2.line(res_frame, (0, center_y), (w, center_y), (0, 255, 255), 1)
        cv2.putText(res_frame, "Horizon Ref (Pitch ~0)", (10, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- 7. 显示画面 ---
        cv2.imshow(window_name, res_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_desktop_fcw()
import cv2
import argparse
import numpy as np
from main import YOLOPv2  # 引用你现有的 main.py

def process_video(input_path, output_path, model_path, cam_height, pitch, fov_v, fcw_threshold):
    # 1. 初始化模型
    print(f"正在加载模型: {model_path} ...")
    model = YOLOPv2(model_path, confThreshold=0.5)

    # 2. 打开输入视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {input_path}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 设置视频写入器
    # 使用 mp4v 编码 (如果失败可以尝试 'XVID' 保存为 .avi)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. 配置几何测距参数
    # 注意：在处理视频文件时，Pitch (俯仰角) 是固定的。
    # 你需要根据录制视频时手机的角度来调整这个值，否则距离计算会不准。
    camera_cfg = {
        'cam_height': cam_height,  # 相机离地高度 (米)
        'pitch': pitch,            # 相机俯仰角 (弧度, 抬头为正, 低头为负)
        'fov_v': fov_v             # 垂直视场角 (度)
    }

    print(f"开始处理: {input_path}")
    print(f"参数配置: 高度={cam_height}m, 俯仰角={pitch:.2f}rad, 预警阈值={fcw_threshold}m")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 5. 执行推理 (包含测距和 FCW 绘图)
        # main.py 中的 detect 函数已经包含了根据 camera_cfg 计算距离和根据 fcw_threshold 变红的逻辑
        res_frame = model.detect(frame, camera_cfg=camera_cfg, fcw_threshold=fcw_threshold)
        
        # 写入结果
        out_writer.write(res_frame)
        
        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"进度: {frame_idx}/{total_frames} 帧", end='\r')

    # 释放资源
    cap.release()
    out_writer.release()
    print(f"\n完成! 视频已保存至: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成带有几何测距功能的演示视频")
    
    # 文件路径参数
    parser.add_argument("--input", type=str, default="test_video.mp4", help="输入视频路径")
    parser.add_argument("--output", type=str, default="demo_result.mp4", help="输出视频路径")
    parser.add_argument("--model", type=str, default="yolopv2_Nx3x480x640.onnx", help="ONNX 模型路径")
    
    # 几何参数 (可以根据实际录制的场景调整)
    parser.add_argument("--height", type=float, default=1.3, help="相机离地高度 (米)")
    parser.add_argument("--pitch", type=float, default=0.0, help="相机俯仰角 (弧度)。0表示水平，负数表示略微向下看")
    parser.add_argument("--warn_dist", type=float, default=15.0, help="FCW 报警阈值 (米)")
    
    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        cam_height=args.height,
        pitch=args.pitch,
        fov_v=55.0, # 默认视场角
        fcw_threshold=args.warn_dist
    )
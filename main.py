#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import math

class YOLOPv2():
    def __init__(self, model_path, confThreshold=0.5):
        try:
            self.classes = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
        except FileNotFoundError:
            self.classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck']
            
        self.num_class = len(self.classes)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        self.confThreshold = confThreshold
        self.nmsThreshold = 0.5
        anchors = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
        self.na = len(anchors[0]) // 2
        self.no = len(self.classes) + 5
        self.stride = [8, 16, 32]
        self.nl = len(self.stride)
        self.anchors = np.asarray(anchors, dtype=np.float32).reshape(3, 3, 1, 1, 2)
        self.generate_grid()

    def generate_grid(self):
        self.grid = []
        for i in range(self.nl):
            h, w = int(self.input_height / self.stride[i]), int(self.input_width / self.stride[i])
            self.grid.append(self._make_grid(w, h))

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), 2).reshape(1, 1, ny, nx, 2).astype(np.float32)

    # [新增] 基于几何视觉的单点测距算法
    def calculate_distance(self, v_bottom, img_h, camera_cfg):
        """
        计算车辆与本车的纵向距离
        :param v_bottom: 车辆边界框底部的Y坐标 (像素)
        :param img_h: 原始图像高度 (像素)
        :param camera_cfg: 包含 cam_height(m), pitch(rad), fov_v(deg) 的字典
        """
        if camera_cfg is None:
            # 默认兜底参数
            camera_cfg = {'cam_height': 1.3, 'pitch': 0.0, 'fov_v': 55.0}

        H_cam = camera_cfg.get('cam_height', 1.3)
        alpha = camera_cfg.get('pitch', 0.0)
        fov_v = camera_cfg.get('fov_v', 55.0)

        # 1. 计算垂直焦距 f_v (pixels)
        # 假设光心位于图像中心 c_y
        f_v = (img_h / 2.0) / math.tan(math.radians(fov_v / 2.0))
        c_y = img_h / 2.0

        # 2. 计算目标接地点相对于相机的俯仰角 theta
        # theta = arctan((v - c_y) / f_v)
        theta = math.atan((v_bottom - c_y) / f_v)

        # 3. 计算物理距离 Z = H / tan(alpha + theta)
        angle_sum = alpha + theta

        # 避免分母为0或视线高于地平线（距离无穷大）的情况
        if angle_sum <= 0.01: 
            return 999.0 

        Z = H_cam / math.tan(angle_sum)
        return Z

    def drawPred(self, frame, classId, conf, left, top, right, bottom, distance=-1.0, fcw_threshold=15.0):
        # FCW (前向碰撞预警) 逻辑
        is_danger = False
        if 0 < distance < fcw_threshold:
            is_danger = True
            color = (0, 0, 255) # 红色警告
        else:
            color = (0, 255, 0) # 绿色安全

        # 绘制边界框 (如果危险，边框也加粗到 3，否则为 2)
        box_thickness = 3 if is_danger else 2
        cv2.rectangle(frame, (left, top), (right, bottom), color, box_thickness)

        # 构造标签：类别 + 置信度 + (可选)距离 + (可选)警告
        label = f'{self.classes[classId - 1]}: {conf:.2f}'
        if distance > 0 and distance < 200:
            label += f' | {distance:.1f}m'
        
        if is_danger:
            label += " [WARNING]"

        # --- [关键修改] 调整字体大小和粗细 ---
        fontScale = 1.0  # 字体大小：原为 0.5，改为 1.0 (变大)
        thickness = 2    # 字体粗细：原为 1，改为 2 (变粗)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        # -------------------------------------

        # 1. 计算文字背景框的大小 (使用新的字体参数计算，否则框会包不住字)
        labelSize, baseLine = cv2.getTextSize(label, fontFace, fontScale, thickness)
        
        # 2. 确保标签不会画到屏幕外面去 (如果框在最上面，标签往下画)
        top = max(top, labelSize[1])
        
        # 3. 绘制文字背景 (填充矩形)
        cv2.rectangle(
            frame, 
            (left, top - round(1.2 * labelSize[1])), # 稍微调整背景高度
            (left + round(1.1 * labelSize[0]), top + baseLine), 
            color, 
            cv2.FILLED
        )
        
        # 4. 绘制文字 (白色)
        cv2.putText(frame, label, (left, top), fontFace, fontScale, (255, 255, 255), thickness)
        
        return frame

    # [修改] 增加 camera_cfg 和 fcw_threshold，修复 IndexError
    def detect(self, frame, camera_cfg=None, fcw_threshold=15.0):
        image_width, image_height = frame.shape[1], frame.shape[0]
        ratioh = image_height / self.input_height
        ratiow = image_width / self.input_width

        # Pre process: Resize, BGR->RGB, float32 cast
        input_image = cv2.resize(frame, dsize=(self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        input_image = input_image / 255.0

        # Inference
        results = self.session.run(None, {self.input_name: input_image})

        z = []
        for i in range(3):
            bs, _, ny, nx = results[i+2].shape
            y = results[i+2].reshape(bs, 3, 5+self.num_class, ny, nx).transpose(0, 1, 3, 4, 2)
            y = 1 / (1 + np.exp(-y))
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchors[i]  # wh
            z.append(y.reshape(bs, -1, 5+self.num_class))
        det_out = np.concatenate(z, axis=1).squeeze(axis=0)

        boxes, confidences, classIds = [], [], []
        for i in range(det_out.shape[0]):
            if det_out[i, 4] * np.max(det_out[i, 5:]) < self.confThreshold:
                continue

            class_id = np.argmax(det_out[i, 5:])
            cx, cy, w, h = det_out[i, :4]
            x = int((cx - 0.5*w) * ratiow)
            y = int((cy - 0.5*h) * ratioh)
            width = int(w * ratiow)
            height = int(h* ratioh)

            boxes.append([x, y, width, height])
            classIds.append(class_id)
            confidences.append(det_out[i, 4] * np.max(det_out[i, 5:]))
            
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        
        # [关键修复] 处理 indices 维度问题
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            
            for i in indices:
                idx = int(i) # 确保索引为整数
                
                box = boxes[idx]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                bottom = top + height

                # 计算距离
                dist = -1.0
                if camera_cfg:
                    dist = self.calculate_distance(v_bottom=bottom, img_h=image_height, camera_cfg=camera_cfg)

                # 绘制结果，传入距离和警告阈值
                frame = self.drawPred(frame, classIds[idx], confidences[idx], left, top, left + width, bottom, 
                                      distance=dist, fcw_threshold=fcw_threshold)

        # Drivable Area Segmentation
        drivable_area = np.squeeze(results[0], axis=0)
        mask = np.argmax(drivable_area, axis=0).astype(np.uint8)
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        frame[mask==1] = [0, 255, 0]
        
        # Lane Line Segmentation
        lane_line = np.squeeze(results[1])
        mask = np.where(lane_line > 0.5, 1, 0).astype(np.uint8)
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        frame[mask==1] = [255, 0, 0]
        
        return frame


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='onnx/yolopv2_Nx3x480x640.onnx', help="model path")
    parser.add_argument("--imgpath", type=str, default='images/0ace96c3-48481887.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence')
    args = parser.parse_args()

    net = YOLOPv2(args.modelpath, confThreshold=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)
    
    if srcimg is None:
        print(f"Error: Could not load image from {args.imgpath}")
    else:
        # 本地测试用的相机参数
        test_camera_cfg = {'cam_height': 1.4, 'pitch': 0.0, 'fov_v': 55.0}
        # 设置一个较大的阈值以便测试变红效果
        srcimg = net.detect(srcimg, camera_cfg=test_camera_cfg, fcw_threshold=20.0)

        cv2.imwrite("result_final.jpg", srcimg)
        print("Result saved to result_final.jpg")
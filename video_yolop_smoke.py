import cv2
import numpy as np
import torch

from main import YOLOPv2                     # 你的 YOLOPv2 封装
from smoke.config import cfg                 # SMOKE 的 config
from smoke.modeling.detector import build_detection_model
from smoke.utils.check_point import DetectronCheckpointer

# ===================== 0. 从 KITTI calib 读取内参 K =====================

def load_k_from_kitti_calib(calib_file):
    """
    从 KITTI 标定文件中读取 P2，并取其前 3x3 作为近似的相机内参 K。

    calib 文件格式大概是：
        P0: ...
        P1: ...
        P2: fx 0 cx 0 fy cy 0 0 1 ...
        ...
    我们只取 P2 后面的 12 个数，reshape 成 3x4，再拿前 3x3。
    """
    with open(calib_file, "r") as f:
        lines = f.readlines()

    P2 = None
    for line in lines:
        if line.startswith("P2:"):
            # 去掉 'P2:' 然后按空格拆分
            parts = line.strip().split()[1:]  # 丢掉 'P2:'
            vals = [float(p) for p in parts]
            if len(vals) != 12:
                raise RuntimeError(f"P2 行长度不是 12: got {len(vals)}")
            P2 = np.array(vals, dtype=np.float32).reshape(3, 4)
            break

    if P2 is None:
        raise RuntimeError(f"在标定文件 {calib_file} 中没有找到 P2 行")

    K = P2[:, :3]  # 取前 3x3
    return K  # (3,3)


# ===================== 1. YOLOPv2 初始化 =====================
yolop_model_path = "yolopv2_Nx3x480x640.onnx"
yolop = YOLOPv2(yolop_model_path, confThreshold=0.5)


# ===================== 2. SMOKE 封装 =====================
class SmokeWrapper:
    """
    简单封一层，提供一个 infer_single_frame(frame_bgr) 接口：
    - 输入：OpenCV BGR 图像 (H, W, 3)
    - 输出：detections: List[np.ndarray]，每个 p 向量一条检测
        p[0]  : cls_id (0=Car,1=Cyclist,2=Pedestrian)
        p[1]  : alpha
        p[2:6]: 2D bbox [left, top, right, bottom]
        p[6:9]: 3D dims [h, w, l]
        p[9:12]: 3D location [x, y, z] (camera coord)
        p[12]: rotation_y
        p[13]: score
    """
    def __init__(self, config_file, ckpt_path=None, kitti_calib_file=None):
        # 载入配置
        cfg.merge_from_file(config_file)
        cfg.freeze()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        # ========== 2.1 读取 KITTI 的 K ==========
        if kitti_calib_file is not None:
            K_np = load_k_from_kitti_calib(kitti_calib_file)
        else:
            # 如果你没给 calib，就先用一个单位矩阵占位（不推荐）
            K_np = np.eye(3, dtype=np.float32)
        # 存成 tensor，方便后面用来投影 3D 框
        self.K = torch.from_numpy(K_np).to(self.device)  # (3,3)

        # ========== 2.2 构建 SMOKE 模型 ==========
        self.model = build_detection_model(cfg).to(self.device)
        self.model.eval()

        # 加载权重，逻辑跟 plain_train_net/test 一致
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=cfg.OUTPUT_DIR)
        ckpt = ckpt_path if ckpt_path is not None else cfg.MODEL.WEIGHT
        _ = checkpointer.load(ckpt, use_latest=ckpt_path is None)

    @torch.no_grad()
    def infer_single_frame(self, frame_bgr):
        """
        返回：list(np.ndarray)，每个 shape=(D,) 的向量 p
        注意：这里的预处理是一个“简化版”，严格来说应该复用 SMOKE 的
        数据增强/归一化流程，你可以根据 data/datasets 里的实现进一步对齐。
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0  # 简单归一化到 [0,1]

        # HWC -> CHW
        img_chw = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)

        # 这里 targets 在推理时可以传 None
        # 目前我们没有把 K 喂进去模型，因为你原版的 test_net/inference
        # 也是直接 model(images, targets)，说明 K 要么是 bake 在数据里，
        # 要么模型内部不显式用 K。这里先跟原实现保持一致。
        outputs = self.model(img_tensor, targets=None)

        # 根据 inference.py 的用法，outputs 支持 .to(cpu) 并可被 for p in outputs 遍历
        outputs_cpu = outputs.to(torch.device("cpu"))

        detections = []
        for p in outputs_cpu:
            # p 是 1D tensor，一条检测
            p_np = p.numpy()
            detections.append(p_np)

        return detections


# ===================== 3. 在图像上画 SMOKE 的 2D BBox =====================
ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}

def draw_smoke_boxes_2d(frame, detections, score_thresh=0.3):
    """
    只基于 SMOKE 的 2D bbox 和 score 叠加到图像上，简单可靠。
    同时把 3D 参数也顺便解析出来，方便你后续画 3D 线框/做 BEV。
    """
    for p in detections:
        if p.shape[0] < 14:
            # 防御：维度不够就跳过
            continue
        cls_id = int(p[0])
        alpha  = float(p[1])
        left, top, right, bottom = p[2:6]
        h, w, l = p[6:9]
        x, y, z = p[9:12]
        ry      = float(p[12])
        score   = float(p[13])

        if score < score_thresh:
            continue

        # 画 2D bbox
        pt1 = (int(left), int(top))
        pt2 = (int(right), int(bottom))
        color = (0, 255, 255)   # 黄一点，和 YOLO 的框区分
        cv2.rectangle(frame, pt1, pt2, color, 2)

        # 文字：类别 + score
        cls_name = ID_TYPE_CONVERSION.get(cls_id, str(cls_id))
        text = f"{cls_name} {score:.2f}"
        cv2.putText(frame, text, (pt1[0], max(0, pt1[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3D 参数 (h,w,l,x,y,z,ry) 你后续可以配合 self.K 做 3D box 投影
    return frame


# ===================== 4. 初始化 SMOKE =====================
smoke_config_file = "configs/smoke_gn_vector.yaml"  # 你的 SMOKE 配置
smoke_ckpt = "/Users/shiyaosun/Desktop/uoft_course/AER1515_project/Initial_result/Project/model_final.pth"

# 👉 这里填一份 KITTI 的 calib 文件路径，例如：
#    kitti_root/training/calib/000000.txt
kitti_calib_file = "/Users/shiyaosun/Desktop/uoft_course/AER1515_project/Initial_result/Project/kitti/testing/calib/000000.txt"

smoke = SmokeWrapper(
    smoke_config_file,
    ckpt_path=smoke_ckpt,
    kitti_calib_file=kitti_calib_file
)

# ===================== 5. 视频读写 & 主循环 =====================
input_video_path = "input.mp4"
output_video_path = "output_yolop_smoke.mp4"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {input_video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLOPv2：2D 检测 + 分割 ---
    frame_yolo = yolop.detect(frame.copy())

    # --- SMOKE：3D 检测（在原始 frame 上跑） ---
    detections = smoke.infer_single_frame(frame)

    # --- 把 SMOKE 的检测画到 YOLO 的结果图上 ---
    frame_fused = draw_smoke_boxes_2d(frame_yolo, detections, score_thresh=0.3)

    out.write(frame_fused)

    cv2.imshow("YOLOPv2 + SMOKE", frame_fused)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"处理完成，共 {frame_idx} 帧，输出保存到: {output_video_path}")

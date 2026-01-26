import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NLTK_QUIET'] = 'True'

import warnings

warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import torch
import cv2
from torchvision.ops import nms
from groundeddino_vl import load_model, predict
from transformers import CLIPProcessor, CLIPModel

# 配置参数
confidence = 0.2
text_threshold = 0.35
alpha = 0.2  # CLIP融合权重

List = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle',
        'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone',
        'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork',
        'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop',
        'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant',
        'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon',
        'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet',
        'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

category_map = {cls: i + 1 for i, cls in enumerate(List)}
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型加载
model = load_model(
    config_path=r"D:\Project\ComSen\GroundedDINO-VL-development\groundeddino_vl\models\configs\GroundingDINO_SwinB_cfg.py",
    checkpoint_path=r"D:\model\dino\groundingdino_swinb_cogcoor.pth",
    device=device
)
model_clip = CLIPModel.from_pretrained(r"D:\model\CLIP").to(device)
processor = CLIPProcessor.from_pretrained(r"D:\model\CLIP", use_fast=True)


def draw_predictions_cv2(img_bgr, predictions, class_list):
    """CLIP重排可视化 - 彩色框 + CLIP类别 + 融合分数"""
    img_draw = img_bgr.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, pred in enumerate(predictions):
        x, y, w, h = map(int, pred['bbox'])
        category_id = pred['category_id']
        score = pred['score']
        class_name = class_list[category_id - 1]
        color = colors[i % len(colors)]

        # 绘制检测框
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 2)

        # 标签背景
        label = f"CLIP:{class_name} {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_draw, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)

        # 标签文字
        cv2.putText(img_draw, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_draw


def apply_nms_post_fusion(temp_predictions, iou_threshold=0.5, conf_threshold=0.1):
    if not temp_predictions:
        return []

    boxes = []
    scores = []
    for pred in temp_predictions:
        x1, y1, w, h = pred['bbox']
        boxes.append([x1, y1, x1 + w, y1 + h])
        scores.append(pred['score'])

    boxes = torch.tensor(boxes, dtype=torch.float32).cpu()
    scores = torch.tensor(scores, dtype=torch.float32).cpu()
    keep_indices = nms(boxes, scores, iou_threshold)
    valid_keep = keep_indices[scores[keep_indices] > conf_threshold]
    return [temp_predictions[i.item()] for i in valid_keep]


def xywh2xyxy(cx, cy, w, h):
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def run_groundingdino_map(class_list, img_paths):
    """GLIP位置 + CLIP类别 + 可视化"""
    caption = ". ".join(class_list)
    all_results = []

    os.makedirs("clip_results", exist_ok=True)

    for i, img_path in enumerate(img_paths):
        print(f"\n处理图片 {i + 1}/{len(img_paths)}: {img_path}")

        I = Image.open(img_path).convert('RGB')
        image = np.array(I)
        j = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]

        # GroundingDINO提供位置提示
        result = predict(
            model=model,
            image=image,
            text_prompt=caption,
            box_threshold=confidence,
            text_threshold=text_threshold,
        )

        temp_predictions = []
        img_id = i + 1

        #CLIP完全接管类别判断
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = np.array(result.boxes)
            scores = np.array(result.scores)
            labels = result.labels

            print(f"DINO检测到 {len(boxes)} 个候选框")

            for box_idx, (box_cxcywh, dino_score, _) in enumerate(zip(boxes, scores, labels)):
                # 反归一化到像素坐标
                boxes_cxcywh = box_cxcywh * np.array([img_w, img_h, img_w, img_h])
                cx, cy, w, h = boxes_cxcywh

                x1, y1, x2, y2 = xywh2xyxy(cx, cy, w, h)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                print('labels:',labels)
                # CLIP重排：裁剪检测区域，用CLIP重新分类
                cropped = I.crop((x1, y1, x2, y2))
                #cropped.show()
                inputs = processor(text=labels, images=cropped, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}  # 移到GPU
                with torch.no_grad():
                    outputs = model_clip(**inputs)

                    probs = outputs.logits_per_image.softmax(dim=1)[0]  # [80个类别的分数]
                    best_idx = probs.argmax().item()
                    best_class = List[best_idx]
                    best_clip_score = probs[best_idx].item()

                # 融合分数：DINO定位置信度 × CLIP分类置信度
                fused_score = dino_score * alpha + best_clip_score * (1 - alpha)
                category_id = category_map[best_class]

                pred = {
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": float(fused_score),
                    "clip_class": best_class  # 保存CLIP选择的类别
                }
                temp_predictions.append(pred)
                print(f"  框{box_idx}: CLIP选 {best_class} (融合分:{fused_score:.3f})")

        # NMS后处理
        nms_results = apply_nms_post_fusion(temp_predictions)
        all_results.extend(nms_results)

        # CLIP结果可视化
        if nms_results:
            annotated_img = draw_predictions_cv2(j, nms_results, class_list)
            cv2.imwrite(f'./result/GroundDINO-clip/{i}.jpg', annotated_img)


if __name__ == "__main__":
    PATH = ['./example/1.jpg', './example/2.jpg', './example/3.jpg']
    results = run_groundingdino_map(List, PATH)


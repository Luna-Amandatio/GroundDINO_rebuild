import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'  # 禁用 Hugging Face Hub 检查
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['NLTK_QUIET'] = 'True'

import warnings

warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from groundeddino_vl import load_model, predict
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision.ops import nms

# 配置参数
num = 500
confidence = 0.2
text_threshold = 0.35
alpha = 0.2  # CLIP融合权重

List = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle',
        'bird', 'boat', 'book',
        'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch',
        'cow', 'cup', 'dining table',
        'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse',
        'hot dog', 'keyboard', 'kite', 'knife',
        'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza',
        'potted plant', 'refrigerator', 'remote', 'sandwich',
        'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase',
        'surfboard', 'teddy bear', 'tennis racket',
        'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase',
        'wine glass', 'zebra']

# 图片位置
base_path = r"D:/Project/ComSen/GLIP/DATASET/Coco.v1i.coco-segmentation_2"
json_path = os.path.join(base_path, "valid", "_annotations.coco.json")
images_path = os.path.join(base_path, "valid")

# 配置文件与模型位置
config_path = r"D:\Project\ComSen\GroundedDINO-VL-development\groundeddino_vl\models\configs\GroundingDINO_SwinB_cfg.py"
checkpoint_path = r"D:\model\dino\groundingdino_swinb_cogcoor.pth"
device = "cuda"

# 模型加载
model = load_model(config_path=config_path, checkpoint_path=checkpoint_path, device=device)

# CLIP加载
model_clip = CLIPModel.from_pretrained(r"D:\model\CLIP").to('cuda')
processor = CLIPProcessor.from_pretrained(r"D:\model\CLIP", use_fast=True)


def apply_nms_post_fusion(temp_predictions, iou_threshold=0.5, conf_threshold=0.1):
    """使用torchvision.ops.nms对融合预测结果后处理"""
    if not temp_predictions:
        return []

    # 转换为torch tensor格式
    boxes = []
    scores = []
    for pred in temp_predictions:
        # torchvision.nms 需要 [x1,y1,x2,y2] 格式
        x1, y1, w, h = pred['bbox']
        boxes.append([x1, y1, x1 + w, y1 + h])
        scores.append(pred['score'])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    # 执行NMS，返回保留的索引
    keep_indices = nms(boxes, scores, iou_threshold)

    # 过滤掉分数太低的框
    valid_keep = keep_indices[scores[keep_indices] > conf_threshold]

    # 转换回原格式
    final_preds = [temp_predictions[i.item()] for i in valid_keep]

    return final_preds


def xywh2xyxy(cx, cy, w, h):
    """将cxcywh转换为xyxy格式"""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def run_groundingdino_map(List):
    caption = ". ".join(List)
    coco_gt = COCO(json_path)
    predictions = []

    # 类别映射到ID
    category_map = {}
    for class_name in List:
        cat_ids = coco_gt.getCatIds(catNms=[class_name])
        category_map[class_name] = cat_ids[0] if cat_ids else 0

    # 获取图像ID，用集合去重
    all_img_ids = set()
    for class_name in List:
        cat_ids = coco_gt.getCatIds(catNms=[class_name])
        img_ids = coco_gt.getImgIds(catIds=cat_ids)
        all_img_ids.update(img_ids)

    img_ids = list(all_img_ids)[:num]
    img_infos = coco_gt.loadImgs(img_ids)

    for i, img_info in enumerate(img_infos):
        img_path = os.path.join(images_path, img_info["file_name"])
        I = Image.open(img_path)
        image = np.array(I.convert('RGB'))
        img_h, img_w = image.shape[:2]

        print(f"[{i + 1}/{len(img_infos)}] {img_info['file_name']}")

        result = predict(
            model=model,
            image=image,
            text_prompt=caption,
            box_threshold=confidence,
            text_threshold=text_threshold,
        )

        # 处理boxes
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = np.array(result.boxes)  # 归一化cxcywh [cx,cy,w,h]
            scores = np.array(result.scores)
            labels = result.labels
            print(f"GroundingDINO检测到 {len(boxes)} 个候选框")

            temp_predictions = []
            for box_idx, (box_cxcywh, score, label) in enumerate(zip(boxes, scores, labels)):
                # 反归一化 cxcywh -> 像素cxcywh
                boxes_cxcywh = box_cxcywh * np.array([img_w, img_h, img_w, img_h])
                cx, cy, w, h = boxes_cxcywh

                # 转换为xyxy用于裁剪
                x1, y1, x2, y2 = xywh2xyxy(cx, cy, w, h)

                # 确保边界不超出图像
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(x1, min(x2, img_w))
                y2 = max(y1, min(y2, img_h))

                # COCO格式 [x,y,w,h]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                # CLIP重排：裁剪检测区域，用CLIP重新分类
                cropped = I.crop((x1, y1, x2, y2))
                inputs = processor(text=labels, images=cropped, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}  # 移到GPU
                with torch.no_grad():
                    outputs = model_clip(**inputs)

                    probs = outputs.logits_per_image.softmax(dim=1)[0]  # [80个类别的分数]
                    best_idx = probs.argmax().item()
                    best_class = List[best_idx]
                    best_clip_score = probs[best_idx].item()

                # 融合分数
                fused_score = score * alpha + best_clip_score * (1 - alpha)

                # 类别匹配
                category_id = category_map.get(best_class, 0)

                pred = {
                    "image_id": img_info["id"],
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "score": float(fused_score)
                }
                temp_predictions.append(pred)

            # NMS后处理
            nms_results = apply_nms_post_fusion(temp_predictions, iou_threshold=0.5)
            predictions.extend(nms_results)

        else:
            print("未检测到目标")

    # 保存预测结果和mAP计算（保持不变）
    pred_file = "groundingdino_clip_predictions.json"
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)

    if predictions:
        cocoDt = coco_gt.loadRes(pred_file)
        cocoEval = COCOeval(coco_gt, cocoDt, "bbox")
        eval_cat_ids = list(category_map.values())
        cocoEval.params.catIds = eval_cat_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAP = cocoEval.stats[1] * 100
        print(f"GroundingDINO+CLIP融合 mAP: {mAP:.5f}%")
        with open('groundeddino_clip_result.txt', 'w', encoding='utf-8') as f:
            f.write(f'mAP结果：{mAP:.7f} %\n')


if __name__ == "__main__":
    run_groundingdino_map(List)


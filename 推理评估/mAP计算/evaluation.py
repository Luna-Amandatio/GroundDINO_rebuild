'''
模型在类别提示词下各类别mAP
'''
#-------------------------------
#保证本地环境的BertTokenizer得以加载，避免联网检查
#-------------------------------
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\LQY1\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)

import warnings
import json
import numpy as np
import torch

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from groundeddino_vl import load_model, predict

warnings.filterwarnings("ignore")

# 配置参数
num = 100
confidence = 0.35
text_threshold = 0.35

List = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book',
 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table',
 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich',
 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']


#图片位置
base_path = r"D:/Project/ComSen/GLIP/DATASET/Coco.v1i.coco-segmentation_2"
json_path = os.path.join(base_path, "valid", "_annotations.coco.json")
images_path = os.path.join(base_path, "valid")

#配置文件与模型位置
config_path = r"../../models/GroundingDINO_SwinB_cfg.py"
checkpoint_path = r"../../models/groundingdino_swinb_cogcoor.pth"
device = "cuda"

#模型加载
model = load_model(config_path=config_path, checkpoint_path=checkpoint_path, device=device)


def xywh2xyxy(cx, cy, w, h):
    """将cxcywh转换为xyxy格式"""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def run_groundingdino_map(CLASSES):

    caption = CLASSES
    coco_gt = COCO(json_path)
    predictions = []
    success_count = 0

    CLASSES_ids = coco_gt.getCatIds(catNms=[CLASSES])
    img_ids = coco_gt.getImgIds(catIds=CLASSES_ids)
    img_infos = coco_gt.loadImgs(img_ids)[:num]
    category_id = CLASSES_ids[0]

    for i, img_info in enumerate(img_infos):
        img_path = os.path.join(images_path, img_info["file_name"])
        image = np.array(Image.open(img_path).convert('RGB'))
        img_h, img_w = image.shape[:2]

        print(f"[{i + 1}/{len(img_infos)}] {img_info['file_name']})")

        #启用半精度推理
        with torch.cuda.amp.autocast(dtype=torch.float16):
            result = predict(
                model=model,
                image=image,
                text_prompt=caption,
                box_threshold=confidence,
                text_threshold=text_threshold,
            )

        # 处理boxes（关键修复！）
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = np.array(result.boxes)  # 假设是归一化cxcywh [cx,cy,w,h]
            scores = np.array(result.scores)

            success_count += 1
            print(f"检测到 {len(boxes)} 个目标")

            # 1. 反归一化 cxcywh -> 像素cxcywh
            boxes_cxcywh = boxes * np.array([img_w, img_h, img_w, img_h])

            # 2. 转换为xyxy
            for box_cxcywh, score in zip(boxes_cxcywh, scores):
                cx, cy, w, h = box_cxcywh

                # 转换为xyxy
                x1, y1, x2, y2 = xywh2xyxy(cx, cy, w, h)

                # COCO格式 [x,y,w,h]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                pred = {
                    "image_id": img_info["id"],
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "score": float(score),
                }
                predictions.append(pred)

        else:
            print(f"未检测到目标")

    # 保存预测结果
    pred_file = "groundingdino_predictions.json"
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=2)

    # mAP计算
    if predictions:
        cocoDt = coco_gt.loadRes(pred_file)
        cocoEval = COCOeval(coco_gt, cocoDt, "bbox")
        cocoEval.params.catIds = CLASSES_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        #print(f"GroundedDINO mAP计算结果:{cocoEval.stats[1] * 100:.5f}%")
        with open('groundeddino_result.txt', 'a', encoding='utf-8') as f:
            f.write(f'类别：{CLASSES},图片总数：{len(img_infos)},mAP结果：{cocoEval.stats[1]*100:.5f}\n')

    else:
        print("无预测结果")


if __name__ == "__main__":
    for CLASSES in List:
        run_groundingdino_map(CLASSES)

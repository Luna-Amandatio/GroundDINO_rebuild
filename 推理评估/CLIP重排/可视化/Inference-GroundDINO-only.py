import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\LQY1\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)

import warnings
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
from groundeddino_vl import load_model, predict, annotate

warnings.filterwarnings("ignore")

# 配置参数
num = 3
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
config_path = r"D:\Project\ComSen\GroundedDINO-VL-development\groundeddino_vl\models\configs\GroundingDINO_SwinB_cfg.py"
checkpoint_path = r"D:\model\dino\groundingdino_swinb_cogcoor.pth"
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


def run_groundingdino_map(List,PATH ):

    caption = ". ".join(List)

    for i, img_path in enumerate(PATH):
        I = cv2.imread(img_path)
        image = np.array(Image.open(img_path).convert('RGB'))

        result = predict(
            model=model,
            image=image,
            text_prompt=caption,
            box_threshold=confidence,
            text_threshold=text_threshold,
        )
        annotated_image = annotate(I, result, show_labels=True)
        file_name = str(i) + '.jpg'
        cv2.imwrite(f'./result/GroundDINO-only/{file_name}', annotated_image)

if __name__ == "__main__":
    PATH = ['./example/1.jpg', './example/2.jpg', './example/3.jpg']

    run_groundingdino_map(List,PATH)
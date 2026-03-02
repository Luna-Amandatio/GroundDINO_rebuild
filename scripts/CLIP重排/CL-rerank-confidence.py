import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到上一级目录（scripts目录）
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
print(f"切换到目录: {os.getcwd()}")
# 将当前目录（scripts）添加到系统路径
sys.path.insert(0, os.getcwd())

from scripts.CLIPRerank import CLIPRerank
import numpy as np


# 配置参数
num = 30

text_threshold = 0.35
 # CLIP融合权重

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


if __name__ == "__main__":
    for confidence in np.arange(0.15, 0.5, 0.05):
        for alpha in np.arange(0, 1.2, 0.2):
            rerank_clip = CLIPRerank(caption=List,
                                     alpha=alpha,
                                     iou_threshold=0.1,
                                     box_threshold=confidence,
                                     text_threshold=0.35,
                                     input_num=500,
                                     output_file=f"GroundDINO与CLIP混合模型在多类别输入下mAP随confidence与alpha变化",
                                     enable_timer=False,
                                     nms=True,
                                     )
            rerank_clip.model_inference()
            rerank_clip.mAP_Calculation()


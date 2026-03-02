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

from scripts.GroundDINO_prediction import GroundDINOInference

List = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book',
 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table',
 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich',
 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']


GroundDINO = GroundDINOInference(caption=List,
                                         iou_threshold = -1,
                                         box_threshold = 0.25,
                                         text_threshold = 0.35,
                                         input_num = 500,
                                         output_file = f"GroundDINO单模型在多类别输入下mAP-未启用nms",
                                         enable_timer = False,
                                         nms=False,
                                 )
GroundDINO.model_inference()
GroundDINO.mAP_Calculation()

"""
GroundDINO = GroundDINOInference(caption=List,
                                         iou_threshold=0.8,
                                         box_threshold=0.35,
                                         text_threshold=0.35,
                                         input_num=500,
                                         output_file=f"GroundDINO单模型在多类别输入mAP-启用nms",
                                         enable_timer=False,
                                         nms=True,
                                 )
GroundDINO.model_inference()
GroundDINO.mAP_Calculation()
"""
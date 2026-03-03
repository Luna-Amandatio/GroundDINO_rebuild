'''
模型在类别提示词下各类别mAP
'''
import os
import sys

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
 'bottle', 'bowl', 'broccoli', 'bus' 'cow', 'cup', 'dining table','dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier',
'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife','laptop']


if __name__ == "__main__":
    for CLASSES in List:
        GroundDINO = GroundDINOInference(caption=[CLASSES],
                                         box_threshold=0.35,
                                         text_threshold=0.35,
                                         input_num=100,
                                         output_file=f"GroundDINO class_map",
                                         enable_timer=True)
        GroundDINO.model_inference()
        GroundDINO.mAP_Calculation()


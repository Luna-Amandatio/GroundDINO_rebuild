'''
比较模型在各置信度下mAP变化
'''
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
import numpy as np


if __name__ == "__main__":
    for confidence in np.arange(0, 1.1, 0.05):
        GroundDINO = GroundDINOInference(caption=['bowl'],
                                         box_threshold=confidence,
                                         text_threshold=0.01,
                                         input_num=100,
                                         output_file=f"模型在box_threshold为{confidence}置信度下mAP变化",
                                         enable_timer=True)
        GroundDINO.model_inference()
        GroundDINO.model_inference()
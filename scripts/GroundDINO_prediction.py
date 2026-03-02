import nltk
def no_download(*args, **kwargs):
    """禁用所有下载尝试"""
    return None
nltk.download = no_download
nltk.data.url = lambda x: None  # 禁用URL访问
# 强制本地路径
nltk.data.path = [r'C:\Users\LQY1\AppData\Roaming\nltk_data']

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NLTK_QUIET'] = 'True'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import json
import torch
import cv2
import time

from groundeddino_vl import load_model, predict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import nms

from dataset_information import dataset_info


class Timer:
    """计时器"""
    def __init__(self, name,enable=False):
        self.enable = enable
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0
        self.name = name

    def __enter__(self):
        """开始计时"""
        if self.enable:
            self.start_time = time.perf_counter()
        else:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """停止计时，返回经过的时间（秒）"""
        if self.enable:
            self.end_time = time.perf_counter()
            self.elapsed_time = self.end_time - self.start_time
            data = {
                "name": self.name,
                "elapsed_seconds": self.elapsed_time if self.elapsed_time else 0,
                "elapsed_ms": self.elapsed_time * 1000 if self.elapsed_time else 0,
            }
            self.save_file("计时结果", data)
            self.reset()
        else:
            pass

    def reset(self) -> None:
        """重置计时器"""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = 0

    def save_file(self , filename, results):
        """保存计时器运行结果"""
        # 创建目录（如果不存在）
        os.makedirs("run_results", exist_ok=True)

        # 保存到 run_results 目录下
        filepath = os.path.join("run_results", filename)
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write('\n')

class GroundDINOInference:
    """

    GroundDINO推理封装,支持单类别与多类别输入

    Attributes:
        caption : 输入glip模型的提示词
        iou_threshold :
        box_threshold : GroundDINO的框置信度
        text_threshold : GroundDINO的文本区域置信度
        num : 单次推理的总图片数量
        #batch : 同时推理批次
        model : glip模型

        dataset : 数据集名
        mode : 模式，决定选择数据集中的什么目录(train or test or valid)
        images_dir : 图片所处目录，由dataset与mode决定

        json_path : 数据集配置的json文件位置
        category_map : 类别映射到ID字典

        enable_timer : 是否启用计时器
        nms : 是否启用NMS后处理

    Methods:
        model_load  : 负责glip模型加载与编译
        dataset_load : 负责数据集图片加载与映射
        model_inference : 负责glip模型推理
        mAP_Calculation : 对生成的.json文件评估mAP等指标
        warmup_model : 模型预热
        xywh2xyxy ： 反归一化函数
        apply_nms_post_fusion : NMS去重


    """
    def __init__(self,
                 caption,
                 iou_threshold = 0.5,
                 box_threshold = 0.5,
                 text_threshold = 0.5,
                 dataset = 'Coco.v1i.coco-segmentation_2',
                 input_num = 100,
                 mode = "valid",
                 output_file = "mAP_results.json",
                 enable_timer = False,
                 nms = False,):

        self.nms = nms
        self.enable_timer = enable_timer
        print(f"选择的图片目录为：{dataset}/{mode}")
        print(f"输入的框置信度为{box_threshold}，输入的文本区域置信度为{text_threshold}")

        self.iou_threshold = iou_threshold
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.num = input_num

        self.input = caption
        self.caption = ". ".join(caption)

        self.model = self.model_load()

        self.dataset = dataset
        self.mode = mode
        self.images_dir = "../DATASET/" + dataset +"/" + mode + "/"

        self.json_path = None
        self.output_file = output_file
        self.category_map = {}

        self.img_infos  = self.dataset_load()



    def model_load(self):
        """加载GLIP模型"""
        with Timer("模型加载", enable=self.enable_timer):
            # 配置文件与模型位置
            config_path = r"../models/GroundingDINO_SwinB_cfg.py"
            checkpoint_path = r"../models/groundingdino_swinb_cogcoor.pth"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # 模型加载
            model = load_model(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
            model.eval()

            with Timer("模型编译", enable=self.enable_timer):
                #pytorch2.0编译加速
                torch.compile(model=model, mode="max-autotune",backend="cudagraphs")
                self.warmup_model(model)

        return model


    def dataset_load(self):
        """加载数据集"""
        with Timer("数据集加载", enable=self.enable_timer):
            img_dir = self.dataset + "/" + self.mode
            try:
                self.json_path = dataset_info.json_files[img_dir]
            except Exception as e:
                print("输入路径：\n",img_dir)
                print(f"生成的数据集名与数据集路径字典：{dataset_info.json_files}\n")
                print(f"请仔细校验{e}")
                raise

            coco_gt = COCO(self.json_path)


            # 类别映射到ID
            for class_name in self.input:
                cat_ids = coco_gt.getCatIds(catNms=[class_name])
                self.category_map[class_name] = cat_ids[0]

            # 获取图像ID，用集合去重
            all_img_ids = set()
            for class_name in self.input:
                cat_ids = coco_gt.getCatIds(catNms=[class_name])

                img_ids = coco_gt.getImgIds(catIds=cat_ids)
                all_img_ids.update(img_ids)

            img_ids = list(all_img_ids)[:self.num]
            img_infos = coco_gt.loadImgs(img_ids)

        return img_infos


    def model_inference(self):
        """模型推理"""
        predictions = []
        temp_predictions = []

        length = len(self.img_infos)
        pbar = tqdm(self.img_infos,
                    desc="推理进度",
                    unit="it",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # 创建从label_id到category_id的映射
        # GLIP的label_id从1开始，对应self.caption列表中的索引
        label_to_catid = {}
        for i, class_name in enumerate(self.input):
            # label_id = i + 1 对应 self.caption[i]
            if class_name in self.category_map:
                label_to_catid[i + 1] = self.category_map[class_name]

        print(f"标签映射关系: {label_to_catid}")
        print(f"类别名称到ID映射: {self.category_map}")

        for i, img_info in enumerate(pbar):
            need_timer = self.enable_timer and i < 10
            try:
                img_path = self.images_dir + img_info['file_name']
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            except:
                continue

            img_h, img_w = image.shape[:2]

            print(f"[{i + 1}/{length}] {img_info['file_name']})")

            # 半精度推理
            with torch.no_grad():
                #with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                    with Timer(f"图片_{i + 1}_推理", enable=need_timer):
                        result = predict(
                            model=self.model,
                            image=image,
                            text_prompt=self.caption,
                            box_threshold=self.box_threshold,
                            text_threshold=self.text_threshold,
                        )

                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = np.array(result.boxes)
                    scores = np.array(result.scores)
                    labels = result.labels

                    # 反归一化 cxcywh -> 像素cxcywh
                    boxes_cxcywh = boxes * np.array([img_w, img_h, img_w, img_h])

                    for box_cxcywh, score, label in zip(boxes_cxcywh, scores, labels):
                        # 关键修改：使用部分匹配查找类别ID（仿照GL_only.py）
                        category_id = None
                        for class_name, cat_id in self.category_map.items():
                            if class_name in label:  # 不区分大小写的部分匹配
                                category_id = cat_id
                                break

                        if category_id is None:
                            continue

                        cx, cy, w, h = box_cxcywh

                        # 转换为xyxy
                        x1, y1, x2, y2 = self.xywh2xyxy(cx, cy, w, h)

                        # COCO格式 [x,y,w,h] - 确保宽度和高度为正数
                        width = max(1, x2 - x1)
                        height = max(1, y2 - y1)

                        bbox = [float(x1), float(y1), float(width), float(height)]

                        pred = {
                            "image_id": int(img_info["id"]),
                            "category_id": int(category_id),
                            "bbox": bbox,
                            "score": float(score),
                        }
                        temp_predictions.append(pred)

                else:
                    print(f"  未检测到目标")

        # NMS后处理
        with Timer(f"NMS后处理", enable=self.enable_timer):
            if self.nms:
                nms_results = self.apply_nms_post_fusion(temp_predictions, iou_threshold=self.iou_threshold)
                predictions.extend(nms_results)
            else:
                predictions.extend(temp_predictions)

        # 保存预测结果
        pred_file = "预测结果.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)

    def mAP_Calculation(self):
        """mAP计算"""
        coco_gt = COCO(self.json_path)
        cocoDt = coco_gt.loadRes("预测结果.json")

        cocoEval = COCOeval(coco_gt, cocoDt, "bbox")

        eval_cat_ids = list(self.category_map.values())
        cocoEval.params.catIds = eval_cat_ids
        cocoEval.params.imgIds = [img_info["id"] for img_info in self.img_infos]  # 只评估有标注的图片

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        results = {
            "mAP": f"{float(cocoEval.stats[0]) * 100:.2f}%",
            "AP_50": f"{float(cocoEval.stats[1]) * 100:.2f}%",
            "AP_75": f"{float(cocoEval.stats[2]) * 100:.2f}%",
            "AP_small": f"{float(cocoEval.stats[3]) * 100:.2f}%",
            "AP_medium": f"{float(cocoEval.stats[4]) * 100:.2f}%",
            "AP_large": f"{float(cocoEval.stats[5]) * 100:.2f}%",
            "AR_1": f"{float(cocoEval.stats[6]) * 100:.2f}%",
            "AR_10": f"{float(cocoEval.stats[7]) * 100:.2f}%",
            "AR_100": f"{float(cocoEval.stats[8]) * 100:.2f}%",
            "AR_small": f"{float(cocoEval.stats[9]) * 100:.2f}%",
            "AR_medium": f"{float(cocoEval.stats[10]) * 100:.2f}%",
            "AR_large": f"{float(cocoEval.stats[11]) * 100:.2f}%",
            "categories": {
                name: cat_id for name, cat_id in self.category_map.items()
            },
            "num_images": len(self.img_infos),
            "iou_threshold": self.iou_threshold,
            "box_threshold" : self.box_threshold,
            "text_threshold" : self.text_threshold,
        }
        # 保存JSON
        save_file(self.output_file, results)



    def warmup_model(self, model, num_iterations=5):
        """预热GLIP模型"""
        print("开始预热模型...")

        # 创建预热用的虚拟图像
        dummy_image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)

        # 使用预热caption
        warmup_caption = self.caption if self.caption else "person"

        print(f"预热迭代次数: {num_iterations}")

        # 预热循环
        for i in range(num_iterations):
            print(f"预热进度: {i + 1}/{num_iterations}")

            with torch.no_grad():
                #with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    try:
                        _ = predict(
                            model=model,
                            image=dummy_image,
                            text_prompt=warmup_caption,
                            box_threshold=self.box_threshold,
                            text_threshold=self.text_threshold,
                        )
                    except Exception as e:
                        print(f"预热过程中出现错误: {e}")
                        continue

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("模型预热完成！")

    def xywh2xyxy(self,cx, cy, w, h):
        """将cxcywh转换为xyxy格式"""
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2


    def apply_nms_post_fusion(self,temp_predictions, iou_threshold=0.5):
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

        # 转换回原格式
        final_preds = [temp_predictions[i.item()] for i in keep_indices]

        return final_preds


def clean():
    """清理当前目录下的所有.json文件"""
    for file in Path('.').glob('*.json'):
        file.unlink()
        print(f"已删除: {file}")


def save_file(filename, results):
    """保存运行结果"""

    filename += '.jsonl'
    # 创建目录（如果不存在）
    os.makedirs("run_results", exist_ok=True)

    # 保存到 run_results 目录下
    filepath = os.path.join("run_results", filename)
    with open(filepath, 'a', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f"结果已保存到 {filepath}")

class Config:
    """GroundDINO 配置类"""

#-------------------------
#测试
#-------------------------
if __name__ == "__main__":
    clean()
    GroundDINO = GroundDINOInference(['bowl'],0.1,0.1,enable_timer=True,input_num=10)
    GroundDINO.model_inference()
    #GroundDINO.mAP_Calculation()



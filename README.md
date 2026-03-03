

### 项目结果

![本地图片](.\assert\result_map.png "本地图片示例")



---



### 项目结构

├─assert          ----示例图片资源   
├─DATASET    ----数据集存放目录   
├─logs             ----profile生成的结果   
├─models       ----模型存放位置   
├─scripts         ----辅助脚本   
└─结果保存     ----可视化生成图片   



---



### 依赖安装

```bash
./编译安装.bat
```
Ground DINO 模型下载 ：https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth   
数据集下载 ： https://app.roboflow.com/ds/qHK8Q42lc8?key=27IHm68o5o  
**注**：请将数据集放于DATASET文件夹，按 **数据集名/模式/xxx.json** 格式

---



### scripts 内各文件功能

1. CLIPRerank :内包含CLIP与GroundDINO混合模型推理类
2. dataset_information ： 内包含数据集信息查询类
3. GroundDINO_prediction ：内包含GroundDINO单模型推理类
4. KeywordGenerate ： 基于blip与nltk分词的关键词生成,用于图片关键词自动标注



---

### CLIP重排功能解析

#### 背景

1. GroundDINO在低置信度下会产生大量破碎框

2. GroundDINO会把背景识别为输入类别 

   

#### 原理

在GroundDINO推理后，对推理的框进行裁剪，同输入提示词列表一同输入CLIP，获取CLIP分数，对较低

CLIP分数的框进行剔除，减少杂框。



#### 效果



---



### 其余创新

1. 对GroundDINO模型进行了pytorch2.0编译，使模型提速
2. 将GroundDINO模型推理使用半精度推理优化
3. 编写了NMS去重函数，用于去除重叠框

'''
可视化各提示词推理结果
'''
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\LQY1\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)

from groundeddino_vl import load_model, predict, annotate
import cv2
import warnings
warnings.filterwarnings("ignore")

IMAGE_LIST = ['./example/bear.jpg', './example/bird.jpg']
classes = ['bear', 'bird']
prompts = []

model = load_model(
    config_path=r"../../models/GroundingDINO_SwinB_cfg.py",
    checkpoint_path = r"../../models/groundingdino_swinb_cogcoor.pth",
    device="cuda"  # or "cpu"
)

for cls in classes:
    prompts.append([
        f"a photo of a {cls}",
        f"a {cls}",
        f"{cls} object",
        f"image containing {cls}"

    ])



for idx, prompt in enumerate(prompts):
    # 可视化并保存
    image = cv2.imread(IMAGE_LIST[idx])
    for caption in prompt:
        result = predict(
            model=model,
            image=IMAGE_LIST[idx],
            text_prompt=caption,
            box_threshold=0.5,
            text_threshold=0.5,
        )

        annotated_image = annotate(image, result, show_labels=True)
        file_name = caption + '.jpg'
        cv2.imwrite(file_name, annotated_image)
print("运行完毕")
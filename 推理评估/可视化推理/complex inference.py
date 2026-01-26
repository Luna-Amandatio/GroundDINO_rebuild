import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'# 禁用 Hugging Face Hub 检查
from transformers import BertTokenizer, BertModel
LOCAL_PATH = r"C:\Users\LQY1\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH,local_files_only=True)
model_Ber = BertModel.from_pretrained(LOCAL_PATH,local_files_only=True)

from groundeddino_vl import load_model, predict, annotate
import cv2

IMAGE_path = './example/tennis.jpg'

prompts = [
    'a man.a racket.a tennis',
    'a photo of a man.a photo of a racket.a photo of a tennis',
    'a black man.white and black racket.white tennis',
    'a black man.white and black racket.spherical white tennis',
]

model = load_model(
    config_path="D:/Project/ComSen/GroundedDINO-VL-development/groundeddino_vl/models/configs/GroundingDINO_SwinB_cfg.py",
    checkpoint_path=r"D:\model\dino\groundingdino_swinb_cogcoor.pth",
    device="cuda"  # or "cpu"
)

for caption in prompts:
    # 可视化并保存
    image = cv2.imread(IMAGE_path)
    result = predict(
            model=model,
            image=IMAGE_path,
            text_prompt=caption,
            box_threshold=0.3,
            text_threshold=0.42,
    )
    annotated_image = annotate(image, result, show_labels=True)
    file_name = caption + '.jpg'
    cv2.imwrite(file_name, annotated_image)
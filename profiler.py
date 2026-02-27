'''
添加模型profiler
'''
import os

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_EVALUATE_OFFLINE'] = '1'

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import warnings

warnings.filterwarnings("ignore")


def detailed_profile_and_export():
    from transformers import BertTokenizer, BertModel
    from groundeddino_vl import load_model, predict
    from PIL import Image
    import numpy as np
    import json

    # 加载BERT模型
    LOCAL_PATH = r"C:\Users\LQY1\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
    print("加载BERT模型...")
    tokenizer = BertTokenizer.from_pretrained(LOCAL_PATH, local_files_only=True)
    model_Ber = BertModel.from_pretrained(
        LOCAL_PATH,
        local_files_only=True,
        attn_implementation="eager"  # 添加这个来避免警告
    ).to("cuda")
    model_Ber.eval()

    # 加载GroundedDINO模型
    print("加载GroundedDINO模型...")
    model = load_model(
        config_path=r"D:\Project\ComSen\GroundedDINO-VL-development\models\GroundingDINO_SwinB_cfg.py",
        checkpoint_path=r"D:\Project\ComSen\GroundedDINO-VL-development\models\groundingdino_swinb_cogcoor.pth",
        device="cuda"
    )

    # 准备数据
    image_path = r"D:\Project\ComSen\GLIP\DATASET\Coco.v1i.coco-segmentation_2\train\000000000785_jpg.rf.6f17e31cb0d96d62bd5fa66926535c6a.jpg"
    text_prompt = "person"

    print("开始详细性能分析...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"初始GPU内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

    # 创建日志目录
    os.makedirs('./logs', exist_ok=True)

    # 1. 标准详细分析 - 直接导出JSON
    print("\n1. 运行标准详细分析...")

    # 使用单个profile会话，完成后直接导出
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=1
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
    ) as prof:

        for iteration in range(4):  # 减少迭代次数
            with record_function(f"## Main_Iteration_{iteration} ##"):

                # 文本处理
                with record_function("1. Text_Tokenization"):
                    inputs = tokenizer(
                        text_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    )
                    input_ids = inputs["input_ids"].to("cuda")
                    attention_mask = inputs["attention_mask"].to("cuda")

                # BERT推理
                with record_function("2. BERT_Inference"):
                    with torch.no_grad():
                        outputs = model_Ber(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )

                # 图像加载
                with record_function("3. Image_Loading"):
                    image = Image.open(image_path).convert("RGB")

                # GroundingDINO推理
                with record_function("4. GroundingDINO_Inference"):
                    result = predict(
                        model=model,
                        image=image_path,
                        text_prompt=text_prompt,
                        box_threshold=0.35,
                        text_threshold=0.25,
                    )

                # 后处理
                with record_function("5. Post_Processing"):
                    if iteration == 3:
                        print(f"找到 {len(result)} 个物体")

            prof.step()

    # 导出主trace
    main_trace_path = "./logs/main_trace.json"
    prof.export_chrome_trace(main_trace_path)
    print(f"主trace已导出: {main_trace_path}")
    print(f"文件大小: {os.path.getsize(main_trace_path) / 1024 ** 2:.2f} MB")

    # 2. 内存专门分析 - 新的profile会话
    print("\n2. 运行内存专门分析...")
    with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            profile_memory=True,
            record_shapes=True
    ) as prof_mem:

        for i in range(3):
            with record_function(f"Memory_Test_{i}"):
                result = predict(
                    model=model,
                    image=image_path,
                    text_prompt=text_prompt,
                    box_threshold=0.35,
                    text_threshold=0.25,
                )

                # 记录内存快照
                with record_function("Memory_Snapshot"):
                    mem_allocated = torch.cuda.memory_allocated()
                    mem_cached = torch.cuda.memory_reserved()

            prof_mem.step()
            torch.cuda.empty_cache()

    # 导出内存trace
    mem_trace_path = "./logs/memory_trace.json"
    prof_mem.export_chrome_trace(mem_trace_path)
    print(f"内存trace已导出: {mem_trace_path}")
    print(f"文件大小: {os.path.getsize(mem_trace_path) / 1024 ** 2:.2f} MB")

    # 3. 算子级别分析 - 新的profile会话
    print("\n3. 运行算子级别分析...")
    with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            record_shapes=True,
            with_flops=True
    ) as prof_kernel:

        for j in range(3):
            with record_function(f"Kernel_Analysis_{j}"):
                # BERT分析
                inputs = tokenizer(text_prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    _ = model_Ber(**inputs)

            prof_kernel.step()

    # 导出kernel trace
    kernel_trace_path = "./logs/kernel_trace.json"
    prof_kernel.export_chrome_trace(kernel_trace_path)
    print(f"Kernel trace已导出: {kernel_trace_path}")
    print(f"文件大小: {os.path.getsize(kernel_trace_path) / 1024 ** 2:.2f} MB")

    # 4. 创建汇总的JSON文件
    print("\n4. 创建汇总JSON文件...")

    # 收集所有trace文件的统计信息
    summary = {
        "gpu_info": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "initial_memory_mb": torch.cuda.memory_allocated() / 1024 ** 2,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "trace_files": [
            {"name": "main_trace", "path": main_trace_path, "size_mb": os.path.getsize(main_trace_path) / 1024 ** 2},
            {"name": "memory_trace", "path": mem_trace_path, "size_mb": os.path.getsize(mem_trace_path) / 1024 ** 2},
            {"name": "kernel_trace", "path": kernel_trace_path,
             "size_mb": os.path.getsize(kernel_trace_path) / 1024 ** 2}
        ]
    }

    # 保存汇总信息
    summary_path = "./logs/profile_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"汇总信息已导出: {summary_path}")

    # 5. 打印性能统计
    print("\n" + "=" * 80)
    print("性能统计摘要")
    print("=" * 80)

    # 从主profiler获取统计信息
    events = prof.key_averages()

    print("\n【Top 10 CUDA时间操作】")
    print(events.table(sort_by="cuda_time_total", row_limit=10))

    print("\n【Top 10 GPU内存使用】")
    print(events.table(sort_by="self_cuda_memory_usage", row_limit=10))

    # 保存统计到文本文件
    stats_path = "./logs/profile_statistics.txt"
    with open(stats_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch Profiler 详细统计\n")
        f.write("=" * 80 + "\n\n")

        f.write("按CUDA时间排序:\n")
        f.write(str(events.table(sort_by="cuda_time_total", row_limit=30)))
        f.write("\n\n按GPU内存排序:\n")
        f.write(str(events.table(sort_by="self_cuda_memory_usage", row_limit=30)))

    print(f"\n详细统计已导出: {stats_path}")

    # 最终汇总
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"生成的JSON文件在 ./logs/ 目录:")
    print(f"  1. {main_trace_path} - 主trace文件")
    print(f"  2. {mem_trace_path} - 内存trace文件")
    print(f"  3. {kernel_trace_path} - 算子trace文件")
    print(f"  4. {summary_path} - 汇总信息")
    print(f"  5. {stats_path} - 统计文本")
    print("\n查看方式:")
    print("  - Chrome浏览器: chrome://tracing 加载JSON文件")
    print("  - TensorBoard: tensorboard --logdir=./logs")
    print(f"\n峰值GPU内存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"当前GPU内存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    # GPU预热和清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")

    detailed_profile_and_export()
import json
import os
import random

def create_unified_validation_set(
    positive_file: str,
    negative_file: str,
    output_file: str,
    shuffle: bool = True
):
    """
    合并正样本和负样本验证集，创建一个统一的验证文件。

    Args:
        positive_file (str): ScanRefer原始正样本验证集文件路径。
        negative_file (str): 已验证的负样本文件路径。
        output_file (str): 输出的统一验证集文件路径。
        shuffle (bool): 是否打乱合并后的样本顺序。
    """
    print("开始创建统一验证集...")

    # 1. 加载正样本数据
    try:
        with open(positive_file, 'r') as f:
            positive_samples = json.load(f)
        print(f"成功加载 {len(positive_samples)} 个正样本来自: {positive_file}")
    except FileNotFoundError:
        print(f"错误: 找不到正样本文件 at {positive_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: 解析正样本文件失败 {positive_file}")
        return

    # 2. 加载负样本数据
    try:
        with open(negative_file, 'r') as f:
            negative_samples = json.load(f)
        print(f"成功加载 {len(negative_samples)} 个负样本来自: {negative_file}")
    except FileNotFoundError:
        print(f"错误: 找不到负样本文件 at {negative_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: 解析负样本文件失败 {negative_file}")
        return

    # 3. 处理正样本：添加 is_negative 标志
    for sample in positive_samples:
        sample['is_negative'] = False

    # 4. 处理负样本：添加 is_negative 标志
    for sample in negative_samples:
        sample['is_negative'] = True

    # 5. 合并数据集
    combined_samples = positive_samples + negative_samples
    print(f"数据集合并完成。总样本数: {len(combined_samples)}")

    # 6. (可选) 打乱数据顺序
    if shuffle:
        print("正在打乱样本顺序...")
        random.shuffle(combined_samples)

    # 7. 保存到输出文件
    try:
        with open(output_file, 'w') as f:
            # 使用 indent=4 以获得与您示例中更接近的格式
            json.dump(combined_samples, f, indent=4) 
        print(f"成功！统一验证集已保存到: {output_file}")
    except IOError as e:
        print(f"错误: 无法写入输出文件 {output_file}. 原因: {e}")


if __name__ == "__main__":
    # --- 文件路径配置 ---
    # ScanRefer的原始验证集 (正样本)
    POSITIVE_SAMPLES_FILE = "../ScanRefer/ScanRefer_filtered_val.json"
    
    # 经过我们严格验证后的负样本集
    NEGATIVE_SAMPLES_FILE = "./final_verified_negatives_val.json"
    
    # 最终生成的统一验证集文件名
    OUTPUT_FILE = "./val_mixed.json"

    # 调用主函数
    create_unified_validation_set(
        positive_file=POSITIVE_SAMPLES_FILE,
        negative_file=NEGATIVE_SAMPLES_FILE,
        output_file=OUTPUT_FILE,
        shuffle=True  # 建议打乱，避免模型在验证时看到连续的同类样本
    )
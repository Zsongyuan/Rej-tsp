import json
import random
import os

# --- 1. 配置区域 (请根据您的实际情况修改) ---

# 指向您存放【完整】数据集文件的目录
# 例如: '/path/to/your/full_dataset/'
ORIGINAL_DATA_DIR = '../tns/' # 假设您的原始json文件在此目录下

# 您希望【保存】生成的【小型】样本文件的目录
# 脚本会自动创建这个目录
SAMPLE_DATA_DIR = './'

# 您希望抽取的样本数量
NUM_TRAIN_SAMPLES = 50
NUM_VAL_SAMPLES = 10

# --- 脚本主逻辑 (通常无需修改) ---

def create_sample_files(config):
    """
    根据配置加载完整数据集，采样后生成小型数据集文件。
    """
    split_name = config['split_name']
    num_samples = config['num_samples']
    json_in_path = config['json_in']
    json_out_path = config['json_out']
    txt_out_path = config['txt_out']

    print(f"--- 开始处理 {split_name} 数据 ---")

    # 步骤 1: 加载完整的JSON标注文件
    if not os.path.exists(json_in_path):
        print(f"错误: 找不到原始JSON文件 -> {json_in_path}")
        print("请检查 ORIGINAL_DATA_DIR 路径是否正确。")
        return
    
    print(f"正在读取: {json_in_path}")
    with open(json_in_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    print(f"原始数据集共有 {len(full_data)} 条标注。")

    # 步骤 2: 随机抽取指定数量的标注
    if len(full_data) < num_samples:
        print(f"警告: 数据集样本数 ({len(full_data)}) 小于您的要求 ({num_samples})。将使用全部样本。")
        sampled_data = full_data
    else:
        sampled_data = random.sample(full_data, num_samples)
    print(f"已成功抽取 {len(sampled_data)} 条标注。")

    # 步骤 3: 从抽样数据中提取唯一的 scene_id
    scene_ids = sorted(list(set(item['scene_id'] for item in sampled_data)))
    print(f"共涉及 {len(scene_ids)} 个唯一的场景 (scene_id)。")

    # 步骤 4: 保存抽样后的JSON文件
    print(f"正在保存小型JSON文件到: {json_out_path}")
    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=4)

    # 步骤 5: 保存包含scene_id的TXT文件
    print(f"正在保存scene_id列表到: {txt_out_path}")
    with open(txt_out_path, 'w', encoding='utf-8') as f:
        for scene_id in scene_ids:
            f.write(scene_id + '\n')
            
    print(f"--- {split_name} 数据处理完成 ---\n")


if __name__ == "__main__":
    # 确保输出目录存在
    if not os.path.exists(SAMPLE_DATA_DIR):
        os.makedirs(SAMPLE_DATA_DIR)

    # 定义训练集和验证集的文件配置
    configs = {
        'train': {
            'split_name': '训练集',
            'num_samples': NUM_TRAIN_SAMPLES,
            'json_in': os.path.join(ORIGINAL_DATA_DIR, 'train_mixed_36665.json'),
            'json_out': os.path.join(SAMPLE_DATA_DIR, 'train_mixed_36665.json'),
            'txt_out': os.path.join(SAMPLE_DATA_DIR, 'ScanRefer_filtered_train.txt')
        },
        'val': {
            'split_name': '验证集',
            'num_samples': NUM_VAL_SAMPLES,
            'json_in': os.path.join(ORIGINAL_DATA_DIR, 'val_mixed.json'),
            'json_out': os.path.join(SAMPLE_DATA_DIR, 'val_mixed.json'),
            'txt_out': os.path.join(SAMPLE_DATA_DIR, 'ScanRefer_filtered_val.txt')
        }
    }

    # 执行处理
    create_sample_files(configs['train'])
    create_sample_files(configs['val'])
    
    print("==============================================")
    print("所有样本文件已成功生成！")
    print(f"请在 '{SAMPLE_DATA_DIR}' 目录下查看生成的文件。")
    print("==============================================")
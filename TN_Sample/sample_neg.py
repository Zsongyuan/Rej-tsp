import json
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_sample_negatives(negative_file, num_samples_to_keep, output_file):
    """
    对负样本进行分层采样，以保持各种生成策略的比例。

    Args:
        negative_file (str): 负样本JSON文件路径。
        num_samples_to_keep (int): 希望保留的样本数量。
        output_file (str): 采样后输出的JSON文件路径。
    """
    # 1. 加载数据为Pandas DataFrame
    with open(negative_file, 'r', encoding='utf-8') as f:
        negative_data = json.load(f)
    df = pd.DataFrame(negative_data)
    
    print("负样本中不同生成策略的分布情况:")
    print(df['generation_strategy'].value_counts(normalize=True))
    
    # 2. 计算采样比例
    total_samples = len(df)
    sample_ratio = num_samples_to_keep / total_samples
    
    if sample_ratio >= 1.0:
        print("希望保留的样本数大于等于总样本数，无需采样。")
        df.to_json(output_file, orient='records', indent=4)
        return

    # 3. 使用 train_test_split 进行分层采样
    # 我们不关心“测试集”，所以用 _ 忽略它
    sampled_df, _ = train_test_split(
        df,
        train_size=sample_ratio,
        stratify=df['generation_strategy'], # <-- 关键：按此列进行分层
        random_state=42  # 保证结果可复现
    )
    
    print(f"\n采样后保留了 {len(sampled_df)} 条样本。")
    print("采样后样本中不同生成策略的分布情况:")
    print(sampled_df['generation_strategy'].value_counts(normalize=True))

    # 4. 保存为JSON文件
    sampled_df.to_json(output_file, orient='records', indent=4)
    print(f"\n分层采样后的负样本已保存至: {output_file}")


if __name__ == "__main__":
    negatives_json = 'final_verified_negatives.json'
    num_positives = 36665
    sampled_negatives_json = f'sampled_negatives_{num_positives}.json'
    
    stratified_sample_negatives(negatives_json, num_positives, sampled_negatives_json)
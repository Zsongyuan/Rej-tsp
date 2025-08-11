import json
import spacy
from collections import Counter
from tqdm import tqdm
import re

def analyze_scanrefer_vocabulary(scanrefer_file_path):
    """
    分析ScanRefer数据集，提取属性词和关系短语。
    (已修正计数逻辑)
    """
    print("Loading ScanRefer data...")
    with open(scanrefer_file_path, 'r') as f:
        data = json.load(f)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # ==================== SOLUTION: Use LIST instead of SET ====================
    # 使用列表来存储所有出现的词，以便正确计数
    adjectives = []
    prepositions = []
    # =======================================================================

    print("Analyzing descriptions...")
    for entry in tqdm(data):
        # 进行一些基础的文本清洗，去除可能的噪声
        description = entry['description'].lower().replace(';', ' ')
        doc = nlp(description)

        # 1. 提取形容词 (属性词)
        for token in doc:
            # 增加一些过滤条件，排除不像形容词的词
            if token.pos_ == 'ADJ' and token.is_alpha and len(token.text) > 2:
                adjectives.append(token.lemma_) # 使用lemma_来获取词元

        # 2. 提取介词 (关系描述的核心)
        for token in doc:
            if token.pos_ == 'ADP': # ADP = Adposition (介词)
                prepositions.append(token.lemma_)

    print("\n--- Analysis Complete ---")
    
    # Counter现在可以正确统计列表中每个词的出现次数
    adj_counter = Counter(adjectives)
    prep_counter = Counter(prepositions)

    print("\nTop 40 Most Common Adjectives (Attributes):")
    for word, count in adj_counter.most_common(40):
        print(f"- {word:<15} (Count: {count})")

    print("\nTop 40 Most Common Prepositions (Relations):")
    for word, count in prep_counter.most_common(40):
        print(f"- {word:<15} (Count: {count})")
        
    return adj_counter, prep_counter


if __name__ == "__main__":
    SCANREFER_TRAIN_FILE = "../ScanRefer/ScanRefer_filtered_train.json"
    adj_counts, prep_counts = analyze_scanrefer_vocabulary(SCANREFER_TRAIN_FILE)
    
    # 现在您可以基于这个准确的统计结果来优化您的 TN_Generator.py
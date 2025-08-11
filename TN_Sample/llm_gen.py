import json
import os
import random
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import Counter

from openai import OpenAI
import spacy

# --- PROMPT 保持不变 ---
PROMPT = """
### ROLE ###
You are an expert Data Augmentation Specialist for a 3D scene understanding project. Your primary function is to generate a list of high-quality "True Negative" descriptions based on ground-truth data.

### GOAL ###
To analyze a single positive description and generate a list of up to 3 distinct, fluent, and logically FALSE descriptions. Each new description must be verifiable as false using ONLY the provided Ground-Truth Data.

### GROUND-TRUTH DATA ###
1.  **Original (Positive) Description:**
    "{original_description}"
2.  **Target Object Details (The main object of the description):**
    - Category: '{target_object_name}'
    - Is this the only object of its category in the scene?: {is_target_unique}
    - True Color: '{target_color}'
    - True Size Descriptor: '{target_size}'
    - True Spatial Relations with other objects: {target_relations_str}
3.  **Mentioned Anchor Objects (Other relevant objects in the description):**
{anchor_objects_str}
4.  **Scene Object Counts (Total count of each object category in the scene):**
{scene_object_counts_str}

### TASK & RULES ###
1.  **CORE PRINCIPLE: MINIMAL CHANGE**: Your goal is to change one single piece of information while keeping the rest of the sentence identical.
2.  **PRESERVE ORIGINAL CONTENT**: You **MUST** keep all parts of the original sentence that are not the direct target of your modification. Do not delete other descriptive words or phrases (e.g., if you change the color 'brown' in "a wooden, brown door", the words 'wooden' and 'door' must be kept).
3.  **NO ADDING NEW ATTRIBUTES**: You **MUST NOT** add new descriptive attributes. If the original sentence does not mention a color, you cannot add a color.
4.  **DYNAMIC GENERATION**: Generate as many *distinct and high-quality* negative descriptions as possible, up to a **maximum of 3**, based on the number of valid modification opportunities that respect all rules.
5.  **TARGET PRIORITY:** Your primary goal is to modify an attribute related to the **Target Object**.
6.  **VALID MODIFICATION STRATEGIES (Choose one per generated sentence):**
    - **Strategy: Add Negation**: **Only if the original sentence contains a verifiable attribute (like color or relation)**, negate that specific attribute. Do not introduce a new attribute to negate.
        - **Correct Example:** Original "a red chair" -> "a chair that is not red".
        - For spatial relations, ONLY use this strategy if the target object is unique in the scene to avoid ambiguity.
    - **Strategy: Change Color**: **Only apply this strategy if the Original Description explicitly mentions a color.** Then, change that color to a contrasting one.
    - **Strategy: Change Spatial Relation**: **Only apply this strategy if the Original Description explicitly mentions a spatial relation.** Then, change it to a clear and unambiguous opposite (e.g., 'beside' -> 'far away from').
    - **Strategy: Change Cardinality (count)**: **Only apply this strategy if the Original Description explicitly mentions a number of items.** Then, change the number to one that is verifiably false.
    - **Strategy: Change Size (Very Conservative)**: **Only apply this strategy if the Original Description explicitly mentions an extreme size ('huge' or 'tiny').** Then, change it to the opposite extreme.
7.  **HIGH-CONTRAST CHANGES:** Modifications must be significant and unambiguous. For colors, change "brown" to "blue" or "green".
8.  **STRICTLY FORBIDDEN MODIFICATIONS**: You **MUST NOT** modify the following types of attributes, as they risk pointing to another real object:
    - **Shape** (e.g., round, square)
    - **Ordinal Numbers** (e.g., first, second)
    - **Superlatives** (e.g., largest, smallest)
    - **Comparatives** (e.g., larger, smaller, taller)
9.  **FLUENCY:** The new sentence must be grammatically correct and sound natural.
10. **OUTPUT FORMAT**: You **MUST** respond with a single JSON list `[...]`. Each element must be an object with "strategy" and "new_description" keys. If no valid modifications are possible, return an empty list `[]`.

### YOUR TURN ###
Based on all the rules and data, generate the JSON list.
"""

class LLM_TrueNegativeGenerator:
    def __init__(self, scannet_file: str, scene_graph_dir: str, output_file: str, model_name: str, num_workers: int = 16):
        self.scannet_data = self._load_json(scannet_file)
        self.scene_graph_dir = scene_graph_dir
        self.output_file = output_file
        self.scene_graphs = {}
        self.model_name = model_name
        self.num_workers = num_workers

        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")

        self.clients = []
        try:
            # 从环境变量中读取以逗号分隔的API密钥列表
            api_keys_str = os.getenv("OPENAI_API_KEYS")
            self.base_url = os.getenv("OPENAI_API_BASE")
            
            if not api_keys_str or not self.base_url:
                raise ValueError("OPENAI_API_KEYS and OPENAI_API_BASE environment variables must be set.")
            
            api_keys = [key.strip() for key in api_keys_str.split(',')]
            if len(api_keys) == 0:
                raise ValueError("OPENAI_API_KEYS environment variable is empty.")

            print(f"Found {len(api_keys)} API keys. Initializing clients...")

            for api_key in api_keys:
                client = OpenAI(api_key=api_key, base_url=self.base_url)
                # 简单检查客户端是否可以连接
                client.models.list()
                self.clients.append(client)
            
            self.client_cycler = itertools.cycle(self.clients)
            print(f"{len(self.clients)} OpenAI-compatible clients initialized successfully.")

        except Exception as e:
            print(f"Error initializing OpenAI clients: {e}")
            self.clients = []

    def _load_json(self, file_path: str) -> List[Dict]:
        if not os.path.exists(file_path):
            print(f"Error: Input file not found at {file_path}")
            return []
        with open(file_path, 'r') as f:
            return json.load(f)

    def _load_scene_graph(self, scene_id: str) -> Optional[Dict]:
        if scene_id in self.scene_graphs: return self.scene_graphs[scene_id]
        file_path = os.path.join(self.scene_graph_dir, f"{scene_id}_scene_graph.json")
        if not os.path.exists(file_path):
            # print(f"Warning: Scene graph not found for {scene_id}") # 在并发中打印会很乱
            return None
        scene_graph = self._load_json(file_path)
        self.scene_graphs[scene_id] = scene_graph
        return scene_graph

    def call_llm_api(self, client: OpenAI, messages: List[Dict], retries=3, delay=5) -> Optional[str]:
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.5,
            "timeout": 180
        }
        for i in range(retries):
            try:
                response = client.chat.completions.create(**request_params)
                response_text = response.choices[0].message.content.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                return response_text.strip()
            except Exception as e:
                # print(f"API call failed on attempt {i+1}/{retries}: {e}") # 在并发中打印会很乱
                if i < retries - 1:
                    time.sleep(delay)
        return None

    def _format_output(self, sample: Dict, llm_response_item: Dict, index: int) -> Dict:
        new_description = llm_response_item.get("new_description", "")
        strategy = llm_response_item.get("strategy", "unknown")

        return {
            "scene_id": sample['scene_id'],
            "object_id": sample['object_id'],
            "object_name": sample['object_name'],
            "ann_id": f"{sample['ann_id']}_neg_dyn_{index}",
            "description": new_description,
            "origDescrip": sample['description'],
            "generation_strategy": strategy,
            "original_ann_id": sample['ann_id'],
            "token": [token.text for token in self.nlp(new_description)]
        }

    def _gather_context_for_prompt(self, sample: Dict, scene_graph: Dict) -> Dict:
        target_obj_id_str = sample.get('object_id')
        if not target_obj_id_str: return {}
        target_obj_data = scene_graph.get('objects', {}).get(target_obj_id_str)
        if not target_obj_data: return {}

        all_objects = scene_graph.get('objects', {})
        category_counts = Counter(obj.get('category') for obj in all_objects.values())
        target_category = target_obj_data.get('category', 'unknown')
        
        context = {
            "original_description": sample['description'],
            "target_object_name": target_category,
            "is_target_unique": category_counts.get(target_category, 0) == 1,
            "target_color": target_obj_data.get('color_info', {}).get('dominant_color', 'unknown'),
            "target_size": target_obj_data.get('size_rank', {}).get('size_descriptor', 'unknown'),
            "target_relations_str": json.dumps(target_obj_data.get('spatial_relations', [])),
            "anchor_objects_str": "N/A",
            "scene_object_counts_str": json.dumps(category_counts)
        }

        doc = self.nlp(sample['description'])
        nouns = {token.lemma_.lower() for token in doc if token.pos_ == 'NOUN'}
        
        anchor_info_list = []
        for obj_id, obj_data in all_objects.items():
            if obj_id != target_obj_id_str and obj_data.get('category', '').lower() in nouns:
                info = (f"- Anchor Category: {obj_data.get('category', 'unknown')} (ID: {obj_id})\n"
                        f"  - True Color: {obj_data.get('color_info', {}).get('dominant_color', 'unknown')}\n"
                        f"  - True Size Descriptor: {obj_data.get('size_rank', {}).get('size_descriptor', 'unknown')}")
                anchor_info_list.append(info)

        if anchor_info_list:
            context["anchor_objects_str"] = "\n".join(anchor_info_list)

        return context

    def process_sample(self, sample: Dict) -> List[Dict]:
        """处理单个样本的完整流程"""
        scene_graph = self._load_scene_graph(sample['scene_id'])
        if not scene_graph:
            return []

        context = self._gather_context_for_prompt(sample, scene_graph)
        if not context:
            return []

        prompt = PROMPT.format(**context)
        messages = [{"role": "user", "content": prompt}]
        
        # 轮流获取一个客户端
        client = next(self.client_cycler)
        response_text = self.call_llm_api(client, messages)
        
        if not response_text:
            return []

        results = []
        try:
            llm_generated_list = json.loads(response_text)
            if not isinstance(llm_generated_list, list):
                # print(f"Warning: LLM response was not a list for ann_id {sample['ann_id']}")
                return []

            for index, item in enumerate(llm_generated_list):
                if "new_description" in item and item["new_description"] != sample["description"]:
                    formatted_result = self._format_output(sample, item, index + 1)
                    results.append(formatted_result)
        except (json.JSONDecodeError, TypeError):
            # print(f"Warning: Failed to decode JSON for ann_id {sample['ann_id']}\nResponse: {response_text}")
            pass
        
        return results

    def generate(self, save_interval=100):
        if not self.clients:
            print("Clients not initialized. Exiting.")
            return

        all_negatives = []
        if os.path.exists(self.output_file):
            print(f"Output file found at {self.output_file}. Resuming generation.")
            all_negatives = self._load_json(self.output_file)
            processed_original_ids = {item['original_ann_id'] for item in all_negatives if 'original_ann_id' in item}
        else:
            processed_original_ids = set()
        
        unprocessed_data = [s for s in self.scannet_data if s['ann_id'] not in processed_original_ids]
        random.shuffle(unprocessed_data)
        
        print(f"Found {len(all_negatives)} existing negative samples. {len(processed_original_ids)} positive samples processed.")
        print(f"Starting generation for {len(unprocessed_data)} remaining samples with {self.num_workers} workers.")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 使用 as_completed 来获取已完成的任务，以便我们可以实时更新进度条和保存
            futures = [executor.submit(self.process_sample, sample) for sample in unprocessed_data]
            
            with tqdm(total=len(unprocessed_data), desc="Overall Progress") as pbar:
                for i, future in enumerate(as_completed(futures)):
                    try:
                        batch_results = future.result()
                        if batch_results:
                            all_negatives.extend(batch_results)
                        
                        # 定期保存结果
                        if (i + 1) % save_interval == 0:
                            print(f"\nProcessed {i+1} samples. Saving {len(all_negatives)} total results to {self.output_file}...")
                            with open(self.output_file, 'w') as f:
                                json.dump(all_negatives, f, indent=4)

                    except Exception as e:
                        print(f"An error occurred while processing a future: {e}")
                    
                    pbar.update(1)

        # 最后再完整保存一次
        print("\nGeneration complete. Final save...")
        with open(self.output_file, 'w') as f:
            json.dump(all_negatives, f, indent=4)
        
        print("All done.")

if __name__ == "__main__":
    # 确保 OPENAI_API_KEYS 和 OPENAI_API_BASE 在您的环境中已设置
    generator = LLM_TrueNegativeGenerator(
        scannet_file="../ScanRefer/ScanRefer_filtered_val.json",
        scene_graph_dir="../build_scene/scene_graphs",
        output_file="./llm_generated_negatives_v3_val.json",
        model_name="gpt-4.1-nano", # 或您的特定模型名称
        num_workers=16 # 指定工作线程数
    )
    generator.generate()
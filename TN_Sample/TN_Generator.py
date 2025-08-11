import json
import os
import random
import re
import spacy
from tqdm import tqdm
from typing import Dict, List, Optional, Any
import matplotlib.colors as mcolors
import colorsys

# ColorHelper 类无任何变化
class ColorHelper:
    def __init__(self):
        self.COMMON_COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white', 'gray']
        self.knowledge_base = self._build_color_knowledge_base()
    def _build_color_knowledge_base(self):
        kb = {}
        for name, hex_val in mcolors.CSS4_COLORS.items():
            try:
                rgb = mcolors.to_rgb(hex_val)
                h, s, v = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
                if s > 0.15 and 0.1 < v < 0.9: kb[name] = {'h': h * 360, 's': s, 'v': v, 'rgb': rgb}
            except ValueError: continue
        return kb
    def _get_opposite_hue(self, original_hue: float): return (original_hue + 180) % 360
    def _find_closest_color_name(self, target_hsv: Dict, candidate_names: List[str]):
        min_dist, closest_name = float('inf'), None
        target_h, target_s, target_v = target_hsv['h'], target_hsv['s'], target_hsv['v']
        for name in candidate_names:
            if name not in self.knowledge_base: continue
            values = self.knowledge_base[name]
            hue = values['h']
            hue_dist = min(abs(target_h - hue), 360 - abs(target_h - hue)) / 180.0
            sat_dist, val_dist = abs(target_s - values['s']), abs(target_v - values['v'])
            dist = hue_dist * 0.5 + sat_dist * 0.25 + val_dist * 0.25
            if dist < min_dist: min_dist, closest_name = dist, name
        return closest_name
    def get_opposite_common_color_name(self, original_color_name: str):
        original_color_name = original_color_name.lower()
        if original_color_name not in self.knowledge_base: return None
        original_hsv = self.knowledge_base[original_color_name]
        opposite_hue = self._get_opposite_hue(original_hsv['h'])
        ideal_opposite_hsv = {'h': opposite_hue, 's': original_hsv['s'], 'v': original_hsv['v']}
        closest_common_name = self._find_closest_color_name(ideal_opposite_hsv, self.COMMON_COLORS)
        if closest_common_name == original_color_name: return None
        return closest_common_name

class TrueNegativeGenerator:
    def __init__(self, scannet_file: str, scene_graph_dir: str):
        self.scannet_data = self._load_json(scannet_file)
        self.scene_graph_dir = scene_graph_dir
        self.scene_graphs = {}
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.vowels = "aeiou"
        print("Building color knowledge base...")
        self.color_helper = ColorHelper()
        print(f"Color KB built. Common colors: {len(self.color_helper.COMMON_COLORS)}.")
        self.relation_swaps = {
        # 只保留最明确的垂直对立
        'on': ['under'],
        'under': ['on'],
        'above': ['below'],
        'below': ['above'],
        
        # 保留明确的方向对立
        'left of': ['right of'],
        'right of': ['left of'],
        'front of': ['behind'],
        'behind': ['front of'],
        
        # 保留明确的内外对立
        'inside': ['outside'],
        'outside': ['inside'],
        
        # 保留明确的远近对立
        'near': ['far from'],
        'beside': ['far from'], # beside 隐含了 near 的意思
        
        # 保留明确的顶部/底部对立
        'atop': ['underneath'],
        'underneath': ['atop'],
        }
        self.relation_phrases = sorted(self.relation_swaps.keys(), key=len, reverse=True)
        self.size_swaps = {
            'large': 'small', 'big': 'small', 'huge': 'tiny', 'small': 'large', 'tiny': 'huge', 'tall': 'short',
            'short': 'tall', 'long': 'short', 'wide': 'narrow', 'narrow': 'wide',
        }
        self.size_adjectives = list(self.size_swaps.keys())
        self.shape_swaps = {
            'rectangular': 'round', 'square': 'circular', 'round': 'square', 'circular': 'square',
            'curved': 'straight', 'straight': 'curved',
        }
        self.shape_adjectives = list(self.shape_swaps.keys())

    # ... (辅助函数 _load_json, _load_scene_graph, _validate_attribute_negative 保持不变) ...
    def _replace_with_article_correction(self, text: str, old_word: str, new_word: str) -> str:
        pattern = re.compile(r"\b(a|an)\s+" + re.escape(old_word) + r"\b", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            pattern = re.compile(r"\b" + re.escape(old_word) + r"\b", re.IGNORECASE)
            if new_word == "": return pattern.sub("", text, 1).replace("  ", " ")
            return pattern.sub(new_word, text, 1)
        if not new_word: return pattern.sub(match.group(1), text, 1)
        new_article = "an" if new_word.lower().startswith(tuple(self.vowels)) else "a"
        replacement = f"{new_article} {new_word}"
        return pattern.sub(replacement, text, 1)
    def _load_json(self, file_path: str):
        with open(file_path, 'r') as f: return json.load(f)
    def _load_scene_graph(self, scene_id: str):
        if scene_id in self.scene_graphs: return self.scene_graphs[scene_id]
        file_path = os.path.join(self.scene_graph_dir, f"{scene_id}_scene_graph.json")
        if not os.path.exists(file_path): return None
        scene_graph = self._load_json(file_path)
        self.scene_graphs[scene_id] = scene_graph
        return scene_graph
    def _validate_attribute_negative(self, scene_graph: Dict, object_name: str, attr_type: str, new_attr_val: str) -> bool:
        for _, obj_data in scene_graph['objects'].items():
            if obj_data['category'] == object_name:
                if attr_type == 'color' and obj_data['color_info'].get('dominant_color') == new_attr_val: return False
                if attr_type == 'size' and obj_data['size_rank'].get('size_descriptor') == new_attr_val: return False
        return True
    
    # ==================== MODIFICATION 3: New Formatting Function ====================
    def _format_as_scanrefer(self, sample: Dict, new_description: str, ann_id_suffix: str):
        """新的格式化函数，自动添加原始描述。"""
        return {
            "scene_id": sample['scene_id'],
            "object_id": "None",
            "object_name": sample['object_name'],
            "ann_id": f"{sample['ann_id']}_{ann_id_suffix}",
            "description": new_description,
            "origin_Desc": sample['description'], # 新增字段
            "token": [token.text for token in self.nlp(new_description)]
        }
    # ==============================================================================

    # ==================== MODIFICATION 1: Relaxed Generation Logic ====================
    def generate_attribute_negatives(self, sample: Dict, scene_graph: Dict):
        description = sample['description'].lower()
        # 在整个描述中寻找任何颜色词
        original_color_name = next((word for word in re.split(r'[\s.,!?]+', description) if word in self.color_helper.knowledge_base), None)
        if not original_color_name: return []
        
        swap_color_name = self.color_helper.get_opposite_common_color_name(original_color_name)
        if not swap_color_name: return []

        # 验证逻辑依然针对核心目标物体
        if self._validate_attribute_negative(scene_graph, sample['object_name'], 'color', swap_color_name):
            new_description = self._replace_with_article_correction(description, original_color_name, swap_color_name)
            ann_id_suffix = f"neg_attr_{swap_color_name}"
            return [self._format_as_scanrefer(sample, new_description, ann_id_suffix)]
        return []

    def generate_size_negatives(self, sample: Dict, scene_graph: Dict):
        description = sample['description'].lower()
        original_size = next((word for word in self.size_adjectives if re.search(r'\b' + word + r'\b', description)), None)
        if not original_size: return []

        swap_size = self.size_swaps.get(original_size)
        if not swap_size: return []

        if self._validate_attribute_negative(scene_graph, sample['object_name'], 'size', swap_size):
            new_description = self._replace_with_article_correction(description, original_size, swap_size)
            ann_id_suffix = f"neg_size_{swap_size}"
            return [self._format_as_scanrefer(sample, new_description, ann_id_suffix)]
        return []

    def generate_shape_negatives(self, sample: Dict, scene_graph: Dict):
        description = sample['description'].lower()
        original_shape = next((word for word in self.shape_adjectives if re.search(r'\b' + word + r'\b', description)), None)
        if not original_shape: return []

        swap_shape = self.shape_swaps.get(original_shape)
        if not swap_shape: return []
        
        new_description = self._replace_with_article_correction(description, original_shape, swap_shape)
        ann_id_suffix = f"neg_shape_{swap_shape}"
        return [self._format_as_scanrefer(sample, new_description, ann_id_suffix)]

    def generate_negation_negatives(self, sample: Dict, scene_graph: Dict):
        description = sample['description']
        object_name = sample['object_name'].replace('_', ' ')
        target_obj = scene_graph['objects'].get(sample['object_id'])
        if not target_obj: return []

        original_color = target_obj['color_info'].get('dominant_color')
        if not original_color or original_color in ['unknown', 'other'] or original_color not in description.lower(): return []
        if any(obj['category'] == target_obj['category'] and obj['color_info'].get('dominant_color') != original_color for _, obj in scene_graph['objects'].items()): return []
        
        doc = self.nlp(description)
        # 优先在核心目标的名词短语中进行否定
        target_chunk = next((chunk for chunk in doc.noun_chunks if object_name in chunk.text.lower() and original_color in chunk.text.lower()), None)
        if not target_chunk: return []
        
        chunk_without_color = self._replace_with_article_correction(target_chunk.text, original_color, "")
        new_chunk_text = f"{chunk_without_color.strip()} that is not {original_color}"
        new_description = description.replace(target_chunk.text, new_chunk_text, 1)
        ann_id_suffix = f"neg_negation_{original_color}"
        return [self._format_as_scanrefer(sample, new_description, ann_id_suffix)]

    def generate_relational_negatives(self, sample: Dict, scene_graph: Dict):
        description = sample['description'].lower()
        original_relation = next((phrase for phrase in self.relation_phrases if phrase in description), None)
        if not original_relation: return []
        
        swap_candidates = self.relation_swaps.get(original_relation, [])
        if not swap_candidates: return []
        
        new_relation = random.choice(swap_candidates)
        new_description = description.replace(original_relation, new_relation, 1)
        # 关系对立的验证较为复杂，暂简化为直接生成
        ann_id_suffix = f"neg_rel_{new_relation.replace(' ', '_')}"
        return [self._format_as_scanrefer(sample, new_description, ann_id_suffix)]
    # ==============================================================================

    def _get_description_complexity(self, description: str) -> int:
        score = 0
        desc_lower = description.lower()
        if any(word in desc_lower for word in self.color_helper.knowledge_base): score += 1
        if any(re.search(r'\b' + word + r'\b', desc_lower) for word in self.size_adjectives): score += 1
        if any(re.search(r'\b' + word + r'\b', desc_lower) for word in self.shape_adjectives): score += 1
        if any(phrase in desc_lower for phrase in self.relation_phrases): score += 1
        return score

    def generate(self, output_file: str):
        all_negatives = []
        print("Starting generation with complexity-aware, multi-negative logic...")

        strategies = [
            self.generate_attribute_negatives, 
            self.generate_relational_negatives,
            self.generate_negation_negatives,
            self.generate_size_negatives,
            self.generate_shape_negatives,
        ]
        
        strategy_success_count = {func.__name__: 0 for func in strategies}

        for sample in tqdm(self.scannet_data, desc="Processing ScanRefer samples"):
            scene_graph = self._load_scene_graph(sample['scene_id'])
            if not scene_graph: continue
            
            complexity_score = self._get_description_complexity(sample['description'])
            if complexity_score >= 3: target_count = 3
            elif complexity_score == 2: target_count = 2
            else: target_count = 1

            generated_negs_for_sample = []
            random.shuffle(strategies)
            
            for strategy_func in strategies:
                if len(generated_negs_for_sample) >= target_count: break
                
                negs = strategy_func(sample, scene_graph)
                if negs:
                    # 检查新生成的负样本与已有的不重复
                    if negs[0]['description'] not in [n['description'] for n in generated_negs_for_sample]:
                        generated_negs_for_sample.extend(negs)
                        strategy_success_count[strategy_func.__name__] += 1
            
            all_negatives.extend(generated_negs_for_sample)
        
        print(f"\nGeneration complete. Total true negative samples: {len(all_negatives)}")
        print("Strategy success distribution:")
        for name, count in sorted(strategy_success_count.items(), key=lambda item: item[1], reverse=True):
            if count > 0: print(f"  - {name}: {count}")

        with open(output_file, 'w') as f:
            json.dump(all_negatives, f, indent=4)
        print(f"True negative samples saved to {output_file}")


if __name__ == "__main__":
    SCANREFER_TRAIN_FILE = "../ScanRefer/ScanRefer_filtered_train.json"
    SCENE_GRAPH_DIR = "../build_scene/scene_graphs"
    OUTPUT_FILE = "./true_negatives_train_v23_multi_neg.json" 

    if not os.path.exists(SCANREFER_TRAIN_FILE):
        print(f"Error: ScanRefer file not found at {SCANREFER_TRAIN_FILE}")
    elif not os.path.exists(SCENE_GRAPH_DIR):
        print(f"Error: Scene graph directory not found at {SCENE_GRAPH_DIR}")
    else:
        generator = TrueNegativeGenerator(
            scannet_file=SCANREFER_TRAIN_FILE,
            scene_graph_dir=SCENE_GRAPH_DIR
        )
        generator.generate(OUTPUT_FILE)
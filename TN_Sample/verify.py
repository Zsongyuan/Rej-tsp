# ==============================================================================
# verify.py (v2.2 - Stricter Property Change Validation)
#
# 主要改进:
# 1. 强化 _verify_property_change 函数，强制要求原始描述中必须存在
#    要修改的同类型属性。
# 2. 解决了 v2.1 版本中允许 LLM "添加" 或 "交换" 属性而非 "修改" 的漏洞。
# ==============================================================================

import json
import os
import spacy
from collections import Counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Set
from multiprocessing import Pool, cpu_count
from functools import partial

def normalize_category_name(name: str) -> str:
    """标准化类别名称，用于比较。"""
    return name.replace('_', ' ').strip().lower()

class SpaCyParser:
    """使用 spaCy 进行文本解析的封装类。"""
    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp
        self.COLOR_LEMMAS: Set[str] = {'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'cyan', 'beige', 'black', 'white', 'gray', 'silver', 'gold'}
        self.SIZE_LEMMAS: Set[str] = {'tiny', 'small', 'medium', 'large', 'huge', 'big', 'tall'}
        
        self.SIMPLE_PREPOSITIONS: Set[str] = {'above', 'behind', 'below', 'beneath', 'beside', 'between', 'near', 'on', 'under'}
        self.COMPLEX_RELATION_MODIFIERS: Set[str] = {'front', 'back', 'left', 'right', 'top', 'side'}

    def parse_property(self, description: str, prop_lemmas: set) -> Optional[Dict[str, str]]:
        """从描述中解析属性（如颜色、大小）及其修饰的对象。"""
        doc = self.nlp(description)
        for token in doc:
            if token.lemma_ in prop_lemmas:
                if token.dep_ == 'amod' and token.head.pos_ == 'NOUN':
                    return {"property": token.lemma_, "object": token.head.lemma_}
                if token.dep_ == 'compound' and token.head.pos_ == 'NOUN':
                    return {"property": token.lemma_, "object": token.head.lemma_}
                if token.dep_ == 'attr' and token.head.lemma_ == 'be':
                    subjects = [child for child in token.head.children if child.dep_ in ('nsubj', 'nsubjpass')]
                    if subjects and subjects[0].pos_ == 'NOUN':
                        return {"property": token.lemma_, "object": subjects[0].lemma_}
        return None

    def parse_cardinality(self, description: str) -> Optional[Dict[str, Any]]:
        """从描述中解析数量和对象。"""
        doc = self.nlp(description)
        for token in doc:
            if token.like_num:
                if token.dep_ == 'nummod' and token.head.pos_ == 'NOUN':
                    count_str = token.text.lower()
                    num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "a": 1, "an": 1}
                    numeric_value = num_map.get(count_str, int(count_str) if count_str.isdigit() else None)
                    if numeric_value:
                        return {"count": numeric_value, "object": token.head.lemma_}
        return None

    def parse_spatial_relation(self, description: str, scene_nouns: Set[str]) -> Optional[Dict[str, str]]:
        """
        (已重构) 从描述中解析一个具体的、可验证的空间关系。
        能同时处理 "behind the shelf" 和 "to the left of the table" 两种结构。
        """
        doc = self.nlp(description)
        
        for token in doc:
            if token.lemma_ in self.SIMPLE_PREPOSITIONS:
                pobjs = [child for child in token.children if child.dep_ == 'pobj']
                if pobjs and pobjs[0].lemma_ in scene_nouns:
                    verb = token.head
                    if verb.lemma_ == 'be':
                        subjects = [child for child in verb.children if child.dep_ == 'nsubj']
                        if subjects:
                            return {
                                "subject": subjects[0].lemma_,
                                "relation": token.lemma_,
                                "object": pobjs[0].lemma_
                            }
            
            if token.lemma_ == 'of':
                modifier = token.head
                if modifier.lemma_ in self.COMPLEX_RELATION_MODIFIERS:
                    pobjs = [child for child in token.children if child.dep_ == 'pobj']
                    if pobjs and pobjs[0].lemma_ in scene_nouns:
                        prep = modifier.head
                        if prep.pos_ == 'ADP': 
                            verb = prep.head
                            if verb.lemma_ == 'be':
                                subjects = [child for child in verb.children if child.dep_ == 'nsubj']
                                if subjects:
                                    return {
                                        "subject": subjects[0].lemma_,
                                        "relation": f"{prep.lemma_} {modifier.lemma_} of",
                                        "object": pobjs[0].lemma_
                                    }
        return None


class NegativeSampleVerifier:
    def __init__(self, scene_graph_dir: str, load_spacy_model: bool = True):
        self.scene_graph_dir = scene_graph_dir
        self.scene_graphs: Dict[str, Dict] = {}
        self.nlp = None
        self.parser = None
        
        if load_spacy_model:
            self._load_spacy()

    def _load_spacy(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading 'en_core_web_sm'... Please wait.")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.parser = SpaCyParser(self.nlp)

    def _load_scene_graph(self, scene_id: str) -> Dict[str, Any]:
        if scene_id in self.scene_graphs:
            return self.scene_graphs[scene_id]
        file_path = os.path.join(self.scene_graph_dir, f"{scene_id}_scene_graph.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Scene graph not found for {scene_id} at {file_path}")
        with open(file_path, 'r') as f:
            graph = json.load(f)
            self.scene_graphs[scene_id] = graph
            return graph
            
    def _normalize_text_for_comparison(self, text: Any) -> str:
        if not isinstance(text, str): return ""
        return ' '.join(text.lower().split())

    def verify_sample(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        if self.nlp is None:
            self._load_spacy()
            
        new_desc = sample.get('description')
        orig_desc = sample.get('origDescrip')
        strategy = sample.get('generation_strategy')

        if not all([new_desc, orig_desc, strategy]):
            return False, "Validation Failed: Sample is missing essential keys."

        if self._normalize_text_for_comparison(new_desc) == self._normalize_text_for_comparison(orig_desc):
            return False, "Validation Failed: Description is effectively unchanged."

        if "Add Negation" in strategy and "not" not in new_desc.lower() and "n't" not in new_desc.lower():
            return False, "Validation Failed: Strategy/content mismatch for 'Add Negation'."

        try:
            scene_graph = self._load_scene_graph(sample['scene_id'])
        except FileNotFoundError as e:
            return False, str(e)

        dispatch_map = {
            "Add Negation": self._verify_negation,
            "Change Color": self._verify_property_change,
            "Change Spatial Relation": self._verify_spatial_relation,
            "Change Cardinality (count)": self._verify_cardinality_change,
            "Change Size (Very Conservative)": self._verify_property_change,
        }
        
        selected_method = None
        for key, method in dispatch_map.items():
            if strategy.startswith(key):
                if "Color" in key:
                    selected_method = partial(method, prop_lemmas=self.parser.COLOR_LEMMAS, prop_name="Color")
                elif "Size" in key:
                    selected_method = partial(method, prop_lemmas=self.parser.SIZE_LEMMAS, prop_name="Size")
                else:
                    selected_method = method
                break

        if not selected_method:
            return False, f"Unknown or unsupported strategy: '{strategy}'"
            
        return selected_method(sample, scene_graph)

    def _verify_negation(self, sample: Dict, scene_graph: Dict) -> Tuple[bool, str]:
        new_desc = sample['description']
        orig_desc = sample['origDescrip']

        doc = self.nlp(new_desc)
        target_noun_lemma = self.nlp(sample['object_name'].replace('_', ' '))[0].lemma_
        for token in doc:
            if token.dep_ == 'neg' and token.head.lemma_ == 'be':
                for child in token.head.children:
                    if child.dep_ == 'attr' and child.lemma_ == target_noun_lemma:
                        return False, f"Validation Failed: Negated object identity '{target_noun_lemma}'."
        
        orig_prop_lemmas = {t.lemma_ for t in self.nlp(orig_desc) if t.lemma_ in self.parser.COLOR_LEMMAS or t.lemma_ in self.parser.SIZE_LEMMAS}
        new_prop_lemmas = {t.lemma_ for t in self.nlp(new_desc) if t.lemma_ in self.parser.COLOR_LEMMAS or t.lemma_ in self.parser.SIZE_LEMMAS}
        
        hallucinated_props = new_prop_lemmas - orig_prop_lemmas
        if hallucinated_props:
            return False, f"Validation Failed: Negation introduced a new hallucinated attribute '{list(hallucinated_props)[0]}'."

        return True, "Validation Passed (Negation)"

    # ================== START: MODIFIED FOR v2.2 ==================
    def _verify_property_change(self, sample: Dict, scene_graph: Dict, prop_lemmas: Set[str], prop_name: str) -> Tuple[bool, str]:
        original_desc = sample['origDescrip']
        new_desc = sample['description']

        # 1. 严格前提检查：原始描述必须包含一个可供“修改”的同类型属性。
        parsed_orig = self.parser.parse_property(original_desc, prop_lemmas)
        if not parsed_orig:
            return False, f"{prop_name} check failed: Precondition failed - original description has no parsable {prop_name.lower()} to change."

        # 2. 类型匹配检查：新描述也必须包含该属性，防止属性被删除。
        parsed_new = self.parser.parse_property(new_desc, prop_lemmas)
        if not parsed_new:
            return False, f"{prop_name} check failed: New description has no parsable {prop_name.lower()}. Attribute may have been deleted or swapped."

        # 3. 检查被描述的对象是否一致
        if parsed_orig['object'] != parsed_new['object']:
            return False, f"{prop_name} check failed: Object being described changed from '{parsed_orig['object']}' to '{parsed_new['object']}'."

        # 4. 检查属性值是否真的改变了
        if parsed_orig['property'] == parsed_new['property']:
            return False, f"Validation Failed: {prop_name} property '{parsed_new['property']}' was not changed."

        # 5. 检查与场景真值的冲突
        obj_id = str(sample.get('object_id'))
        target_obj = scene_graph['objects'].get(obj_id)
        if not target_obj: return False, f"{prop_name} check failed: Object ID {obj_id} not in scene graph"
        
        true_prop = ""
        if prop_name == "Color": true_prop = target_obj.get('color_info', {}).get('dominant_color')
        elif prop_name == "Size": true_prop = target_obj.get('size_rank', {}).get('size_descriptor')
        
        if true_prop and parsed_new['property'] == true_prop:
            return False, f"Validation Failed: New {prop_name.lower()} is same as true {prop_name.lower()}."

        # 6. 歧义性检查
        target_category_normalized = normalize_category_name(target_obj['category'])
        for other_id, other_obj in scene_graph['objects'].items():
            if other_id == obj_id: continue
            other_category_normalized = normalize_category_name(other_obj['category'])
            
            other_prop = ""
            if prop_name == "Color": other_prop = other_obj.get('color_info', {}).get('dominant_color')
            elif prop_name == "Size": other_prop = other_obj.get('size_rank', {}).get('size_descriptor')

            if other_prop and (other_category_normalized == target_category_normalized and other_prop == parsed_new['property']):
                return False, f"Validation Failed: Ambiguous - another object matches new description."
        
        return True, f"Validation Passed ({prop_name})"
    # =================== END: MODIFIED FOR v2.2 ===================


    def _verify_cardinality_change(self, sample: Dict, scene_graph: Dict) -> Tuple[bool, str]:
        parsed_new = self.parser.parse_cardinality(sample['description'])
        if not parsed_new: return False, "Cardinality check failed: spaCy could not parse number and object."
        
        key_from_desc = normalize_category_name(parsed_new['object'])
        scene_categories = [normalize_category_name(obj['category']) for obj in scene_graph['objects'].values()]
        category_counts = Counter(scene_categories)
        true_count = category_counts.get(key_from_desc, 0)
        
        if parsed_new['count'] == true_count:
            return False, f"Cardinality check failed: New count is same as true count."
        
        return True, "Validation Passed (Cardinality)"

    def _verify_spatial_relation(self, sample: Dict, scene_graph: Dict) -> Tuple[bool, str]:
        original_desc = sample['origDescrip']
        new_desc = sample['description']

        scene_object_lemmas = {self.nlp(normalize_category_name(obj['category']))[0].lemma_ for obj in scene_graph['objects'].values()}
        
        parsed_orig = self.parser.parse_spatial_relation(original_desc, scene_object_lemmas)
        if not parsed_orig:
            return False, "Relation check failed: Precondition failed - no clear, verifiable spatial relation found in original description."

        parsed_new = self.parser.parse_spatial_relation(new_desc, scene_object_lemmas)
        if not parsed_new:
            return False, "Relation check failed: No clear, verifiable spatial relation found in new description. Type may have swapped."
        
        if parsed_orig['subject'] != parsed_new['subject']:
            main_object_lemma = self.nlp(sample['object_name'])[0].lemma_
            if not (parsed_orig['subject'] in ('it', main_object_lemma) and parsed_new['subject'] in ('it', main_object_lemma)):
                 return False, f"Relation check failed: Subject of relation changed from '{parsed_orig['subject']}' to '{parsed_new['subject']}'."

        if parsed_orig['relation'] == parsed_new['relation'] and parsed_orig['object'] == parsed_new['object']:
            return False, "Relation check failed: The spatial relation appears to be unchanged."

        return True, "Validation Passed (Relation)"


# ==============================================================================
# Main execution logic (no changes)
# ==============================================================================
verifier_process_global = None
def process_sample_worker(sample: Dict, scene_graph_dir: str) -> Tuple[bool, str, Dict]:
    global verifier_process_global
    if verifier_process_global is None:
        verifier_process_global = NegativeSampleVerifier(scene_graph_dir, load_spacy_model=True)
    is_valid, reason = verifier_process_global.verify_sample(sample)
    return is_valid, reason, sample

if __name__ == "__main__":
    GENERATED_NEGATIVES_FILE = "./llm_generated_negatives_v3.json"
    SCENE_GRAPH_DIR = "./scene_graphs"
    VERIFIED_OUTPUT_FILE = "./final_verified_negatives.json"
    REJECTED_OUTPUT_FILE = "./final_rejected_negatives.json"

    if not os.path.exists(GENERATED_NEGATIVES_FILE):
        print(f"Error: Input file not found at {GENERATED_NEGATIVES_FILE}")
    elif not os.path.exists(SCENE_GRAPH_DIR):
        print(f"Error: Scene graph directory not found at {SCENE_GRAPH_DIR}")
    else:
        with open(GENERATED_NEGATIVES_FILE, 'r') as f:
            all_samples = json.load(f)
        
        verified_samples = []
        rejected_samples = []

        print(f"\nStarting final verification for {len(all_samples)} samples with stricter property validation...")
        try:
            spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' is available.")
        except OSError:
            print("Downloading 'en_core_web_sm' for worker processes... Please wait.")
            spacy.cli.download("en_core_web_sm")

        num_processes = cpu_count()
        print(f"Using {num_processes} processes for verification.")
        task_func = partial(process_sample_worker, scene_graph_dir=SCENE_GRAPH_DIR)
        
        with Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(task_func, all_samples)
            for is_valid, reason, sample in tqdm(results_iterator, total=len(all_samples), desc="Verifying Samples"):
                if is_valid:
                    verified_samples.append(sample)
                else:
                    rejected_info = sample.copy()
                    rejected_info['rejection_reason'] = reason
                    rejected_samples.append(rejected_info)
        
        with open(VERIFIED_OUTPUT_FILE, 'w') as f: json.dump(verified_samples, f, indent=2)
        with open(REJECTED_OUTPUT_FILE, 'w') as f: json.dump(rejected_samples, f, indent=2)
            
        print("\nVerification Complete.")
        print(f"  {len(verified_samples)} samples PASSED. Saved to {VERIFIED_OUTPUT_FILE}")
        print(f"  {len(rejected_samples)} samples FAILED. Saved to {REJECTED_OUTPUT_FILE}")
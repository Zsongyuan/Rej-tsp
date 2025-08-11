import json
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
import colorsys  # 保留以防需要，但本次修改未用
from dataclasses import dataclass, asdict
import os

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)
    
@dataclass
class Object3D:
    """3D物体的完整表示"""
    object_id: int
    instance_id: int
    category: str
    bbox: np.ndarray  # [x_min, y_min, z_min, x_max, y_max, z_max]
    center: np.ndarray  # [x, y, z]
    size: np.ndarray  # [width, height, depth]
    volume: float
    color_info: Dict  # 包含主色、是否单色等信息
    size_rank: Dict  # 相对尺寸信息
    spatial_relations: List[Dict]  # 与其他物体的空间关系

class SceneGraphBuilder:
    """构建ScanNet场景的全局场景图"""
    
    def __init__(self, scannet_path: str):
        self.scannet_path = scannet_path
        
        # RGB颜色名称映射（RGB范围到颜色名，基于常见计算机视觉阈值）
        self.rgb_color_names = {
            'red': (150, 255, 0, 100, 0, 100),      # R高, G/B低
            'green': (0, 100, 150, 255, 0, 100),
            'blue': (0, 100, 0, 100, 150, 255),
            'yellow': (150, 255, 150, 255, 0, 100),
            'orange': (150, 255, 50, 150, 0, 50),
            'purple': (100, 200, 0, 100, 150, 255),
            'pink': (150, 255, 50, 150, 100, 200),
            'brown': (50, 150, 0, 100, 0, 50),
            'cyan': (0, 100, 150, 255, 150, 255),
            'beige': (150, 255, 150, 255, 100, 200)
        }
        
        # 添加典型RGB中心值，用于相似度计算
        self.color_centers = {
            'red': (200, 50, 50),
            'green': (50, 200, 50),
            'blue': (50, 50, 200),
            'yellow': (200, 200, 50),
            'orange': (200, 100, 50),
            'purple': (150, 50, 200),
            'pink': (200, 100, 150),
            'brown': (100, 50, 25),
            'black': (25, 25, 25),
            'white': (225, 225, 225),
            'gray': (125, 125, 125),
            'cyan': (50, 200, 200),
            'beige': (200, 200, 150)
        }
        
        # 尺寸形容词（未修改）
        self.size_descriptors = {
            'tiny': (0, 0.1),      # 前10%
            'small': (0.1, 0.3),   # 10-30%
            'medium': (0.3, 0.7),  # 30-70%
            'large': (0.7, 0.9),   # 70-90%
            'huge': (0.9, 1.0)     # 前10%
        }
        
    def build_scene_graph(self, scan_id: str) -> Dict:
        """构建完整的场景图"""
        print(f"Building scene graph for {scan_id}...")
        
        # 1. 加载场景数据
        scene_data = self._load_scene_data(scan_id)
        
        # 2. 提取物体信息
        objects = self._extract_objects(scene_data, scan_id)
        
        # 3. 分析颜色属性（已改为RGB）
        self._analyze_colors(objects, scene_data, scan_id)
        
        # 4. 计算相对尺寸
        self._compute_relative_sizes(objects)
        
        # 5. 建立空间关系
        self._compute_spatial_relations(objects)
        
        # 6. 构建最终场景图
        scene_graph = self._build_final_graph(scan_id, objects)
        
        return scene_graph
    
    def _load_scene_data(self, scan_id: str) -> Dict:
        """加载ScanNet场景数据"""
        scene_path = os.path.join(self.scannet_path, "scans", scan_id)
        
        # 加载聚合信息
        aggregation_file = os.path.join(scene_path, f"{scan_id}.aggregation.json")
        with open(aggregation_file, 'r') as f:
            aggregation = json.load(f)
        
        # 加载分割信息
        segs_file = os.path.join(scene_path, f"{scan_id}_vh_clean_2.0.010000.segs.json")
        with open(segs_file, 'r') as f:
            segmentation = json.load(f)
        
        # 加载元信息
        txt_file = os.path.join(scene_path, f"{scan_id}.txt")
        metadata = {}
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                for line in f:
                    if ' = ' in line:
                        key, value = line.strip().split(' = ')
                        metadata[key] = value
        
        return {
            'aggregation': aggregation,
            'segmentation': segmentation,
            'metadata': metadata
        }
    
    def _extract_objects(self, scene_data: Dict, scan_id: str) -> List[Object3D]:
        """从场景数据中提取物体信息"""
        objects = []
        
        for seg_group in scene_data['aggregation']['segGroups']:
            # 计算边界框
            segments = seg_group['segments']
            bbox = self._compute_bbox_from_segments(segments, scene_data, scan_id)
            
            if bbox is None:
                continue
            
            center = (bbox[:3] + bbox[3:]) / 2
            size = bbox[3:] - bbox[:3]
            volume = np.prod(size)
            
            obj = Object3D(
                object_id=seg_group['objectId'],
                instance_id=seg_group['id'],
                category=seg_group['label'],
                bbox=bbox,
                center=center,
                size=size,
                volume=volume,
                color_info={},
                size_rank={},
                spatial_relations=[]
            )
            
            objects.append(obj)
        
        return objects
    
    def _compute_bbox_from_segments(self, segments: List[int], 
                                  scene_data: Dict, 
                                  scan_id: str) -> Optional[np.ndarray]:
        """从分割计算边界框"""
        scene_path = os.path.join(self.scannet_path, "scans", scan_id)
        
        # 加载点云
        ply_file = os.path.join(scene_path, f"{scan_id}_vh_clean_2.ply")
        mesh = o3d.io.read_triangle_mesh(ply_file)
        vertices = np.asarray(mesh.vertices)
        
        # 加载顶点到分割的映射
        seg_indices = scene_data['segmentation']['segIndices']
        
        # 收集属于这些段的所有顶点
        object_vertices = []
        for i, seg_id in enumerate(seg_indices):
            if seg_id in segments:
                if i < len(vertices):
                    object_vertices.append(vertices[i])
        
        if not object_vertices:
            return None
        
        object_vertices = np.array(object_vertices)
        min_coords = np.min(object_vertices, axis=0)
        max_coords = np.max(object_vertices, axis=0)
        
        return np.concatenate([min_coords, max_coords])
    
    def _analyze_colors(self, objects: List[Object3D], 
                       scene_data: Dict, 
                       scan_id: str):
        """分析物体颜色属性"""
        scene_path = os.path.join(self.scannet_path, "scans", scan_id)
        
        # 加载带颜色的点云
        ply_file = os.path.join(scene_path, f"{scan_id}_vh_clean_2.ply")
        mesh = o3d.io.read_triangle_mesh(ply_file)
        vertices = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)  # RGB in [0,1]
        
        seg_indices = scene_data['segmentation']['segIndices']
        
        for obj in objects:
            # 获取物体的所有颜色
            segments = None
            for seg_group in scene_data['aggregation']['segGroups']:
                if seg_group['objectId'] == obj.object_id:
                    segments = seg_group['segments']
                    break
            
            if segments is None:
                obj.color_info = {'dominant_color': 'unknown', 'is_single_color': False}
                continue
            
            # 收集物体的所有颜色
            object_colors = []
            for i, seg_id in enumerate(seg_indices):
                if seg_id in segments and i < len(colors):
                    object_colors.append(colors[i])
            
            if not object_colors:
                obj.color_info = {'dominant_color': 'unknown', 'is_single_color': False}
                continue
            
            object_colors = np.array(object_colors)
            
            # 分析颜色（已改为RGB）
            color_analysis = self._analyze_object_color(object_colors)
            obj.color_info = color_analysis
    
    def _analyze_object_color(self, colors: np.ndarray) -> Dict:
        """分析物体颜色，判断是否单色（使用RGB直方图分析）"""
        # 转换为RGB uint8
        colors_rgb = (colors * 255).astype(np.uint8)
        
        if len(colors_rgb) == 0:
            return {
                'dominant_color': 'unknown',
                'similar_colors': [],
                'is_single_color': False,
                'color_confidence': 0.0,
                'num_color_clusters': 0
            }
        
        # 计算每个通道的直方图（bins=32以减少噪声，覆盖0-255）
        bins = 32
        hist_r, bin_edges_r = np.histogram(colors_rgb[:, 0], bins=bins, range=(0, 255))
        hist_g, bin_edges_g = np.histogram(colors_rgb[:, 1], bins=bins, range=(0, 255))
        hist_b, bin_edges_b = np.histogram(colors_rgb[:, 2], bins=bins, range=(0, 255))
        
        # 找到每个通道的主导峰值（bin中心作为代表值）
        peak_r_idx = np.argmax(hist_r)
        peak_g_idx = np.argmax(hist_g)
        peak_b_idx = np.argmax(hist_b)
        
        # 计算峰值RGB（使用bin中心）
        peak_r = (bin_edges_r[peak_r_idx] + bin_edges_r[peak_r_idx + 1]) / 2
        peak_g = (bin_edges_g[peak_g_idx] + bin_edges_g[peak_g_idx + 1]) / 2
        peak_b = (bin_edges_b[peak_b_idx] + bin_edges_b[peak_b_idx + 1]) / 2
        
        dominant_rgb = np.array([peak_r, peak_g, peak_b])
        
        # 将RGB转换为颜色名称，包括主色和相近颜色
        color_name, similar_colors = self._rgb_to_color_name(dominant_rgb)
        
        # 计算主导峰值占比作为置信度（总点数的比例）
        peak_r_count = hist_r[peak_r_idx]
        peak_g_count = hist_g[peak_g_idx]
        peak_b_count = hist_b[peak_b_idx]
        # 平均峰值占比
        dominant_ratio = (peak_r_count + peak_g_count + peak_b_count) / (3 * len(colors_rgb))
        
        # 判断是否单色：如果所有通道的峰值占比>70%且方差小
        is_single_color = dominant_ratio > 0.7 and np.std(colors_rgb, axis=0).mean() < 30  # 方差阈值
        
        # 估算颜色簇数：通道峰值数>1视为多簇（简化）
        num_peaks = len(np.where(hist_r > 0.1 * hist_r.max())[0]) + \
                    len(np.where(hist_g > 0.1 * hist_g.max())[0]) + \
                    len(np.where(hist_b > 0.1 * hist_b.max())[0])
        num_color_clusters = max(1, num_peaks // 3)
        
        return {
            'dominant_color': color_name,
            'similar_colors': similar_colors,
            'is_single_color': is_single_color,
            'color_confidence': dominant_ratio,
            'num_color_clusters': num_color_clusters
        }
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> Tuple[str, List[str]]:
        """将RGB值转换为颜色名称，包括主色和相近颜色"""
        r, g, b = rgb
        
        # 先检查特殊情况：黑、白、灰
        if r < 50 and g < 50 and b < 50:
            return 'black', ['gray']
        if r > 200 and g > 200 and b > 200:
            return 'white', ['beige', 'gray']
        if 50 <= r <= 200 and 50 <= g <= 200 and 50 <= b <= 200 and max(abs(r-g), abs(g-b), abs(r-b)) < 30:
            return 'gray', ['white', 'black']
        
        # 找到主色：匹配范围
        dominant_color = 'other'
        for color_name, (r_min, r_max, g_min, g_max, b_min, b_max) in self.rgb_color_names.items():
            if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
                dominant_color = color_name
                break
        
        # 计算相近颜色：基于欧几里德距离到颜色中心，前3个（排除主色）
        distances = {}
        for color, center in self.color_centers.items():
            dist = np.linalg.norm(rgb - np.array(center))
            distances[color] = dist
        
        # 排序距离，选前3个（包括主色，但返回时排除主色作为similar）
        sorted_colors = sorted(distances, key=distances.get)
        similar_colors = [c for c in sorted_colors[1:4] if c != dominant_color]  # 前3个相近，排除自身
        
        return dominant_color, similar_colors
    
    def _compute_relative_sizes(self, objects: List[Object3D]):
        """计算物体的相对尺寸"""
        if not objects:
            return
        
        # 按体积排序
        sorted_by_volume = sorted(objects, key=lambda x: x.volume)
        
        # 计算每个物体的尺寸百分位
        for i, obj in enumerate(sorted_by_volume):
            percentile = i / len(sorted_by_volume)
            
            # 确定尺寸描述词
            size_desc = 'medium'
            for desc, (min_p, max_p) in self.size_descriptors.items():
                if min_p <= percentile < max_p:
                    size_desc = desc
                    break
            
            obj.size_rank = {
                'percentile': percentile,
                'rank': i + 1,
                'total': len(objects),
                'size_descriptor': size_desc,
                'volume': obj.volume
            }
        
        # 按类别计算相对尺寸
        category_objects = defaultdict(list)
        for obj in objects:
            category_objects[obj.category].append(obj)
        
        for category, cat_objects in category_objects.items():
            sorted_cat = sorted(cat_objects, key=lambda x: x.volume)
            for i, obj in enumerate(sorted_cat):
                percentile = i / len(sorted_cat)
                
                size_desc = 'medium'
                for desc, (min_p, max_p) in self.size_descriptors.items():
                    if min_p <= percentile < max_p:
                        size_desc = desc
                        break
                
                obj.size_rank[f'{category}_percentile'] = percentile
                obj.size_rank[f'{category}_rank'] = i + 1
                obj.size_rank[f'{category}_total'] = len(sorted_cat)
                obj.size_rank[f'{category}_size'] = size_desc
    
    def _compute_spatial_relations(self, objects: List[Object3D], max_distance: float = 2.0):
        """
        计算物体间的空间关系（基于距离阈值）。
        每个物体只与在max_distance范围内的其他物体建立关系。
        """
        if not objects or len(objects) < 2:
            return

        obj_map = {obj.object_id: obj for obj in objects}
        
        # --- 步骤 1: 对每个物体，找出其距离内的所有邻居，并建立无向关系对 ---
        neighbor_pairs = set()
        for obj1 in objects:
            # 计算到所有其他物体的距离
            distances = []
            for obj2 in objects:
                if obj1.object_id == obj2.object_id:
                    continue
                distance = np.linalg.norm(obj1.center - obj2.center)
                distances.append((distance, obj2.object_id))
            
            # === 核心修改: 根据距离阈值选择邻居 ===
            target_ids = set()
            for distance, obj_id in distances:
                # 如果距离在设定的最大距离之内，则将其视为邻居
                if distance <= max_distance:
                    target_ids.add(obj_id)
            
            # 将关系对添加到集合中，使用排序后的元组确保唯一性和无向性
            for target_id in target_ids:
                pair = tuple(sorted((obj1.object_id, target_id)))
                neighbor_pairs.add(pair)

        # --- 步骤 2: 为每个对称的关系对双向计算关系 (此部分逻辑不变) ---
        for obj in objects:
            obj.spatial_relations = []
            
        inverse_map = {
            'left': 'right', 'right': 'left',
            'above': 'below', 'below': 'above',
            'front': 'behind', 'behind': 'front',
            'contains': 'inside', 'inside': 'contains'
        }
        
        for id1, id2 in neighbor_pairs:
            obj1 = obj_map[id1]
            obj2 = obj_map[id2]
            
            relations_1_to_2 = self._get_precise_relation(obj1, obj2)
            rel_pos_1_to_2 = obj2.center - obj1.center
            distance = np.linalg.norm(rel_pos_1_to_2)
            
            common_relations = []
            if distance < 0.5: common_relations.append('near')
            if self._is_touching(obj1, obj2): common_relations.append('touching')
            
            final_relations_1_to_2 = relations_1_to_2 + common_relations
            if self._is_inside(obj1, obj2):
                final_relations_1_to_2.append('inside')
            elif self._is_inside(obj2, obj1):
                final_relations_1_to_2.append('contains')

            final_relations_2_to_1 = [inverse_map.get(rel, rel) for rel in final_relations_1_to_2]

            if final_relations_1_to_2:
                obj1.spatial_relations.append({
                    'target_id': obj2.object_id,
                    'target_category': obj2.category,
                    'relations': sorted(list(set(final_relations_1_to_2))),
                    'distance': distance,
                    'relative_position': rel_pos_1_to_2.tolist()
                })

            if final_relations_2_to_1:
                obj2.spatial_relations.append({
                    'target_id': obj1.object_id,
                    'target_category': obj1.category,
                    'relations': sorted(list(set(final_relations_2_to_1))),
                    'distance': distance,
                    'relative_position': (-rel_pos_1_to_2).tolist()
                })

    def _get_precise_relation(self, obj1: Object3D, obj2: Object3D, margin: float = 0.05) -> List[str]:
        """
        计算两个物体之间最精确的方向关系。(统一逻辑版)
        只有当物体在次要轴上对齐时，才判断其主要方向关系。
        """
        bbox1, bbox2 = obj1.bbox, obj2.bbox

        # 1. 为每个轴定义重叠条件，增加一点容差(margin)
        x_overlap = (bbox1[0] < bbox2[3] + margin) and (bbox2[0] < bbox1[3] + margin)
        y_overlap = (bbox1[1] < bbox2[4] + margin) and (bbox2[1] < bbox1[4] + margin)
        z_overlap = (bbox1[2] < bbox2[5] + margin) and (bbox2[2] < bbox1[5] + margin)

        # 2. 检查是否存在一个清晰、无歧义的主要方向关系
        # 案例一: 清晰的“上/下”关系 (在XY平面上对齐，在Z轴上分离)
        if x_overlap and y_overlap and not z_overlap:
            # obj1的底部在obj2的顶部之上 -> obj1在obj2上方
            if bbox1[2] > bbox2[5]:
                return ['above']
            # obj2的底部在obj1的顶部之上 -> obj1在obj2下方
            else:
                return ['below']

        # 案例二: 清晰的“左/右”关系 (在YZ平面上对齐，在X轴上分离)
        if y_overlap and z_overlap and not x_overlap:
            # obj1的左边界在obj2的右边界之右 -> obj1在obj2右方
            if bbox1[0] > bbox2[3]:
                return ['right']
            # obj2的左边界在obj1的右边界之右 -> obj1在obj2左方
            else:
                return ['left']

        # 案例三: 清晰的“前/后”关系 (在XZ平面上对齐，在Y轴上分离)
        if x_overlap and z_overlap and not y_overlap:
            # obj1的前边界在obj2的后边界之后 (Y值更大) -> obj1在obj2后方
            if bbox1[1] > bbox2[4]:
                return ['behind']
            # obj2的前边界在obj1的后边界之后 -> obj1在obj2前方
            else:
                return ['front']

        # 3. 后备逻辑: 处理完全重叠或对角线等模糊情况
        # 使用连接两个物体中心的向量的主导轴作为判断依据
        rel_pos = obj2.center - obj1.center
        size1 = obj1.size + 1e-6 # 防止除以零
        normalized_rel_pos = np.abs(rel_pos / size1)
        dominant_axis_idx = np.argmax(normalized_rel_pos)
        
        relations = []
        # rel_pos = obj2 - obj1
        # 关系是描述 obj1 相对于 obj2
        if dominant_axis_idx == 0: # X轴主导
            relations.append('left' if rel_pos[0] > 0 else 'right')
        elif dominant_axis_idx == 1: # Y轴主导
            relations.append('front' if rel_pos[1] > 0 else 'behind')
        elif dominant_axis_idx == 2: # Z轴主导
            relations.append('below' if rel_pos[2] > 0 else 'above')
            
        return relations
    
    def _is_inside(self, obj1: Object3D, obj2: Object3D) -> bool:
        """判断obj1是否在obj2内部"""
        return (obj1.bbox[:3] >= obj2.bbox[:3]).all() and \
               (obj1.bbox[3:] <= obj2.bbox[3:]).all()
    
    def _is_touching(self, obj1: Object3D, obj2: Object3D, 
                    threshold: float = 0.05) -> bool:
        """判断两个物体是否接触"""
        # 检查边界框是否重叠或非常接近
        separation = np.maximum(
            obj1.bbox[:3] - obj2.bbox[3:],
            obj2.bbox[:3] - obj1.bbox[3:]
        )
        return np.all(separation <= threshold)
    
    def _build_final_graph(self, scan_id: str, 
                          objects: List[Object3D]) -> Dict:
        """构建最终的场景图"""
        # 转换物体为字典格式
        objects_dict = {}
        for obj in objects:
            obj_dict = asdict(obj)
            
            # 将numpy数组转换为列表
            for key in ['bbox', 'center', 'size']:
                if key in obj_dict and isinstance(obj_dict[key], np.ndarray):
                    obj_dict[key] = obj_dict[key].tolist()
            
            objects_dict[obj.object_id] = obj_dict
        
        # 构建关系列表
        all_relations = []
        for obj in objects:
            for rel in obj.spatial_relations:
                all_relations.append({
                    'subject_id': obj.object_id,
                    'object_id': rel['target_id'],
                    'predicates': rel['relations'],
                    'distance': rel['distance']
                })
        
        # 统计信息
        stats = {
            'num_objects': len(objects),
            'categories': list(set(obj.category for obj in objects)),
            'single_color_objects': sum(1 for obj in objects 
                                      if obj.color_info.get('is_single_color', False)),
            'num_relations': len(all_relations)
        }
        
        return {
            'scan_id': scan_id,
            'objects': objects_dict,
            'relations': all_relations,
            'stats': stats
        }
    
    def save_scene_graph(self, scene_graph, file_path):
        """Saves a scene graph to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(scene_graph, f, indent=2, cls=NumpyEncoder) # 使用自定义的编码器


# 批量处理脚本
def process_all_scenes(scannet_path: str, output_dir: str, 
                      scene_list_file: Optional[str] = None):
    """批量处理所有场景"""
    builder = SceneGraphBuilder(scannet_path)
    
    # 获取场景列表
    if scene_list_file:
        with open(scene_list_file, 'r') as f:
            scene_ids = [line.strip() for line in f]
    else:
        scans_dir = os.path.join(scannet_path, "scans")
        scene_ids = [d for d in os.listdir(scans_dir) 
                    if os.path.isdir(os.path.join(scans_dir, d))]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个场景
    for scene_id in scene_ids:
        try:
            print(f"\nProcessing {scene_id}...")
            scene_graph = builder.build_scene_graph(scene_id)
            
            output_file = os.path.join(output_dir, f"{scene_id}_scene_graph.json")
            builder.save_scene_graph(scene_graph, output_file)
            
            # 打印统计信息
            stats = scene_graph['stats']
            print(f"  Objects: {stats['num_objects']}")
            print(f"  Single-color objects: {stats['single_color_objects']}")
            print(f"  Relations: {stats['num_relations']}")
            
        except Exception as e:
            print(f"Error processing {scene_id}: {str(e)}")
            continue
    
    print(f"\nAll scene graphs saved to {output_dir}")


# 使用示例
if __name__ == "__main__":
    # 单个场景
    '''scannet_path = "/path/to/scannet"
    builder = SceneGraphBuilder(scannet_path)
    
    scene_graph = builder.build_scene_graph("scene0000_00")
    builder.save_scene_graph(scene_graph, "scene0000_00_graph.json")'''
    
    # 批量处理
    process_all_scenes(
        scannet_path="/home/zhu/multimodal/TSP3D-main/scannet/",
        output_dir="./scene_graphs",
        # scene_list_file="train_scenes.txt"  # 可选
    )
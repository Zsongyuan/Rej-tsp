import json
import numpy as np
from plyfile import PlyData

def build_scene_graph(scan_path, scan_id):
    """
    Builds a structured scene graph from ScanNet scene data.
    
    Args:
        scan_path (str): Path to the directory containing the ScanNet scan files (e.g., '/path/to/scannet/scans/scene0000_00').
        scan_id (str): The scan ID (e.g., 'scene0000_00').
    
    Returns:
        list: A list of dictionaries, each representing an object with keys: 'id', 'category', 'position', 'size', 'color'.
    """
    # File paths based on ScanNet conventions
    mesh_file = f"{scan_path}/{scan_id}_vh_clean_2.ply"
    segs_file = f"{scan_path}/{scan_id}_vh_clean_2.0.010000.segs.json"
    agg_file = f"{scan_path}/{scan_id}.aggregation.json"
    
    # Load the mesh PLY file for vertices and colors
    plydata = PlyData.read(mesh_file)
    num_vertices = len(plydata['vertex'])
    vertices = np.zeros((num_vertices, 6))
    vertices[:, 0] = plydata['vertex']['x']
    vertices[:, 1] = plydata['vertex']['y']
    vertices[:, 2] = plydata['vertex']['z']
    vertices[:, 3] = plydata['vertex']['red']
    vertices[:, 4] = plydata['vertex']['green']
    vertices[:, 5] = plydata['vertex']['blue']
    
    # Load the segmentation indices from JSON
    with open(segs_file, 'r') as f:
        segs = json.load(f)
    seg_indices = np.array(segs['segIndices'])
    
    # Load the aggregation JSON for object groups
    with open(agg_file, 'r') as f:
        agg = json.load(f)
    
    # Build the scene graph
    scene_graph = []
    for group in agg['segGroups']:
        obj = {}
        obj['id'] = group['objectId']  # Use objectId as the primary ID
        obj['category'] = group['label']  # Category label (e.g., 'chair', 'table')
        
        # Get mask for vertices belonging to this object's segments
        seg_list = group['segments']
        mask = np.isin(seg_indices, seg_list)
        
        if np.sum(mask) == 0:
            continue  # Skip if no vertices found
        
        obj_verts = vertices[mask]
        
        # Compute position (centroid)
        center = np.mean(obj_verts[:, :3], axis=0).tolist()
        obj['position'] = {'x': center[0], 'y': center[1], 'z': center[2]}
        
        # Compute size (axis-aligned bounding box dimensions)
        min_pt = np.min(obj_verts[:, :3], axis=0)
        max_pt = np.max(obj_verts[:, :3], axis=0)
        bbox_size = (max_pt - min_pt).tolist()
        obj['size'] = {'length': bbox_size[0], 'width': bbox_size[1], 'height': bbox_size[2]}  # Arbitrary axis naming; adjust as needed
        
        # Compute average color
        avg_color = np.mean(obj_verts[:, 3:], axis=0).tolist()
        obj['color'] = {'r': avg_color[0], 'g': avg_color[1], 'b': avg_color[2]}
        
        scene_graph.append(obj)
    
    return scene_graph

# Example usage:
# Replace with your actual paths
# scan_path = '/path/to/your/scannet/scans/scene0000_00'  # Directory containing the files for a specific scan
# scan_id = 'scene0000_00'
# scene_graph = build_scene_graph(scan_path, scan_id)
# with open('scene_graph.json', 'w') as f:
#     json.dump(scene_graph, f, indent=4)
# print("Scene graph saved to scene_graph.json")

if __name__ == "__main__":
    scan_path = '../TSP3D-main/scannet/scans/scene0000_00'  # Directory containing the files for a specific scan
    scan_id = 'scene0000_00'
    scene_graph = build_scene_graph(scan_path, scan_id)
    with open('scene_graph.json', 'w') as f:
        json.dump(scene_graph, f, indent=4)
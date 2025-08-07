import argparse
from src.joint_det_dataset import save_data  # 导入 TSP3D 的 save_data

parser = argparse.ArgumentParser()
parser.add_argument('--scannet_data', required=True)  # ScanNet 数据路径 (scans/)
parser.add_argument('--data_root', required=True)     # 输出 pkl 路径
parser.add_argument('--vigil_txt_dir', default='./ViGiL3D/', help='ViGiL3D txt 文件目录')
args, _ = parser.parse_known_args()

splits = ['val']  # 根据需要生成 val/train

for sp in splits:
    print(f'Start packing the {sp} set for ViGiL3D...')
    
    # 加载 ViGiL3D 的 scan_ids（替换原 meta_data/scannetv2_{sp}.txt）
    vigil_txt = f'{args.vigil_txt_dir}/ViGiL3D_filtered_{sp}.txt'
    with open(vigil_txt) as f:
        scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
    
    # 调用 save_data，但传入自定义 scan_ids 和 data_path
    # 注意：save_data 原函数基于 split 读 txt，但您需修改 save_data 或重写
    # 这里假设您修改 joint_det_dataset.py 的 save_data 为接受 scan_ids 参数
    # 如果未改，用以下方式模拟
    save_data(f'{args.data_root}/{sp}_vigil3d_v3scans.pkl', sp, args.scannet_data, custom_scan_ids=scan_ids)  # 如果 save_data 支持 custom_scan_ids

    print(f'The {sp} set for ViGiL3D is packed!')
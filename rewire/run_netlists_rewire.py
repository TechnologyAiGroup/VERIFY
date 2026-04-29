import os
# Fix the libgomp warning by setting a valid OMP_NUM_THREADS value
os.environ["OMP_NUM_THREADS"] = "1"

import time
import torch
import numpy as np
import shutil
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from preprocessing import rewire
import argparse


def load_circuit_data(circuit_type, circuit_id, netlists_dir="netlists"):
    """
    加载单个电路数据
    
    Parameters:
    -----------
    circuit_type : str
        电路类型，如 'b11', 'c1355' 等
    circuit_id : int
        电路编号，如 0, 1, 2 等
    netlists_dir : str
        netlists目录路径
    
    Returns:
    --------
    Data : PyTorch Geometric Data对象
    """
    circuit_dir = os.path.join(netlists_dir, circuit_type, str(circuit_id))
    
    # 加载边列表 (edges.txt)
    edges_file = os.path.join(circuit_dir, "edges.txt")
    with open(edges_file, 'r') as f:
        lines = f.readlines()
    
    # 解析边 - 第一行是源节点列表，第二行是目标节点列表
    source_nodes = list(map(int, lines[0].strip().split()))
    target_nodes = list(map(int, lines[1].strip().split()))
    
    # 创建edge_index (2, num_edges)
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    # 加载特征 (feat.txt)
    feat_file = os.path.join(circuit_dir, "feat.txt")
    features = []
    with open(feat_file, 'r') as f:
        for line in f:
            feat = list(map(float, line.strip().split()))
            features.append(feat)
    x = torch.tensor(features, dtype=torch.float)
    
    # 加载标签 (label.txt)
    label_file = os.path.join(circuit_dir, "label.txt")
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            label = int(line.strip())
            labels.append(label)
    y = torch.tensor(labels, dtype=torch.long)
    
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # 转换为无向图
    data.edge_index = to_undirected(data.edge_index)
    
    return data


def get_available_circuit_ids(circuit_type, netlists_dir="netlists"):
    """
    获取指定电路类型的所有可用电路编号
    
    Parameters:
    -----------
    circuit_type : str
        电路类型
    netlists_dir : str
        netlists目录路径
    
    Returns:
    --------
    list : 可用的电路编号列表
    """
    circuit_type_dir = os.path.join(netlists_dir, circuit_type)
    if not os.path.exists(circuit_type_dir):
        return []
    
    circuit_ids = []
    for item in os.listdir(circuit_type_dir):
        item_path = os.path.join(circuit_type_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            circuit_ids.append(int(item))
    
    return sorted(circuit_ids)


def save_rewired_edges(edge_index, save_path):
    """
    保存重布线后的边数据（2*N格式，无注释）
    
    Parameters:
    -----------
    edge_index : torch.Tensor
        边索引张量，形状为 (2, num_edges)
    save_path : str
        保存文件路径
    """
    edges = edge_index.cpu().numpy()
    
    with open(save_path, 'w') as f:
        # 第一行：所有源节点
        source_nodes = edges[0]
        f.write(' '.join(map(str, source_nodes)) + '\n')
        
        # 第二行：所有目标节点
        target_nodes = edges[1]
        f.write(' '.join(map(str, target_nodes)) + '\n')


def log_to_file(message, filename="results/netlists_rewire.txt"):
    print(message)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as file:
        file.write(message + "\n")


def main():
    parser = argparse.ArgumentParser(description='Rewire circuit netlists')
    parser.add_argument('--circuit_type', type=str, required=True, 
                        help='Circuit type (e.g., b11, c1355, etc.)')
    parser.add_argument('--num_circuits', type=int, required=True,
                        help='Number of circuits to process (starting from 0)')
    parser.add_argument('--num_iterations', type=int, required=True,
                        help='Number of rewiring iterations')
    parser.add_argument('--borf_batch_add', type=int, required=True,
                        help='BORF batch add parameter')
    parser.add_argument('--borf_batch_remove', type=int, required=True,
                        help='BORF batch remove parameter')
    parser.add_argument('--netlists_dir', type=str, default='netlists',
                        help='Path to netlists directory')
    parser.add_argument('--output_dir', type=str, default='graphData/rewired',
                        help='Output directory for rewired graphs')
    
    args = parser.parse_args()
    
    # 获取所有可用的电路编号
    available_ids = get_available_circuit_ids(args.circuit_type, args.netlists_dir)
    
    if not available_ids:
        raise ValueError(f"No circuits found for type '{args.circuit_type}' in {args.netlists_dir}")
    
    # 选择要处理的电路
    circuits_to_process = available_ids[:args.num_circuits]
    
    if len(circuits_to_process) < args.num_circuits:
        print(f"[WARNING] Only {len(circuits_to_process)} circuits available, "
              f"but {args.num_circuits} requested.")
    
    log_to_file(f"=" * 60)
    log_to_file(f"[INFO] Circuit Type: {args.circuit_type}")
    log_to_file(f"[INFO] Number of circuits to process: {len(circuits_to_process)}")
    log_to_file(f"[INFO] Circuit IDs: {circuits_to_process}")
    log_to_file(f"[INFO] BORF Parameters:")
    log_to_file(f"       num_iterations = {args.num_iterations}")
    log_to_file(f"       batch_add = {args.borf_batch_add}")
    log_to_file(f"       batch_remove = {args.borf_batch_remove}")
    log_to_file(f"=" * 60)
    
    # 处理每个电路
    for circuit_id in circuits_to_process:
        log_to_file(f"\n[INFO] Processing circuit {args.circuit_type}/{circuit_id}")
        
        # 加载电路数据
        try:
            data = load_circuit_data(args.circuit_type, circuit_id, args.netlists_dir)
            log_to_file(f"       Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}, Features: {data.x.shape[1]}")
        except Exception as e:
            log_to_file(f"[ERROR] Failed to load circuit {circuit_id}: {e}")
            continue
        
        # 执行重布线
        start = time.time()
        try:
            new_edge_index, edge_type = rewire.rewireProcess(
                data,
                loops=args.num_iterations,
                remove_edges=False,
                is_undirected=True,
                batch_add=args.borf_batch_add,
                batch_remove=args.borf_batch_remove,
                dataset_name=f"{args.circuit_type}_{circuit_id}",
                graph_index=0
            )
            data.edge_index = new_edge_index
            data.edge_type = edge_type
            
            rewiring_duration = time.time() - start
            log_to_file(f"       Rewiring time: {rewiring_duration:.2f}s")
            log_to_file(f"       Rewired edges: {data.edge_index.shape[1]}")
            
            # 准备保存目录
            output_circuit_dir = os.path.join(args.output_dir, args.circuit_type, str(circuit_id))
            os.makedirs(output_circuit_dir, exist_ok=True)
            
            # 原始数据目录
            source_dir = os.path.join(args.netlists_dir, args.circuit_type, str(circuit_id))
            
            # === 保存重布线后的数据 ===
            # 1. edges.txt - 保存重布线后的边（2*N格式）
            save_rewired_edges(data.edge_index, os.path.join(output_circuit_dir, "edges.txt"))
            
            # 2. feat.txt - 直接复制（特征不变）
            shutil.copy(
                os.path.join(source_dir, "feat.txt"),
                os.path.join(output_circuit_dir, "feat.txt")
            )
            
            # 3. label.txt - 直接复制（标签不变）
            shutil.copy(
                os.path.join(source_dir, "label.txt"),
                os.path.join(output_circuit_dir, "label.txt")
            )
            
            # 4. keys.txt - 如果存在则复制（关键节点不变）
            keys_src = os.path.join(source_dir, "keys.txt")
            if os.path.exists(keys_src):
                shutil.copy(keys_src, os.path.join(output_circuit_dir, "keys.txt"))
            
            log_to_file(f"       Saved to: {output_circuit_dir}")
            
        except Exception as e:
            log_to_file(f"[ERROR] Failed to rewire circuit {circuit_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log_to_file(f"\n[INFO] Finished processing {len(circuits_to_process)} circuits")
    log_to_file("=" * 60)


if __name__ == "__main__":
    main()

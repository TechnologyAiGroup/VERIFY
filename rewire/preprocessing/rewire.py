import os
import ot
import time
import torch
import pathlib
import numpy as np
import pandas as pd
import multiprocessing as mp
import networkx as nx
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
from torch_geometric.datasets import TUDataset
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.sparse.linalg import eigsh

class CurvaturePlainGraph():
    def __init__(self, G, device=None):
        self.G = G
        self.V = len(G.nodes)
        self.E = list(G.edges)
        self.adjacency_matrix = np.full((self.V,self.V),np.inf)
        self.dist = self.adjacency_matrix.copy()

        if(device is None):
          self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
          self.device = device
        
        for index in range(self.V):
            self.adjacency_matrix[index, index] = 0
        for index, edge in enumerate(self.E):
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1
        
        # Floyd Warshall
        self.dist = self._floyd_warshall()

    def __str__(self):
        return f'The graph contains {self.V} nodes and {len(self.E)} edges {self.E}. '

    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.E)
        nx.draw_networkx(G)
        plt.show()

    def _dijkstra(self):
        for i in range(self.V):
            for j in range(self.V):
                try:
                    self.dist[i][j] = len(nx.dijkstra_path(self.G, i, j))
                except nx.NetworkXNoPath:
                    continue
        return self.dist

    def _floyd_warshall(self):
        self.dist = self.adjacency_matrix.copy()
        for k in range(self.V):
            for i in range(self.V):
                for j in range(self.V):
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][k] + self.dist[k][j])
        return self.dist

    def _to_tensor(self, x):
        x = torch.Tensor(x).to(self.device)
        return x

    def _to_numpy(self, x):
        if(torch.cuda.is_available()):
            return x.cpu().detach().numpy()
        return x.detach().numpy()

    def _transport_plan_uv(self, u, v, method = 'OTD', u_neighbors=None, v_neighbors=None):
        u_neighbors = [p for p in range(self.V) if self.adjacency_matrix[u][p] == 1] if u_neighbors is None else u_neighbors
        v_neighbors = [q for q in range(self.V) if self.adjacency_matrix[v][q] == 1] if v_neighbors is None else v_neighbors
        u_deg = len(u_neighbors)
        v_deg = len(v_neighbors)

        # Instead of using fractions [1/n,...,1/n], [1/m,...,1/m], we use [m,...,m], [n,...,n] and then divides by mn
        mu = self._to_tensor(np.full(u_deg, v_deg))
        mv = self._to_tensor(np.full(v_deg, u_deg))
        sub_indices = np.ix_(u_neighbors, v_neighbors)
        dist_matrix = self._to_tensor(self.dist[sub_indices])
        dist_matrix[dist_matrix == np.inf] = 0 # Correct the dist matrix
        
        # Update distance matrix
        self.d = dist_matrix
        if method == 'OTD':
            optimal_plan = self._to_numpy(ot.emd(mu, mv, dist_matrix))
        else:
            raise NotImplemented
        optimal_plan = optimal_plan/(u_deg*v_deg) # PI
        optimal_cost = optimal_plan*self._to_numpy(dist_matrix)
        optimal_total_cost = np.sum(optimal_cost)
        optimal_cost = pd.DataFrame(optimal_cost, columns=v_neighbors, index=u_neighbors)
        return optimal_total_cost, optimal_cost

    def add_edge(self, p, q, inter_up, inter_vq):
        # TODO : Need to replace with a more efficient algorithm
        self.adjacency_matrix[p, q] = 1
        self.adjacency_matrix[q, p] = 1
        
        # self.dist = self._floyd_warshall()
        self.dist[p, q] = 1
        self.dist[q, p] = 1

        # Add edge to edge list
        self.E.append((p, q))

        for k in inter_up:
            self.dist[k, q] = min(2, self.dist[k, q])
            
        for l in inter_vq:
            self.dist[l, p] = min(2, self.dist[l, p])

    def remove_edge(self, i, j):
        self.adjacency_matrix[i, j] = 0
        self.adjacency_matrix[j, i] = 0
        # self.dist = self._floyd_warshall()
        self.dist[i, j] = np.inf
        self.dist[j, i] = np.inf

        self.E.remove((i, j))

    def curvature_uv(self, u, v, method = 'OTD', u_neighbors=None, v_neighbors=None):
        optimal_cost, optimal_plan = self._transport_plan_uv(u, v, method, u_neighbors=u_neighbors, v_neighbors=v_neighbors)
        return 1 - optimal_cost/self.dist[u,v], optimal_plan

    def edge_curvatures(self, method = 'OTD', return_transport_cost=False):
        edge_curvature_dict = {}
        transport_plan_dict = {}
        for edge in self.E:
            edge_curvature_dict[edge], transport_plan_dict[edge] = self.curvature_uv(edge[0], edge[1], method)

        if(return_transport_cost):
            return edge_curvature_dict, transport_plan_dict
        return edge_curvature_dict

    def all_curvatures(self, method = 'OTD'):
        C = np.zeros((self.V, self.V))
        for u in range(self.V):
            for v in range(u+1, self.V):
                C[u,v] = self.curvature_uv(u,v,method)
        C = C + np.transpose(C) + np.eye(self.V)
        C = np.hstack((np.reshape([str(u) for u in np.arange(self.V)],(self.V,1)), C))
        head = ['C'] + [str(u) for u in range(self.V)]
        print(tabulate(C, floatfmt=".2f", headers=head, tablefmt="presto"))

def _softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()

def _preprocess_data(data, is_undirected=False):
    # Get necessary data information
    N = data.x.shape[0]
    m = data.edge_index.shape[1]

    # Compute the adjacency matrix
    if not "edge_type" in data.keys:
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type

    # Convert graph to Networkx
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()

    return G, N, edge_type

def _get_neighbors(x, G, is_undirected=False, is_source=False):
    if is_undirected:
        x_neighbors = list(G.neighbors(x)) #+ [x]
    else:
        if(is_source):
          x_neighbors = list(G.successors(x)) #+ [x]
        else:
          x_neighbors = list(G.predecessors(x)) #+ [x]
    return x_neighbors

def _get_rewire_candidates(G, x_neighbors, y_neighbors):
    candidates = []
    for i in x_neighbors:
        for j in y_neighbors:
            if (i != j) and (not G.has_edge(i, j)):
                candidates.append((i, j))
    return candidates

def _calculate_improvement(graph, C, x, y, x_neighbors, y_neighbors, k, l):
    """
    Calculate the curvature performance of x -> y when k -> l is added.
    """
    graph.add_edge(k, l)
    old_curvature = C[(x, y)]

    new_curvature, _ = graph.curvature_uv(x, y, u_neighbors=x_neighbors, v_neighbors=y_neighbors)
    improvement = new_curvature - old_curvature
    graph.remove_edge(k, l)

    return new_curvature, old_curvature


def _compute_fiedler_vector(G):
    """Compute Fiedler vector (second smallest eigenvector of normalized Laplacian)"""
    # Ensure the graph is undirected for Laplacian computation
    if G.is_directed():
        G = G.to_undirected()
    
    # Handle edge case: empty graph or single node
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return np.array([])
    if n_nodes == 1:
        return np.array([0.0])
    
    try:
        L = nx.normalized_laplacian_matrix(G)
        # Compute two smallest eigenvalues/vectors
        eigvals, eigvecs = eigsh(L, k=2, which='SM', tol=1e-3, maxiter=1000)
        fiedler = eigvecs[:, 1]  # second smallest eigenvector
    except Exception as e:
        print(f"[WARNING] Failed to compute Fiedler vector: {e}. Using random.")
        fiedler = np.random.randn(n_nodes)
    return fiedler


def rewireProcess(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    batch_add=4,
    batch_remove=2,
    device=None,
    save_dir='rewired_graphs',
    dataset_name=None,
    graph_index=0,
    debug=False,
    target_change_ratio=0.2
):
    # Preprocess data
    G, N, edge_type = _preprocess_data(data)
    original_num_edges = len(G.edges)
    
    if debug:
        print(f"[INFO] Original graph has {original_num_edges} edges")
    
    # === 关键修改：禁用自动缩放，使用固定参数 ===
    # （你已手动覆盖，保留即可）
    loops = 1
    batch_add = 50
    batch_remove = 10
    target_changes = int(original_num_edges * target_change_ratio)
    
    print(f"[INFO] Parameters: loops={loops}, batch_add={batch_add}, batch_remove={batch_remove}")
    print(f"[INFO] Target changes: {target_changes} edges ({target_change_ratio*100:.1f}%)")
    
    # 缓存检查（略，保持不变）

    dirname = f'{save_dir}/{dataset_name}'
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    edge_index_filename = os.path.join(
        dirname, 
        f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_change_{target_change_ratio}_edge_index_{graph_index}.pt'
    )
    edge_type_filename = os.path.join(
        dirname, 
        f'iters_{loops}_add_{batch_add}_remove_{batch_remove}_change_{target_change_ratio}_edge_type_{graph_index}.pt'
    )

    if os.path.exists(edge_index_filename) and os.path.exists(edge_type_filename):
        if debug: 
            print(f'[INFO] Loading cached rewired graph...')
        with open(edge_index_filename, 'rb') as f:
            edge_index = torch.load(f)
        with open(edge_type_filename, 'rb') as f:
            edge_type = torch.load(f)
        return edge_index, edge_type

    total_added = 0
    total_removed = 0
    
    for iteration in range(loops):
        remaining_changes = target_changes - (total_added + total_removed)
        if remaining_changes <= 0:
            break
        
        remaining_loops = loops - iteration
        current_batch_add = min(batch_add, max(0, int(remaining_changes * 0.5 / remaining_loops)))
        current_batch_remove = min(batch_remove, max(0, int(remaining_changes * 0.5 / remaining_loops)))
        if current_batch_add == 0 and current_batch_remove == 0:
            break
        
        # Step 1: Compute ORC
        orc = OllivierRicci(G, alpha=0)
        orc.compute_ricci_curvature()
        
        # Step 2: Compute Fiedler vector
        fiedler = _compute_fiedler_vector(G)  # shape: [N]
        
        # Step 3: Build list of edges with (curvature, fiedler_dist, score_for_add, score_for_remove)
        # === 优化3：向量化边信息计算 ===
        # 获取所有边并转换为 NumPy 数组
        edges = list(orc.G.edges())
        u_nodes = np.array([e[0] for e in edges])
        v_nodes = np.array([e[1] for e in edges])
        
        # 提取曲率（这一步仍有循环，但后续操作全部向量化）
        curvatures = np.array([
            orc.G[u][v]['ricciCurvature']['rc_curvature'] 
            for u, v in edges
        ])
        
        # 批量计算 Fiedler 距离（向量化）
        fiedler_array = np.array(fiedler)
        fiedler_diff = np.abs(fiedler_array[u_nodes] - fiedler_array[v_nodes])
        
        # 批量计算分数（向量化）
        d_uv_safe = fiedler_diff + 1e-8
        score_add = curvatures * fiedler_diff
        score_remove = curvatures / d_uv_safe
        
        # 使用 NumPy 排序（比 Python sorted 快）
        sorted_add_idx = np.argsort(score_add)  # 升序：最负的在前
        sorted_remove_idx = np.argsort(score_remove)[::-1]  # 降序：最大的在前
        
        # 提取 top-K 边
        most_neg_edges = [(u_nodes[i], v_nodes[i]) for i in sorted_add_idx[:current_batch_add]]
        most_pos_edges = [(u_nodes[i], v_nodes[i]) for i in sorted_remove_idx[:current_batch_remove]]

        # Add edges (same as before)
        added_this_iter = 0
        for (u, v) in most_neg_edges:
            pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            if p != q and not G.has_edge(p, q):
                G.add_edge(p, q)
                added_this_iter += 1
                total_added += 1
                if total_added + total_removed >= target_changes:
                    break
        
        # Remove edges
        removed_this_iter = 0
        for (u, v) in most_pos_edges:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                removed_this_iter += 1
                total_removed += 1
                if total_added + total_removed >= target_changes:
                    break
        
        if debug and (added_this_iter > 0 or removed_this_iter > 0):
            print(f"[INFO] Iter {iteration+1}: Added {added_this_iter}, Removed {removed_this_iter}, "
                  f"Total: +{total_added}, -{total_removed}, Current edges: {len(G.edges)}")
        
        if total_added + total_removed >= target_changes:
            break

    # Save (unchanged)
    edge_index = from_networkx(G).edge_index
    edge_type = torch.zeros(size=(len(G.edges),)).type(torch.LongTensor)

    with open(edge_index_filename, 'wb') as f:
        torch.save(edge_index, f)
    with open(edge_type_filename, 'wb') as f:
        torch.save(edge_type, f)

    if debug:
        final_num_edges = len(G.edges)
        actual_change_ratio = abs(final_num_edges - original_num_edges) / original_num_edges
        print(f"[INFO] Final: Original {original_num_edges} -> Final {final_num_edges} "
              f"(change: {actual_change_ratio*100:.2f}%)")

    return edge_index, edge_type
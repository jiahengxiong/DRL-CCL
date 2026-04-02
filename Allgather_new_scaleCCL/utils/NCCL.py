import networkx as nx
from collections import defaultdict

def nccl_topo(topology):
    nccl_topology = topology.copy()
    
    def get_edge_cost(G, u, v):
        d = G.edges[u, v]
        return float(d.get('propagation_latency', 0)) + float(d.get('transmission_latency', 0))
    
    dc_gpus = defaultdict(list)
    for node, data in topology.nodes(data=True):
        if data.get('type') == 'GPU':
            dc = data['DC']
            dc_gpus[dc].append(node)
            
    dc_keys = sorted(list(dc_gpus.keys()))
    
    def find_chain(G, gpus):
        start = min(gpus)
        end = max(gpus)
        best_path = None
        min_cost = float('inf')
        
        def dfs(curr, path, prop_sum, max_trans):
            nonlocal best_path, min_cost
            cost = prop_sum + max_trans
            if cost >= min_cost:
                return
            if len(path) == len(gpus):
                if curr == end:
                    best_path = list(path)
                    min_cost = cost
                return
            
            for nxt in G.successors(curr):
                if nxt in gpus and nxt not in path:
                    d = G.edges[curr, nxt]
                    p_lat = float(d.get('propagation_latency', 0))
                    t_lat = float(d.get('transmission_latency', 0))
                    dfs(nxt, path + [nxt], prop_sum + p_lat, max(max_trans, t_lat))
                
        dfs(start, [start], 0.0, 0.0)
        return best_path

    def get_cross_dc_path(G, src, dst, exclude_switches=None):
        if exclude_switches is None:
            exclude_switches = set()
            
        allowed_nodes = {n for n, d in G.nodes(data=True) if d.get('type') == 'switch' and n not in exclude_switches}
        allowed_nodes.add(src)
        allowed_nodes.add(dst)
        
        subG = G.subgraph(allowed_nodes)
        
        best_path = None
        min_cost = float('inf')
        
        # 使用全路径搜索来计算非累加成本：最大传输延迟 + 累加传播延迟
        for path in nx.all_simple_paths(subG, source=src, target=dst):
            prop_sum = 0.0
            max_trans = 0.0
            for i in range(len(path) - 1):
                d = subG.edges[path[i], path[i+1]]
                prop_sum += float(d.get('propagation_latency', 0))
                max_trans = max(max_trans, float(d.get('transmission_latency', 0)))
            
            cost = prop_sum + max_trans
            if cost < min_cost:
                min_cost = cost
                best_path = path
                
        if best_path is None:
            raise nx.NetworkXNoPath(f"No path between {src} and {dst}")
            
        return best_path

    # 1. 在每个 DC 找到只包含 GPU 的 chain
    chains = []
    for dc in dc_keys:
        gpus = dc_gpus[dc]
        chains.append(find_chain(topology, gpus))
        
    # 2. 找 2 条 min cost 路径把这俩 chain 连起来（保证 Switch 不复用，构成真正的 Ring）
    keep_edges = set()
    def mark_path(path):
        for i in range(len(path) - 1):
            keep_edges.add((path[i], path[i+1]))

    # 标记两条 chain 内部的边，并将其首尾闭合成 Ring
    for chain in chains:
        mark_path(chain)
        # 将链尾连回链头，构成局部单向 Ring
        if topology.has_edge(chain[-1], chain[0]):
            keep_edges.add((chain[-1], chain[0]))
        
    # 找第一条跨 DC 路径
    src1 = chains[0][-1]
    dst1 = chains[1][0]
    path1 = get_cross_dc_path(topology, src1, dst1)
    mark_path(path1)
    
    # 找第二条跨 DC 路径，不需要剔除第一条路径用过的 Switch
    src2 = chains[1][-1]
    dst2 = chains[0][0]
    
    path2 = get_cross_dc_path(topology, src2, dst2)
        
    mark_path(path2)

    # 3. 移除多余边
    edges_to_remove = [e for e in nccl_topology.edges() if e not in keep_edges]
    nccl_topology.remove_edges_from(edges_to_remove)

    # 4. 自适应清理节点身上的旧路由表 (job, added_job 等)
    for node in nccl_topology.nodes():
        node_data = nccl_topology.nodes[node]
        if 'job' in node_data:
            # 只保留存在于 keep_edges 中的 (src, dst) 任务
            node_data['job'] = {
                k: v for k, v in node_data['job'].items() if k in keep_edges
            }
        if 'added_job' in node_data:
            node_data['added_job'] = {
                k: v for k, v in node_data['added_job'].items() if k in keep_edges
            }

    # print(nccl_topology.edges())

    return nccl_topology

def fast_nccl_topo(topology, nccl_topology):
    # 拷贝一个基于 topology 的完整图，我们在这个图上进行剪裁
    fast_nccl_topology = topology.copy()
    
    # 记录需要保留的边
    keep_edges = set()
    
    # 1. 保持 nccl_topology 的数据中心内部拓扑不变
    # 获取 nccl_topology 中的所有边，筛选出属于数据中心内部的边（即不涉及 WAN link）
    for u, v in nccl_topology.edges():
        u_type = nccl_topology.nodes[u].get('type')
        v_type = nccl_topology.nodes[v].get('type')
        u_dc = nccl_topology.nodes[u].get('DC')
        v_dc = nccl_topology.nodes[v].get('DC')
        
        # 只要不是跨 DC 的 switch-switch 连线，都认为是 DC 内部拓扑的一部分
        if not (u_type == 'switch' and v_type == 'switch'):
            keep_edges.add((u, v))
            
    # 2. 将 topology 里面的所有 border_router 之间的链路都加进 keep_edges
    for u, v, data in topology.edges(data=True):
        u_type = topology.nodes[u].get('type')
        v_type = topology.nodes[v].get('type')
        if u_type == 'switch':
            keep_edges.add((u, v))
        # if u_type == 'switch' and v_type == 'switch':
        #     keep_edges.add((u, v))
            
    # 3. 移除多余边，生成中间拓扑
    edges_to_remove = [e for e in fast_nccl_topology.edges() if e not in keep_edges]
    fast_nccl_topology.remove_edges_from(edges_to_remove)
    
    # # 4. 清理中间拓扑节点的旧路由表，为 nccl_topo 调用做准备
    # for node in fast_nccl_topology.nodes():
    #     node_data = fast_nccl_topology.nodes[node]
    #     if 'job' in node_data:
    #         node_data['job'] = {
    #             k: v for k, v in node_data['job'].items() if k in keep_edges
    #         }
    #     if 'added_job' in node_data:
    #         node_data['added_job'] = {
    #             k: v for k, v in node_data['added_job'].items() if k in keep_edges
    #         }
            
    # 5. 调用 nccl_topo 跑一下，它会在这个中间拓扑上重新计算最优的跨 DC 路径
    return nccl_topo(fast_nccl_topology)

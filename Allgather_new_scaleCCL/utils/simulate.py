import simpy
import networkx as nx
from typing import Dict, Tuple, Any, Optional
from itertools import islice
import numpy as np
import config
from Allgather_new_scaleCCL.CCL_Simulator_copy.simcore import Sim, PolicyEntry
from Allgather_new_scaleCCL.utils.util import load_topology


_WAN_NUM_NODES = 40
_WAN_EDGE_LIST = [
    (0, 3),
    (0, 5),
    (0, 7),
    (0, 8),
    (0, 9),
    (1, 31),
    (1, 33),
    (2, 11),
    (2, 15),
    (2, 21),
    (2, 22),
    (3, 34),
    (4, 6),
    (4, 36),
    (4, 37),
    (5, 8),
    (5, 26),
    (5, 30),
    (5, 35),
    (5, 36),
    (6, 10),
    (6, 26),
    (6, 28),
    (7, 21),
    (7, 30),
    (9, 19),
    (9, 39),
    (10, 12),
    (10, 29),
    (11, 13),
    (11, 16),
    (12, 14),
    (12, 15),
    (12, 25),
    (13, 39),
    (14, 22),
    (15, 22),
    (16, 18),
    (16, 27),
    (17, 25),
    (17, 26),
    (18, 22),
    (18, 27),
    (19, 24),
    (20, 21),
    (20, 25),
    (20, 35),
    (21, 30),
    (23, 24),
    (23, 34),
    (25, 12),
    (25, 20),
    (26, 35),
    (26, 36),
    (28, 32),
    (28, 38),
    (29, 31),
    (31, 32),
    (32, 33),
    (37, 38),
]
_BR_NODES = (0, 1, 2, 3)
_LOGICAL_WAN_LINKS = ((0, 2), (0, 3), (1, 2), (1, 3))

_FIXED_WAN_ROUTES: Optional[Dict[Tuple[int, int], list]] = None
_FIXED_BR_ATTACHES: Optional[Dict[int, Tuple[str, str]]] = None
_CACHED_SIM_TOPO: Optional[nx.DiGraph] = None
_CACHED_BASE_NODESET: Optional[frozenset] = None


def _wan_node(i: int) -> str:
    return f"wan_{int(i)}"


def _edge_key(a: Any, b: Any) -> Tuple[str, str]:
    sa, sb = str(a), str(b)
    return (sa, sb) if sa <= sb else (sb, sa)


def _find_edge_disjoint_routes() -> Dict[Tuple[int, int], list]:
    wan = nx.Graph()
    wan.add_nodes_from(range(_WAN_NUM_NODES))
    for a, b in _WAN_EDGE_LIST:
        if int(a) == int(b):
            continue
        wan.add_edge(int(a), int(b))

    deg = sorted(wan.degree, key=lambda x: x[1], reverse=True)
    cand_nodes = [int(n) for n, d in deg if d > 0]
    cand_nodes = cand_nodes[:40]

    def k_shortest_paths(s: int, t: int, k: int, cutoff: int):
        try:
            gen = nx.shortest_simple_paths(wan, s, t)
        except nx.NetworkXNoPath:
            return []
        out = []
        for p in islice(gen, k):
            if len(p) - 1 > cutoff:
                continue
            out.append(p)
        return out

    attach_used: Dict[int, set] = {br: set() for br in _BR_NODES}
    used_edges: set = set()
    chosen: Dict[Tuple[int, int], list] = {}

    order = _LOGICAL_WAN_LINKS

    def backtrack(i: int) -> bool:
        if i >= len(order):
            return True
        u, v = order[i]
        src_used = attach_used[u]
        dst_used = attach_used[v]

        if len(src_used) == 0:
            src_opts = cand_nodes
        elif len(src_used) == 1:
            (only,) = tuple(src_used)
            src_opts = [n for n in cand_nodes if n != only]
        else:
            src_opts = list(src_used)

        if len(dst_used) == 0:
            dst_opts = cand_nodes
        elif len(dst_used) == 1:
            (only,) = tuple(dst_used)
            dst_opts = [n for n in cand_nodes if n != only]
        else:
            dst_opts = list(dst_used)

        for s in src_opts:
            for t in dst_opts:
                if int(s) == int(t):
                    continue
                paths = k_shortest_paths(int(s), int(t), k=40, cutoff=20)
                for p in paths:
                    route = [u] + [_wan_node(x) for x in p] + [v]
                    edges = {_edge_key(a, b) for a, b in zip(route, route[1:])}
                    if edges & used_edges:
                        continue
                    added_src = None
                    added_dst = None
                    if len(attach_used[u]) < 2 and int(s) not in attach_used[u]:
                        attach_used[u].add(int(s))
                        added_src = int(s)
                    if len(attach_used[v]) < 2 and int(t) not in attach_used[v]:
                        attach_used[v].add(int(t))
                        added_dst = int(t)
                    if len(attach_used[u]) > 2 or len(attach_used[v]) > 2:
                        if added_src is not None:
                            attach_used[u].remove(added_src)
                        if added_dst is not None:
                            attach_used[v].remove(added_dst)
                        continue
                    chosen[(u, v)] = route
                    used_edges.update(edges)
                    if backtrack(i + 1):
                        return True
                    used_edges.difference_update(edges)
                    del chosen[(u, v)]
                    if added_src is not None:
                        attach_used[u].remove(added_src)
                    if added_dst is not None:
                        attach_used[v].remove(added_dst)
        return False

    ok = backtrack(0)
    if not ok:
        raise RuntimeError("No edge-disjoint WAN routes found for logical links")

    routes: Dict[Tuple[int, int], list] = {}
    for (u, v), r in chosen.items():
        routes[(u, v)] = r
        routes[(v, u)] = list(reversed(r))
    return routes


def _get_fixed_wan_routes_and_attaches() -> Tuple[Dict[Tuple[int, int], list], Dict[int, Tuple[str, str]]]:
    global _FIXED_WAN_ROUTES, _FIXED_BR_ATTACHES
    if isinstance(_FIXED_WAN_ROUTES, dict) and _FIXED_WAN_ROUTES and isinstance(_FIXED_BR_ATTACHES, dict) and _FIXED_BR_ATTACHES:
        return _FIXED_WAN_ROUTES, _FIXED_BR_ATTACHES

    routes = _find_edge_disjoint_routes()
    br_attaches: Dict[int, set] = {br: set() for br in _BR_NODES}
    for u, v in _LOGICAL_WAN_LINKS:
        r = routes.get((u, v))
        if not isinstance(r, list) or len(r) < 3:
            raise RuntimeError(f"Invalid WAN route for ({u},{v}): {r}")
        br_attaches[u].add(r[1])
        br_attaches[v].add(r[-2])

    attaches_out: Dict[int, Tuple[str, str]] = {}
    for br, ws in br_attaches.items():
        if len(ws) != 2:
            raise RuntimeError(f"BR {br} must attach to 2 WAN nodes, got {sorted(list(ws))}")
        a, b = sorted(list(ws), key=str)
        attaches_out[int(br)] = (str(a), str(b))

    _FIXED_WAN_ROUTES = routes
    _FIXED_BR_ATTACHES = attaches_out
    return routes, attaches_out


def _build_sim_topology(base: nx.DiGraph, true_stream: Dict[Tuple[Any, Any], np.ndarray], rate_scale: float = 1.0) -> nx.DiGraph:
    global _CACHED_SIM_TOPO, _CACHED_BASE_NODESET
    routes, br_attaches = _get_fixed_wan_routes_and_attaches()

    base_nodeset = frozenset(base.nodes)
    if _CACHED_SIM_TOPO is None or _CACHED_BASE_NODESET != base_nodeset:
        H = nx.DiGraph()
        for n, attrs in base.nodes(data=True):
            t = attrs.get("type")
            nt = "gpu" if t == "GPU" else "switch"
            H.add_node(
                n,
                type=nt,
                num_qps=int(attrs.get("num_qps", 1)),
                quantum_packets=int(attrs.get("quantum_packets", 1)),
                tx_proc_delay=float(attrs.get("tx_proc_delay", 0.0)),
                sw_proc_delay=float(attrs.get("sw_proc_delay", 0.0)),
                gpu_store_delay=float(attrs.get("gpu_store_delay", 0.0)),
            )

        for u, v, e in base.edges(data=True):
            if (
                u in _BR_NODES
                and v in _BR_NODES
                and base.nodes[u].get("type") == "switch"
                and base.nodes[v].get("type") == "switch"
                and base.nodes[u].get("DC") != base.nodes[v].get("DC")
            ):
                continue
            cap = e.get("link_capcapacity")
            r = float(cap) * float(rate_scale)
            pd = float(e.get("propagation_latency"))
            if r <= 0:
                r = 1e-9
            H.add_edge(u, v, link_rate_bps=r, prop_delay=pd)

        for i in range(_WAN_NUM_NODES):
            H.add_node(_wan_node(i), type="switch", num_qps=1, quantum_packets=1, tx_proc_delay=0.0, sw_proc_delay=0.0, gpu_store_delay=0.0)

        for a, b in _WAN_EDGE_LIST:
            ua, vb = _wan_node(int(a)), _wan_node(int(b))
            if ua == vb:
                continue
            H.add_edge(ua, vb, link_rate_bps=1e-9, prop_delay=0.0)
            H.add_edge(vb, ua, link_rate_bps=1e-9, prop_delay=0.0)

        for br, (w0, w1) in br_attaches.items():
            H.add_edge(br, w0, link_rate_bps=1e-9, prop_delay=0.0)
            H.add_edge(w0, br, link_rate_bps=1e-9, prop_delay=0.0)
            H.add_edge(br, w1, link_rate_bps=1e-9, prop_delay=0.0)
            H.add_edge(w1, br, link_rate_bps=1e-9, prop_delay=0.0)

        _CACHED_SIM_TOPO = H
        _CACHED_BASE_NODESET = base_nodeset
    else:
        H = _CACHED_SIM_TOPO

    for u, v in routes.keys():
        if u not in _BR_NODES or v not in _BR_NODES:
            continue
        if not base.has_edge(u, v):
            continue
        base_rate = float(true_stream[(u, v)][0]) if (u, v) in true_stream else float(base.edges[(u, v)].get("link_capcapacity", 1e-9))
        r = float(base_rate) * float(rate_scale)
        D = float(base.edges[(u, v)].get("propagation_latency", 0.0))
        route = routes[(u, v)]
        k = max(1, len(route) - 1)
        per_pd = D / float(k)
        for a, b in zip(route, route[1:]):
            H.edges[(a, b)]["link_rate_bps"] = r
            H.edges[(a, b)]["prop_delay"] = per_pd
    return H


def _entries_from_policy(topology: nx.DiGraph, policy: list, chunk_size_bytes: Optional[int]) -> list:
    # mapping = extract_gpu_gpu_policies(topology, policy)
    mapping = policy
    entries = []
    routes, _ = _get_fixed_wan_routes_and_attaches()
    if chunk_size_bytes is None:
        chunk_size_bytes = int(float(config.packet_size) * (1024 ** 3))
    for buf, pairs in mapping.items():
        for (src, dst), path in pairs.items():
            path_list = list(path)
            if isinstance(routes, dict) and routes:
                expanded = [path_list[0]] if path_list else []
                for nxt in path_list[1:]:
                    prev = expanded[-1] if expanded else None
                    seg = routes.get((prev, nxt)) if prev is not None else None
                    if isinstance(seg, list) and len(seg) >= 2:
                        expanded.extend(seg[1:])
                    else:
                        expanded.append(nxt)
                path_list = expanded
            entries.append(
                PolicyEntry(
                    str(buf),
                    src,
                    dst,
                    0,
                    "Max",
                    int(chunk_size_bytes),
                    path_list,
                    time=0.0,
                    dependency=[],
                )
            )
    return entries


def simulate_policy_with_true_stream(policy: list,
                                     true_stream: Dict[Tuple[Any, Any], np.ndarray],
                                     packet_size_bytes: int = 1500,
                                     header_size_bytes: int = 0,
                                     chunk_size_bytes: Optional[int] = None,
                                     rate_scale: float = 1.0,
                                     topology: Optional[nx.DiGraph] = None) -> float:
    # print(policy, config.packet_size)
    if topology is None:
        dc = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk, chassis=config.chassis, name=config.topology_name)
        base = dc.topology
    else:
        base = topology
    eff_rate_scale = float(rate_scale) * 25 * 1024 * 1024 * 1024 * 8
    H = _build_sim_topology(base, true_stream, rate_scale=eff_rate_scale)
    entries = _entries_from_policy(base, policy, chunk_size_bytes)
    env = simpy.Environment()
    sim = Sim(env, H, packet_size_bytes=int(packet_size_bytes), header_size_bytes=int(header_size_bytes))
    sim.load_policy(entries)
    sim.start()
    sim.run()
    tx_times = dict(sim.tx_complete_time)
    makespan = max(tx_times.values()) if tx_times else 0.0
    # return {"makespan": makespan, "tx_complete_time": tx_times}
    return makespan

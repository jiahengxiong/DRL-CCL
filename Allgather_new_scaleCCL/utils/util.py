import os
from decimal import Decimal
import numpy as np
import config
from Allgather_TECCL.utils.NVD2_1_topology import NVD2_1_topology
from Allgather_TECCL.utils.DGX import DGX2
from NewAgent.load_dataset import load_snvang_dataset_by_day
from NewAgent.train import train_rnn_from_path_and_save, train_rnn_from_loaded_data_and_save
from NewAgent.deploy import RNNDeployer


def load_topology(packet_size, num_chunk, chassis, name, propagation_latency=None):
    global topology
    if name == 'NVD2':
        topology = NVD2_1_topology(
            num_chunk=num_chunk,
            packet_size=packet_size,
            propagation_latency=propagation_latency,
        )
    elif name == 'DGX':
        topology = DGX2(num_chunk=config.num_chunk, packet_size=config.packet_size)

    return topology


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_wan_dataset_dir():
    root = _project_root()
    env_dir = os.environ.get("WAN_LINK_DS_DIR")
    candidates = [
        env_dir if env_dir else None,
        os.path.join(root, "Allgather_new_scaleCCL", "WAN_link_dataset"),
        os.path.join(root, "NewAgent", "WAN_link_dataset"),
        os.path.join(root, "WAN_link_dataset"),
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p, True
    return os.path.join(root, "NewAgent", "WAN_link_dataset"), False


def resolve_wan_model_dir():
    root = _project_root()
    candidates = [
        os.path.join(root, "Allgather_new_scaleCCL", "WAN_link_models"),
        os.path.join(root, "NewAgent", "WAN_link_models"),
        os.path.join(root, "WAN_link_models"),
    ]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            continue
    return candidates[0]


def candidate_files_directional(u, v):
    a, b = str(u), str(v)
    return [
        f"{a}-{b}.txt",
        f"{a}_{b}.txt",
        f"link_{a}_{b}.txt",
        f"link_{a}-{b}.txt",
        f"node{a}_node{b}.txt",
        f"node{a}-node{b}.txt",
    ]


def _maybe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _wan_synth_weight() -> float:
    try:
        w = float(getattr(config, "wan_synth_weight", 0.9))
    except Exception:
        w = 0.9
    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return w


def _mix_real_syn(real: np.ndarray, syn: np.ndarray) -> np.ndarray:
    w = _wan_synth_weight()
    return ((1.0 - w) * real + w * syn).astype(np.float32)


def _synthetic_4link_values(T: int) -> np.ndarray:
    t = np.arange(T)
    k = np.floor(t / 10)

    sum_mid = 0.55
    sum_amp = 0.45
    sum0 = sum_mid + sum_amp * ((-1) ** k)
    sum1 = sum_mid - sum_amp * ((-1) ** k)

    frac0 = 0.80 + 0.05 * np.sin(0.10 * t)
    frac1 = 0.80 + 0.05 * np.cos(0.07 * t)
    d0 = sum0 * frac0
    d1 = sum1 * frac1

    r02 = (sum0 + d0 * ((-1) ** t)) / 2
    r03 = (sum0 - d0 * ((-1) ** t)) / 2
    r12 = (sum1 + d1 * ((-1) ** (t + 1))) / 2
    r13 = (sum1 - d1 * ((-1) ** (t + 1))) / 2

    return np.stack([r02, r03, r12, r13], axis=1).astype(np.float32)


def _apply_synthetic_correction(data: dict, pair) -> dict:
    u_i = _maybe_int(pair[0])
    v_i = _maybe_int(pair[1])
    if u_i is None or v_i is None:
        return data

    forward_map = {(0, 2): 0, (0, 3): 1, (1, 2): 2, (1, 3): 3}
    reverse_map = {(2, 0): 0, (2, 1): 1, (3, 0): 2, (3, 1): 3}
    col = forward_map.get((u_i, v_i))
    if col is None:
        col = reverse_map.get((u_i, v_i))
    if col is None:
        return data

    train_days = data.get("train_days", [])
    test_days = data.get("test_days", [])

    def _apply_segments(segments, col_idx: int):
        total_len = 0
        for seg in list(segments):
            x = getattr(seg, "x", None)
            if x is None:
                continue
            total_len += int(x.shape[0])
        if total_len <= 0:
            return
        f_all = _synthetic_4link_values(total_len)[:, col_idx]
        cursor = 0
        for seg in list(segments):
            x = getattr(seg, "x", None)
            if x is None or x.shape[0] == 0:
                continue
            T = int(x.shape[0])
            fx = f_all[cursor : cursor + T]
            cursor += T
            x1 = x.astype(np.float32).reshape(-1)
            x_new = _mix_real_syn(x1, fx.astype(np.float32)).reshape(-1, 1)
            seg.x = x_new

    _apply_segments(train_days, col)
    _apply_segments(test_days, col)

    return data



def load_directional_streams(graph, dataset_dir, runs, model_dir):
    if graph is None:
        dc = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk, chassis=config.chassis, name=config.topology_name)
        graph = dc.topology
    per_link_source = {}
    per_link_path = {}
    preview_printed = False
    for u, v in graph.edges:
        if graph.nodes[u].get('type') == 'switch' and graph.nodes[v].get('type') == 'switch':
            pair = (u, v)
            stream = None
            chosen_path = None
            cand_list = candidate_files_directional(pair[0], pair[1])
            for fname in cand_list:
                fpath = os.path.join(dataset_dir, fname)
                if os.path.exists(fpath):
                    chosen_path = fpath
                    model_name = f"rnn_link_{u}_{v}.pt"
                    model_path = os.path.join(model_dir, model_name)
                    if not os.path.exists(model_path):
                        train_model_if_needed(fpath, model_dir, u, v)
                    # Build predicted stream using Offline + Online FT (No Replay), consistent with NewAgent/plot.py scenario 2
                    data = load_snvang_dataset_by_day(fpath, train_ratio=0.5, scale_range=(0.1, 1.0))
                    deployer = RNNDeployer.from_checkpoint(model_path)
                    # Concat all test segments to a long stream, then predict with continual_learning=True
                    xs = []
                    for seg in data["test_days"]:
                        x = seg.x
                        if hasattr(x, "ndim") and x.ndim == 2 and x.shape[1] == 1:
                            x = x[:, 0]
                        xs.append(x.astype(np.float32))
                    y_stream = np.concatenate(xs, axis=0) if len(xs) > 0 else np.zeros((0,), dtype=np.float32)
                    deployer.begin_day()
                    preds = np.empty((len(y_stream),), dtype=np.float32)
                    for i, xt in enumerate(y_stream):
                        preds[i] = np.float32(deployer.predict(float(xt), continual_learning=True))
                    if len(preds) > 0:
                        stream = preds
                    break
            if not preview_printed:
                print(f"First WAN pair {pair}, candidate files: {cand_list}")
                preview_printed = True
            if stream is None or stream.size == 0:
                stream = np.ones((runs,), dtype=np.float32)
                per_link_source[pair] = "fallback_ones"
            else:
                per_link_source[pair] = f"pred:{os.path.basename(chosen_path)}" if chosen_path else "pred:unknown"
            per_link_stream[pair] = stream
            per_link_path[pair] = chosen_path
    return per_link_stream, per_link_source, per_link_path


def load_wan_models_and_data(graph, dataset_dir, model_dir):
    """
    加载 WAN 链路的模型和测试集数据。
    返回:
    - wan_env: dict, key=(u, v), value={'deployer': RNNDeployer, 'test_data': np.array, 'data_path': str}
    """
    if graph is None:
        dc = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk, chassis=config.chassis, name=config.topology_name)
        graph = dc.topology
        
    wan_env = {}
    
    for u, v in graph.edges:
        if graph.nodes[u].get('type') == 'switch' and graph.nodes[v].get('type') == 'switch':
            pair = (u, v)
            cand_list = candidate_files_directional(pair[0], pair[1])
            chosen_path = None
            
            for fname in cand_list:
                fpath = os.path.join(dataset_dir, fname)
                if os.path.exists(fpath):
                    chosen_path = fpath
                    break
            
            if chosen_path:
                # 加载数据 (只取测试集)
                data = load_snvang_dataset_by_day(chosen_path, train_ratio=0.5, scale_range=(0.1, 1.0))

                xs_raw = []
                for seg in data.get("test_days", []):
                    x = getattr(seg, "x", None)
                    if x is None:
                        continue
                    if hasattr(x, "ndim") and x.ndim == 2 and x.shape[1] == 1:
                        x = x[:, 0]
                    xs_raw.append(x.astype(np.float32))
                raw_test_stream = np.concatenate(xs_raw, axis=0) if len(xs_raw) > 0 else np.zeros((0,), dtype=np.float32)

                data = _apply_synthetic_correction(data, pair)

                model_name = f"rnn_link_{u}_{v}.pt"
                model_path = os.path.join(model_dir, model_name)
                if not os.path.exists(model_path):
                    train_rnn_from_loaded_data_and_save(
                        data=data,
                        save_path=model_path,
                        hidden_dim=64,
                        lr=1e-3,
                        epochs=200,
                        early_stop_patience=5,
                        early_stop_min_delta=0.0,
                    )

                test_stream = raw_test_stream
                
                # 加载 Deployer
                deployer = RNNDeployer.from_checkpoint(model_path)
                deployer.begin_day() # 初始化状态
                
                wan_env[pair] = {
                    'deployer': deployer,
                    'test_data': test_stream,
                    'data_path': chosen_path
                }
            else:
                # Fallback: 无数据，用全1
                wan_env[pair] = {
                    'deployer': None,
                    'test_data': np.ones((1000,), dtype=np.float32), # 假数据
                    'data_path': None
                }

    lengths_real = []
    for env in wan_env.values():
        td = env.get("test_data")
        if env.get("data_path") is not None and td is not None and int(getattr(td, "shape", [0])[0]) > 1:
            lengths_real.append(int(td.shape[0]))
    if lengths_real:
        min_len = min(lengths_real)
    else:
        lengths_any = []
        for env in wan_env.values():
            td = env.get("test_data")
            if td is not None and int(getattr(td, "shape", [0])[0]) > 1:
                lengths_any.append(int(td.shape[0]))
        min_len = min(lengths_any) if lengths_any else 0

    target_len = max(int(min_len), 1000)
    if target_len < 2:
        target_len = 2

    forward_map = {(0, 2): 0, (0, 3): 1, (1, 2): 2, (1, 3): 3}
    reverse_map = {(2, 0): 0, (2, 1): 1, (3, 0): 2, (3, 1): 3}
    syn = _synthetic_4link_values(target_len)

    def _pad_to_len(x: np.ndarray, n: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] >= n:
            return x[:n].astype(np.float32)
        if x.shape[0] == 0:
            return np.ones((n,), dtype=np.float32)
        pad_val = np.float32(x[-1])
        pad = np.full((n - x.shape[0],), pad_val, dtype=np.float32)
        return np.concatenate([x, pad], axis=0)

    for pair, env in wan_env.items():
        u_i = _maybe_int(pair[0])
        v_i = _maybe_int(pair[1])
        col = None
        if u_i is not None and v_i is not None:
            col = forward_map.get((u_i, v_i))
            if col is None:
                col = reverse_map.get((u_i, v_i))
        if col is not None:
            td = env.get("test_data")
            if td is None:
                td = np.ones((target_len,), dtype=np.float32)
            else:
                td = np.asarray(td, dtype=np.float32).reshape(-1)
                if min_len > 0:
                    td = td[:min_len]
                td = _pad_to_len(td, target_len)
            env["test_data"] = _mix_real_syn(td, syn[:, col].astype(np.float32))
        else:
            td = env.get("test_data")
            if td is None:
                td = np.ones((target_len,), dtype=np.float32)
            else:
                td = np.asarray(td, dtype=np.float32).reshape(-1)
                if min_len > 0:
                    td = td[:min_len]
                td = _pad_to_len(td, target_len)
            env["test_data"] = td

    return wan_env


def train_model_if_needed(
    data_path,
    model_dir,
    src,
    dst,
    epochs=5,
    hidden_dim=64,
    lr=1e-3,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.0,
):
    if not data_path:
        return
    model_name = f"rnn_link_{src}_{dst}.pt"
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        return
    train_rnn_from_path_and_save(
        data_path=data_path,
        save_path=model_path,
        train_ratio=0.5,
        scale_range=(0.1, 1.0),
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )


def update_wan_caps(topology, per_link_stream, datacenter, run_id=None, pos_map=None, advance=False):
    updated_caps = []
    for u, v in topology.edges:
        if topology.nodes[u].get('type') == 'switch' and topology.nodes[v].get('type') == 'switch':
            pair = (u, v)
            stream = per_link_stream.get(pair)
            if stream is None or getattr(stream, "size", 0) == 0:
                x_scaled = 1.0
            else:
                if pos_map is not None:
                    idx = pos_map.get(pair, 0)
                else:
                    idx = (run_id or 0) % len(stream)
                x_scaled = float(stream[idx])
            cap = Decimal('25') * Decimal(str(max(x_scaled, 1e-9)))
            e = topology.edges[(u, v)]
            e['link_capcapacity'] = cap
            e['transmission_latency'] = datacenter.packet_size / cap
            if 'weight' in e:
                e['weight'] = e.get('propagation_latency', 0) + e['transmission_latency']
            updated_caps.append((pair, cap))
            if pos_map is not None and advance and getattr(stream, "size", 0) > 0:
                pos_map[pair] = (idx + 1) % len(stream)
    return updated_caps

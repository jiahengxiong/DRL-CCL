import json
from typing import List
from decimal import Decimal
from collections import defaultdict

def rotate_ids(flows, shift=4):
    """
    将 rank 和 chunk_id 做循环左移 shift 位：
      new_rank = (old_rank - shift) mod N
      new_chunk = (old_chunk - shift) mod C
    其中 N = 最大 rank + 1（假设从0开始连续编号），
        C = 最大 chunk_id + 1
    """
    # 推断 rank 总数（假设 rank 连续从 0 开始）
    max_rank = -1
    for _, s, d, _ in flows:
        if int(s) > max_rank: max_rank = int(s)
        if int(d) > max_rank: max_rank = int(d)
    N = max_rank + 1

    # 推断 chunk 总数（假设 chunk_id 从 0 连续）
    max_chunk = max((int(f[0]) for f in flows), default=-1)
    C = max_chunk + 1 if max_chunk >= 0 else 0

    def map_rank(r):
        return (int(r) - shift) % N if N > 0 else int(r)

    def map_chunk(c):
        return (int(c) - shift) % C if C > 0 else int(c)

    rotated = []
    for c, s, d, t in flows:
        rotated.append([c, map_rank(s), map_rank(d), t])
    return rotated

def harmonize_prev_per_pair(single_flows):
    """
    让同一 (src,dst) 的所有 flow 的 prev 完全一致：
      - 若该 (src,dst) 对中存在非空 prev，则用“第一次出现的非空 prev”作为基准，
        将这条链路上所有 flow 的 prev 统一成该基准；
      - 若这条链路上的 prev 全为 []，则全部保持为空 []。
    不修改 parent_flow_id / child_flow_id 等其它字段。
    """
    from collections import defaultdict

    pair_to_idxs = defaultdict(list)
    for i, sf in enumerate(single_flows):
        pair_to_idxs[(sf["src"], sf["dest"])].append(i)

    for (src, dst), idxs in pair_to_idxs.items():
        base_prev = None
        for i in idxs:
            if single_flows[i]["prev"]:
                base_prev = list(single_flows[i]["prev"])
                break
        if base_prev is None:
            # 这条 (src,dst) 上全为空：全部保持空
            for i in idxs:
                single_flows[i]["prev"] = []
        else:
            # 统一为基准 prev
            for i in idxs:
                single_flows[i]["prev"] = list(base_prev)

def renumber_chunks_per_pair(flows, single_flows):
    """
    flows: [[old_chunk_id, src, dst, time], ...]
    single_flows: 你已经构造好的 SingleFlow 列表（包含 prev 和 parent/child 依赖）
    只修改：chunk_id、chunk_count；其它字段保持不变
    """
    pair_to_indices = defaultdict(list)
    for i, (_old_cid, src, dst, _t) in enumerate(flows):
        pair_to_indices[(int(src), int(dst))].append(i)

    for (src, dst), idx_list in pair_to_indices.items():
        total = len(idx_list)
        for new_cid, i in enumerate(idx_list):
            single_flows[i]["chunk_id"] = new_cid
            single_flows[i]["chunk_count"] = total

def flows_to_singleflow_json(
    flows: List[List[int]],
    json_path: str = "flow_model.json",
    default_flow_size: int = 0,
    default_channel_id: int = 0,
    conn_type: str = "RING",
):
    """
    flows: 形如 [[chunk_id, src, dst, time], ...]
    会生成一组 SingleFlow 字典并写入 json_path
    """
    # 计算 chunk_count（按最大 chunk_id + 1）
    max_chunk = max((f[0] for f in flows), default=-1)
    chunk_count = max_chunk + 1 if max_chunk >= 0 else 0

    single_flows = []
    flow_id = 0
    for chunk_id, src, dst, t in flows:
        sf = {
            "flow_id": flow_id,
            "src": int(src),
            "dest": int(dst),
            "flow_size": int(default_flow_size),
            "prev": [],                 # 无依赖则留空
            "parent_flow_id": [],       # 无父依赖则留空
            "child_flow_id": [],        # 无子依赖则留空
            "channel_id": default_channel_id,
            "chunk_id": int(chunk_id),
            "chunk_count": int(chunk_count),
            "conn_type": str(conn_type),

            # # 不是 SingleFlow 原生字段，但为保留信息写进 JSON
            # "time": t
        }
        single_flows.append(sf)
        flow_id += 1
    renumber_chunks_per_pair(flows, single_flows)
    for i in range(0, len(flows)):
        flow = flows[i]
        single_flow = single_flows[i]
        chunk_id = flow[0]
        src = flow[1]
        dst = flow[2]
        #必须收到之后才能发送这个chunk
        for j in range(0, i):
            if j != i:
                possible_flow = flows[j]
                possible_single_flow = single_flows[j]
                possible_chunk_id = possible_flow[0]
                possible_src = possible_flow[1]
                possible_dst = possible_flow[2]
                if chunk_id == possible_chunk_id and src == possible_dst:
                    single_flows[i]["parent_flow_id"].append(j)
                    single_flows[j]["child_flow_id"].append(i)
                    if len(single_flows[i]["prev"]) == 0:
                        single_flows[i]["prev"].append(possible_src)
                    break
        # single_flows[i]["prev"][0] = src
        #排序，必须等前一个flow发完之后，才发送这个flow
        for j in range(i - 1, -1, -1):  # 从 i-1 到 0，倒序
            possible_flow = flows[j]
            possible_single_flow = single_flows[j]
            possible_chunk_id = possible_flow[0]
            possible_src = possible_flow[1]
            possible_dst = possible_flow[2]
            if src == possible_src and dst == possible_dst:
                # if single_flows[i]["parent_flow_id"]:
                #     if flows[j][3] >= flows[single_flows[i]["parent_flow_id"][0]][3]:
                #         single_flows[i]["parent_flow_id"] = [j]
                #         single_flows[j]["child_flow_id"] = [i]
                single_flows[i]["parent_flow_id"].append(j)
                single_flows[j]["child_flow_id"].append(i)
                break
        if len(single_flows[i]["prev"]) == 0:
            for j in  range(0, len(flows)):
                if j != i:
                    possible_flow = flows[j]
                    possible_single_flow = single_flows[j]
                    possible_chunk_id = possible_flow[0]
                    possible_src = possible_flow[1]
                    possible_dst = possible_flow[2]
                    if possible_dst == src and possible_src not in [0,1,2,3] and possible_src != dst:
                        single_flows[i]["prev"].append(possible_src)
                        break
    harmonize_prev_per_pair(single_flows)

    # 也可以按“FlowModels”的习惯包装；这里输出一个简单的顶层对象
    out = {
        "FlowModels": single_flows
    }

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    return out

# 示例
if __name__ == "__main__":
    # 假设你的输入
    flows = [[0, 4, 0, Decimal('0.0003132')], [0, 4, 5, Decimal('0.00015695')], [0, 4, 7, Decimal('0.00015695')], [0, 4, 9, Decimal('0.00015695')], [1, 5, 1, Decimal('0.0003132')], [1, 5, 4, Decimal('0.00015695')], [1, 5, 6, Decimal('0.00015695')], [1, 5, 8, Decimal('0.00015695')], [2, 6, 0, Decimal('0.0003132')], [2, 6, 1, Decimal('0.0003132')], [2, 6, 5, Decimal('0.00015695')], [2, 6, 7, Decimal('0.00015695')], [2, 6, 8, Decimal('0.00015695')], [2, 6, 9, Decimal('0.00015695')], [3, 7, 0, Decimal('0.0003132')], [3, 7, 1, Decimal('0.0003132')], [3, 7, 4, Decimal('0.00015695')], [3, 7, 6, Decimal('0.00015695')], [3, 7, 8, Decimal('0.00015695')], [4, 8, 1, Decimal('0.0003132')], [4, 8, 5, Decimal('0.00015695')], [4, 8, 6, Decimal('0.00015695')], [4, 8, 7, Decimal('0.00015695')], [4, 8, 9, Decimal('0.00015695')], [5, 9, 1, Decimal('0.0003132')], [5, 9, 4, Decimal('0.00015695')], [5, 9, 6, Decimal('0.00015695')], [5, 9, 8, Decimal('0.00015695')], [6, 10, 2, Decimal('0.0003132')], [6, 10, 11, Decimal('0.00015695')], [6, 10, 14, Decimal('0.00015695')], [7, 11, 3, Decimal('0.0003132')], [7, 11, 10, Decimal('0.00015695')], [7, 11, 12, Decimal('0.00015695')], [7, 11, 13, Decimal('0.00015695')], [7, 11, 14, Decimal('0.00015695')], [7, 11, 15, Decimal('0.00015695')], [8, 12, 2, Decimal('0.0003132')], [8, 12, 11, Decimal('0.00015695')], [8, 12, 13, Decimal('0.00015695')], [8, 12, 14, Decimal('0.00015695')], [9, 13, 3, Decimal('0.0003132')], [9, 13, 11, Decimal('0.00015695')], [9, 13, 12, Decimal('0.00015695')], [9, 13, 14, Decimal('0.00015695')], [9, 13, 15, Decimal('0.00015695')], [10, 14, 2, Decimal('0.0003132')], [10, 14, 10, Decimal('0.00015695')], [10, 14, 11, Decimal('0.00015695')], [10, 14, 12, Decimal('0.00015695')], [10, 14, 13, Decimal('0.00015695')], [10, 14, 15, Decimal('0.00015695')], [11, 15, 2, Decimal('0.0003132')], [11, 15, 3, Decimal('0.0003132')], [11, 15, 11, Decimal('0.00015695')], [11, 15, 13, Decimal('0.00015695')], [11, 15, 14, Decimal('0.00015695')], [3, 0, 2, Decimal('0.0003632')], [2, 0, 3, Decimal('0.0003632')], [5, 1, 2, Decimal('0.0003632')], [4, 1, 3, Decimal('0.0003632')], [10, 2, 0, Decimal('0.0003632')], [11, 2, 1, Decimal('0.0003632')], [7, 3, 0, Decimal('0.0003632')], [9, 3, 1, Decimal('0.0003632')], [10, 0, 4, Decimal('0.0003639')], [7, 0, 6, Decimal('0.0003639')], [7, 0, 7, Decimal('0.0003639')], [11, 1, 5, Decimal('0.0003639')], [9, 1, 6, Decimal('0.0003639')], [9, 1, 7, Decimal('0.0003639')], [9, 1, 8, Decimal('0.0003639')], [9, 1, 9, Decimal('0.0003639')], [3, 2, 10, Decimal('0.0003639')], [5, 2, 12, Decimal('0.0003639')], [5, 2, 14, Decimal('0.0003639')], [5, 2, 15, Decimal('0.0003639')], [2, 3, 11, Decimal('0.0003639')], [4, 3, 13, Decimal('0.0003639')], [4, 3, 15, Decimal('0.0003639')], [3, 4, 5, Decimal('0.00031390')], [1, 4, 7, Decimal('0.00031390')], [1, 4, 9, Decimal('0.00031390')], [2, 5, 4, Decimal('0.00031390')], [0, 5, 6, Decimal('0.00031390')], [0, 5, 8, Decimal('0.00031390')], [5, 6, 5, Decimal('0.00031390')], [5, 6, 7, Decimal('0.00031390')], [3, 6, 9, Decimal('0.00031390')], [4, 7, 4, Decimal('0.00031390')], [8, 11, 10, Decimal('0.00031390')], [6, 11, 12, Decimal('0.00031390')], [6, 11, 13, Decimal('0.00031390')], [6, 11, 15, Decimal('0.00031390')], [11, 13, 12, Decimal('0.00031390')], [8, 13, 15, Decimal('0.00031390')], [9, 14, 10, Decimal('0.00031390')], [0, 0, 2, Decimal('0.0006757')], [1, 1, 2, Decimal('0.0006757')], [6, 2, 0, Decimal('0.0006757')], [8, 2, 1, Decimal('0.0006757')], [11, 11, 10, Decimal('0.00047015')], [7, 0, 4, Decimal('0.0006764')], [6, 0, 6, Decimal('0.0006764')], [6, 0, 7, Decimal('0.0006764')], [9, 1, 5, Decimal('0.0006764')], [8, 1, 6, Decimal('0.0006764')], [8, 1, 7, Decimal('0.0006764')], [8, 1, 8, Decimal('0.0006764')], [8, 1, 9, Decimal('0.0006764')], [5, 2, 10, Decimal('0.0006764')], [0, 2, 12, Decimal('0.0006764')], [0, 2, 14, Decimal('0.0006764')], [0, 2, 15, Decimal('0.0006764')], [4, 3, 11, Decimal('0.0006764')], [2, 3, 13, Decimal('0.0006764')], [2, 3, 15, Decimal('0.0006764')], [10, 4, 5, Decimal('0.00052085')], [10, 4, 7, Decimal('0.00052085')], [10, 4, 9, Decimal('0.00052085')], [11, 5, 4, Decimal('0.00052085')], [11, 5, 6, Decimal('0.00052085')], [11, 5, 8, Decimal('0.00052085')], [7, 6, 5, Decimal('0.00052085')], [7, 6, 8, Decimal('0.00052085')], [7, 6, 9, Decimal('0.00052085')], [9, 7, 4, Decimal('0.00052085')], [3, 10, 11, Decimal('0.00052085')], [3, 10, 14, Decimal('0.00052085')], [2, 11, 12, Decimal('0.00052085')], [2, 11, 14, Decimal('0.00052085')], [5, 12, 11, Decimal('0.00052085')], [5, 12, 13, Decimal('0.00052085')], [4, 13, 12, Decimal('0.00052085')], [4, 13, 14, Decimal('0.00052085')], [2, 11, 10, Decimal('0.00062640')], [11, 4, 7, Decimal('0.00067780')], [11, 4, 9, Decimal('0.00067780')], [10, 5, 6, Decimal('0.00067780')], [10, 5, 8, Decimal('0.00067780')], [3, 11, 12, Decimal('0.00067780')], [3, 11, 13, Decimal('0.00067780')], [3, 11, 15, Decimal('0.00067780')], [4, 14, 10, Decimal('0.00067780')], [6, 0, 4, Decimal('0.0009889')], [8, 1, 5, Decimal('0.0009889')], [0, 2, 10, Decimal('0.0009889')], [1, 2, 12, Decimal('0.0009889')], [1, 2, 14, Decimal('0.0009889')], [1, 2, 15, Decimal('0.0009889')], [6, 6, 5, Decimal('0.00083335')], [6, 6, 8, Decimal('0.00083335')], [6, 6, 9, Decimal('0.00083335')], [8, 7, 4, Decimal('0.00083335')], [0, 12, 11, Decimal('0.00083335')], [0, 12, 13, Decimal('0.00083335')], [1, 2, 10, Decimal('0.0013014')], [1, 12, 11, Decimal('0.00114585')], [1, 12, 13, Decimal('0.00114585')]]
    # 使用方式：
    # 1) 先旋转
    # flows_rot = rotate_ids(flows, shift=4)
    # 2) 再用你现有的 JSON 生成函数
    flows_to_singleflow_json(flows, "flow_model.json",
                             default_flow_size=16 * 8 * 1024,
                             default_channel_id=0)
    # flows_to_singleflow_json(flows, "flow_model.json", default_flow_size=16*8*1024, default_channel_id=0)
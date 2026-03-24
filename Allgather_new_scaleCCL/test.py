from path_ass import solve_time_indexed_milp

# 1) 定义 flows（单位：size 比特，arrival_time 秒，rate_limit bps）
flows = [
    {"id": 101, "size": 1.6e9, "arrival_time": 0.00, "rate_limit": 2.5e9},
    {"id": 102, "size": 1.2e9, "arrival_time": 0.01, "rate_limit": 2.5e9},
    {"id": 103, "size": 2.0e9, "arrival_time": 0.00, "rate_limit": 2.5e9},
]

# 2) 定义共享路径（capacity bps，delay 秒）
shared_paths = [
    {"id": 0, "capacity": 6.5e9, "delay": 0.0},
    {"id": 1, "capacity": 15.6e9, "delay": 0.0},
]

# 3) 为每个 flow（按索引 i）指定可用路径 id 集合
flow_path_map = {
    0: [0, 1],
    1: [0, 1],
    2: [0, 1],
}

# 4) 时间参数（秒）
sum_size = sum(f["size"] for f in flows)
sum_cap = sum(p["capacity"] for p in shared_paths)
sum_rmax = sum(f.get("rate_limit", 0.0) for f in flows)
max_arrival = max(f["arrival_time"] for f in flows)
effective_total_rate = min(sum_cap, sum_rmax) if (sum_cap > 0 and sum_rmax > 0) else max(sum_cap, sum_rmax, 1.0)
lb_total = sum_size / effective_total_rate
lb_flow = max(f["size"] / f.get("rate_limit", 1.0) for f in flows)
time_horizon = max_arrival + 2.0 * max(lb_total, lb_flow)
min_eff_rate = min(min(p["capacity"] for p in shared_paths), min(f.get("rate_limit", 1.0) for f in flows))
time_slot_duration = max(1e-6, (min(f["size"] for f in flows) / min_eff_rate) / 50.0)

# 5) 调用 MILP
sol = solve_time_indexed_milp(
    flows=flows,
    shared_paths=shared_paths,
    flow_path_map=flow_path_map,
    time_horizon=time_horizon,
    time_slot_duration=time_slot_duration,
    use_fairness=True,
    time_limit=60,
    subslots_per_segment=5,
)

print("status:", sol["status"])
print("makespan:", sol["makespan"])
print("assignments:", sol["assignments"])
print("completion_times:", sol["completion_times"])

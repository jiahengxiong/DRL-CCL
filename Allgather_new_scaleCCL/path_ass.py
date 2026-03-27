#!/usr/bin/env python3
"""
Time-Indexed MILP with Injection-Capped Fair Sharing
Path Assignment Optimization for Allgather
Based on user-provided formulation.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import json
import networkx as nx
import itertools


# ... (keep generate_candidate_paths but maybe unused) ...

def solve_time_indexed_milp(flows, shared_paths, flow_path_map, time_horizon, time_slot_duration=1e-5, use_fairness=True, time_limit=60, subslots_per_segment=5):
    """
    Solves the Time-Indexed MILP formulation.
    
    Args:
        flows: List of flow dictionaries
        shared_paths: List of path dictionaries (the shared resources)
                      Each dict must have 'id', 'capacity', 'delay'
        flow_path_map: Dictionary mapping flow_id to list of allowed path_ids
        time_horizon: Total simulation time (seconds)
        time_slot_duration: Delta (seconds)
        
    Returns:
        solution: Dictionary containing results
    """
    
    # Parameters
    num_flows = len(flows)
    horizon_rounded = round(float(time_horizon), 9)

    # Use provided paths mapping
    paths_per_flow = flow_path_map

        
    # Constants
    epsilon = 1e-6
    epsilon_rate = 1e-6
    
    # Flow parameters
    s = {}
    a = {}
    r_max = {}
    
    for i, flow in enumerate(flows):
        s[i] = flow['size']
        a[i] = flow['arrival_time']
        r_max[i] = flow.get('rate_limit', 100.0)
        
    # Path parameters
    C = {}
    d = {}
    for p in shared_paths:
        C[p['id']] = p['capacity']
        d[p['id']] = p['delay']

    min_size = float(min(s.values())) if s else 0.0
    max_cap = float(max(C.values())) if C else 0.0
    if min_size > 0 and max_cap > 0:
        Delta = min_size / max_cap
    else:
        Delta = float(time_slot_duration) if float(time_slot_duration) > 0 else 1e-5

    start_times = []
    end_times = []
    durations = []
    t = 0.0
    while t < horizon_rounded - 1e-12:
        t2 = min(horizon_rounded, t + Delta)
        start_times.append(t)
        end_times.append(t2)
        durations.append(t2 - t)
        t = t2

    K = len(durations)
    print(f"MILP timeslot: Delta={Delta:.6g}s, K={K}")
        
    # Big M constants
    # Use time_horizon for upper bound on total service time
    M_bits = {i: r_max[i] * time_horizon for i in range(num_flows)}
    
    # M_rate >= max(max C_j, max r_max_i)
    max_C = max(C.values()) if C else 0
    max_r = max(r_max.values()) if r_max else 0
    M_rate = max(max_C, max_r) * 2 # Safety margin
    
    # Initialize Model
    model = gp.Model("TimeIndexedMILP")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPFocus', 1)
    model.setParam('Heuristics', 0.2)
    
    # --- Decision Variables ---
    
    # x_ij: 1 if flow i uses path j
    x = {}
    for i in range(num_flows):
        for j in paths_per_flow[i]:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            
    # y_ijk: 1 if flow i is active on (j, k)
    y = {}
    # r_ijk: rate of flow i on (j, k)
    r = {}
    # b_ijk: 1 if flow i is capped on (j, k)
    b = {}
    # u_ijk: 1 if flow i is uncapped on (j, k)
    u = {}
    
    for i in range(num_flows):
        for j in paths_per_flow[i]:
            for k in range(K):
                y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")
                r[i, j, k] = model.addVar(lb=0.0, name=f"r_{i}_{j}_{k}")
                if use_fairness:
                    b[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}_{k}")
                    u[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{j}_{k}")

    # z_ik: 1 if flow i completes in slot k
    z = {}
    for i in range(num_flows):
        for k in range(K):
            z[i, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")
            
    # T_i: completion time of flow i
    T = {}
    for i in range(num_flows):
        T[i] = model.addVar(lb=0.0, name=f"T_{i}")
        
    # M: makespan
    M = model.addVar(lb=0.0, name="Makespan")
    
    # q_jk: fair-share water level for path j at slot k
    q = {}
    if use_fairness:
        for p in shared_paths:
            j = p['id']
            for k in range(K):
                q[j, k] = model.addVar(lb=0.0, name=f"q_{j}_{k}")
            
    # Auxiliary: B_ik cumulative bits
    # ... (rest of constraints remain similar, checking loops)
    
    # --- Constraints ---
    
    # (1) Path assignment
    for i in range(num_flows):
        model.addConstr(gp.quicksum(x[i, j] for j in paths_per_flow[i]) == 1, name=f"PathAssign_{i}")
        
    # (2) Activity only on chosen path
    for i in range(num_flows):
        for j in paths_per_flow[i]:
            for k in range(K):
                model.addConstr(y[i, j, k] <= x[i, j], name=f"ActivityPath_{i}_{j}_{k}")
                
    # (3) No service before arrival
    for i in range(num_flows):
        for k in range(K):
            if end_times[k] <= a[i]:
                for j in paths_per_flow[i]:
                    model.addConstr(y[i, j, k] == 0, name=f"NoServiceBeforeArrival_{i}_{j}_{k}")
                
    # (4) Injection bound
    for i in range(num_flows):
        for j in paths_per_flow[i]:
            for k in range(K):
                model.addConstr(r[i, j, k] <= r_max[i] * y[i, j, k], name=f"InjectionBound_{i}_{j}_{k}")
                
    # (5) Capacity constraint
    for p in shared_paths:
        j = p['id']
        for k in range(K):
            # Sum over all flows that MIGHT use path j
            relevant_flows = [i for i in range(num_flows) if j in paths_per_flow[i]]
            if relevant_flows:
                model.addConstr(gp.quicksum(r[i, j, k] for i in relevant_flows) <= C[j], name=f"Capacity_{j}_{k}")
            
    # (6) Completion slot
    for i in range(num_flows):
        model.addConstr(gp.quicksum(z[i, k] for k in range(K)) == 1, name=f"CompletionSlot_{i}")
        
    # (6.1) No completion before arrival
    for i in range(num_flows):
        for k in range(K):
            if end_times[k] <= a[i]:
                model.addConstr(z[i, k] == 0, name=f"NoCompletionBeforeArrival_{i}_{k}")
            
    # (7) First time reaching size
    B = {}
    for i in range(num_flows):
        for k in range(K):
            B[i, k] = model.addVar(lb=0.0, name=f"B_{i}_{k}")
            
    for i in range(num_flows):
        for k in range(K):
            prev_B = B[i, k-1] if k > 0 else 0
            current_rate_sum = gp.quicksum(r[i, j, k] for j in paths_per_flow[i])
            model.addConstr(B[i, k] == prev_B + current_rate_sum * durations[k], name=f"B_accum_{i}_{k}")
            model.addConstr(B[i, k] >= s[i] * z[i, k], name=f"SizeReached_{i}_{k}")
            if k > 0:
                model.addConstr(B[i, k-1] <= (s[i] - epsilon) + M_bits[i] * (1 - z[i, k]), name=f"NotFinishedYet_{i}_{k}")
                
    # (8) Switch semantics (event-based arrival index)
    for i in range(num_flows):
        for k in range(K):
            if end_times[k] > a[i]:
                sum_z_prev = gp.quicksum(z[i, m] for m in range(k))
                for j in paths_per_flow[i]:
                    model.addConstr(y[i, j, k] >= x[i, j] - sum_z_prev, name=f"NoIdling_{i}_{j}_{k}")
                
    # (9) No transmission after completion
    for i in range(num_flows):
        for k in range(K):
            sum_z_prev = gp.quicksum(z[i, m] for m in range(k))
            for j in paths_per_flow[i]:
                model.addConstr(y[i, j, k] <= 1 - sum_z_prev, name=f"NoTxAfterComp_{i}_{j}_{k}")

    # (10) Completion time
    for i in range(num_flows):
        model.addConstr(T[i] == gp.quicksum(end_times[k] * z[i, k] for k in range(K)), name=f"CompTimeCalc_{i}")
        
    # (11) Makespan
    for i in range(num_flows):
        path_delay = gp.quicksum(d[j] * x[i, j] for j in paths_per_flow[i])
        model.addConstr(M >= T[i] + path_delay, name=f"MakespanConstr_{i}")
        
    # (12) Fair sharing constraints
    if use_fairness:
        for i in range(num_flows):
            for j in paths_per_flow[i]:
                for k in range(K):
                    model.addConstr(b[i, j, k] <= y[i, j, k], name=f"CapActive_{i}_{j}_{k}")
                    model.addConstr(u[i, j, k] <= y[i, j, k], name=f"UncapActive_{i}_{j}_{k}")
                    model.addConstr(u[i, j, k] + b[i, j, k] == y[i, j, k], name=f"SplitState_{i}_{j}_{k}")
                    model.addGenConstrIndicator(y[i, j, k], 0, r[i, j, k], GRB.EQUAL, 0.0, name=f"InactiveRate_{i}_{j}_{k}")
                    model.addGenConstrIndicator(u[i, j, k], 1, r[i, j, k] - q[j, k], GRB.EQUAL, 0.0, name=f"UncappedEq_{i}_{j}_{k}")
                    model.addGenConstrIndicator(b[i, j, k], 1, r[i, j, k] - r_max[i], GRB.EQUAL, 0.0, name=f"CappedEq_{i}_{j}_{k}")
                    # Consistency: uncapped implies q <= r_max[i]
                    model.addGenConstrIndicator(u[i, j, k], 1, q[j, k] - r_max[i], GRB.LESS_EQUAL, 0.0, name=f"UncappedBound_{i}_{j}_{k}")
                    # Consistency: capped implies q >= r_max[i] + epsilon_rate
                    model.addGenConstrIndicator(b[i, j, k], 1, q[j, k] - r_max[i], GRB.GREATER_EQUAL, epsilon_rate, name=f"CappedBound_{i}_{j}_{k}")
                
    # (12.4) q <= C_j
    if use_fairness:
        for p in shared_paths:
            j = p['id']
            for k in range(K):
                model.addConstr(q[j, k] <= C[j], name=f"FairShareCap_{j}_{k}")
            
    # Objective
    model.setObjective(M, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    # Extract results
    solution = {
        'status': model.status,
        'obj_val': None,
        'assignments': {},
        'completion_times': {},
        'makespan': None
    }
    
    if model.SolCount and model.SolCount > 0:
        solution['obj_val'] = model.objVal
        try:
            solution['makespan'] = M.X
        except Exception:
            solution['makespan'] = None
        for i in range(num_flows):
            fid_raw = flows[i].get('id', i)
            try:
                fid = int(fid_raw)
            except Exception:
                fid = fid_raw
            try:
                solution['completion_times'][fid] = T[i].X
            except Exception:
                solution['completion_times'][fid] = None
            for j in paths_per_flow[i]:
                try:
                    if x[i, j].X > 0.5:
                        solution['assignments'][fid] = j
                except Exception:
                    pass
        print(f"Incumbent objective: {solution['obj_val']}")
        print("Assignments:", solution['assignments'])
        print("Completion Times:", solution['completion_times'])
        print("Summary (flow_id, arrival, completion, path):")
        for i in range(num_flows):
            fid_raw = flows[i].get('id', i)
            try:
                fid = int(fid_raw)
            except Exception:
                fid = fid_raw
            arr = a[i]
            comp = solution['completion_times'].get(fid, None)
            path = solution['assignments'].get(fid, None)
            print(f"{fid}\\t{arr:.6f}\\t{(comp if comp is None else round(comp, 6))}\\t{path}")
    else:
        print(f"Optimization ended with status {model.status}")
        if model.status == GRB.INFEASIBLE:
             model.computeIIS()
             model.write("model.ilp")
             print("IIS written to model.ilp")
        
    return solution

def build_instance(n_inputs=3, m_outputs=2, flows_per_input=None, size_bits=200, rate_limit=1e9, capacities=None, delays=None, delta=0.05, horizon=None, input_rate_limits=None, arrival_range=None, seed=None, flow_arrival_times=None, flow_arrival_map=None, flow_ids=None, desired_slots_per_flow=50, max_slots=None, restrict_to_fast_path=False):
    import random
    if seed is not None:
        random.seed(seed)
    if flows_per_input is None:
        flows_per_input = [1] * n_inputs
    total_flows = sum(flows_per_input)
    ids_list = None
    if flow_ids is not None and len(flow_ids) == total_flows:
        ids_list = list(flow_ids)
    elif flow_arrival_map is not None and len(flow_arrival_map) == total_flows:
        ids_list = list(flow_arrival_map.keys())
    flows = []
    flow_id = 0
    for src in range(n_inputs):
        for _ in range(flows_per_input[src]):
            rl = input_rate_limits[src] if (input_rate_limits is not None and src < len(input_rate_limits)) else rate_limit
            if ids_list is not None:
                ext_id = ids_list[flow_id]
            else:
                ext_id = flow_id
            if flow_arrival_times is not None and flow_id < len(flow_arrival_times):
                a_time = float(flow_arrival_times[flow_id])
            elif arrival_range is not None:
                a_time = random.uniform(arrival_range[0], arrival_range[1])
            else:
                a_time = 0.0
            if flow_arrival_map is not None and ext_id in flow_arrival_map:
                a_time = float(flow_arrival_map[ext_id])
            flows.append({"id": ext_id, "src": src, "dst": 0, "size": size_bits, "arrival_time": a_time, "rate_limit": rl})
            flow_id += 1
    if capacities is None:
        capacities = [1000.0] * m_outputs
    if delays is None:
        delays = [0.001 * (1 + j) for j in range(m_outputs)]
    shared_paths = [{"id": j, "capacity": capacities[j], "delay": delays[j]} for j in range(m_outputs)]
    if restrict_to_fast_path:
        j_best = int(np.argmax(capacities)) if capacities else 0
        flow_path_map = {i: [j_best] for i in range(total_flows)}
    else:
        flow_path_map = {i: list(range(m_outputs)) for i in range(total_flows)}
    if horizon is None:
        total_bits = sum(f["size"] for f in flows)
        sum_C = sum(capacities) if capacities else 1.0
        max_arrival = max((f["arrival_time"] for f in flows), default=0.0)
        horizon = max_arrival + 1.5 * total_bits / sum_C
    if delta is None:
        min_C = min(capacities) if capacities else 1.0
        base_size = flows[0]["size"] if flows else 0.0
        delta = max(1e-6, (base_size / min_C) / float(desired_slots_per_flow))
    if max_slots is not None:
        K_est = int(horizon / delta) + 1
        if K_est > max_slots:
            delta = horizon / float(max_slots)
    return flows, shared_paths, flow_path_map, horizon, delta
 
def solve_single_node(n_inputs=3, m_outputs=2, flows_per_input=None, size_bits=200, rate_limit=1e9, capacities=None, delays=None, delta=0.05, horizon=None, input_rate_limits=None, arrival_range=None, seed=None, flow_arrival_times=None, flow_arrival_map=None, flow_ids=None, use_fairness=True, desired_slots_per_flow=50, max_slots=None, restrict_to_fast_path=False, time_limit=60):
    flows, shared_paths, flow_path_map, H, dlt = build_instance(n_inputs, m_outputs, flows_per_input, size_bits, rate_limit, capacities, delays, delta, horizon, input_rate_limits, arrival_range, seed, flow_arrival_times, flow_arrival_map, flow_ids, desired_slots_per_flow, max_slots, restrict_to_fast_path)
    return solve_time_indexed_milp(flows, shared_paths, flow_path_map, time_horizon=H, time_slot_duration=dlt, use_fairness=use_fairness, time_limit=time_limit)
 
def _fair_share_rates(capacity, rmax):
    n = len(rmax)
    if n == 0:
        return []
    rmax_sorted = sorted(rmax)
    prefix = [0.0]
    for v in rmax_sorted:
        prefix.append(prefix[-1] + v)
    q = 0.0
    for k in range(n):
        q_candidate = (capacity - prefix[k]) / float(n - k)
        if k == n - 1 or q_candidate <= rmax_sorted[k + 1]:
            q = max(0.0, q_candidate)
            break
    rates = []
    for v in rmax:
        rates.append(min(v, q))
    return rates
 
def simulate_fair_completion(flows, capacities, assignments):
    ids = [f.get("id", i) for i, f in enumerate(flows)]
    id_to_index = {ids[i]: i for i in range(len(ids))}
    rem = {ids[i]: float(flows[i]["size"]) for i in range(len(ids))}
    rmax = {ids[i]: float(flows[i].get("rate_limit", 100.0)) for i in range(len(ids))}
    arrivals = {ids[i]: float(flows[i]["arrival_time"]) for i in range(len(ids))}
    by_path = {}
    for fid, j in assignments.items():
        by_path.setdefault(j, []).append(fid)
    active = {j: [] for j in range(len(capacities))}
    done = set()
    comp = {}
    t = 0.0
    while True:
        for j in range(len(capacities)):
            for fid in by_path.get(j, []):
                if fid not in active[j] and fid not in done and arrivals[fid] <= t and rem[fid] > 0.0:
                    active[j].append(fid)
        rates_per_path = {}
        for j in range(len(capacities)):
            fids = [fid for fid in active[j] if rem[fid] > 0.0]
            rmax_list = [rmax[fid] for fid in fids]
            rates = _fair_share_rates(capacities[j], rmax_list) if fids else []
            rates_per_path[j] = dict(zip(fids, rates))
        next_events = []
        for j in range(len(capacities)):
            for fid, rate in rates_per_path[j].items():
                if rate > 0.0 and rem[fid] > 0.0:
                    next_events.append(rem[fid] / rate)
        upcoming_arrivals = []
        for j in range(len(capacities)):
            for fid in by_path.get(j, []):
                if arrivals[fid] > t and fid not in done and rem[fid] > 0.0:
                    upcoming_arrivals.append(arrivals[fid] - t)
        if not next_events and not upcoming_arrivals:
            break
        dt = min(next_events + upcoming_arrivals) if (next_events or upcoming_arrivals) else 0.0
        if dt <= 0.0:
            dt = 0.0
        for j in range(len(capacities)):
            for fid, rate in rates_per_path[j].items():
                rem[fid] = max(0.0, rem[fid] - rate * dt)
        t += dt
        for j in range(len(capacities)):
            new_active = []
            for fid in active[j]:
                if rem[fid] <= 0.0 and fid not in done:
                    done.add(fid)
                    comp[fid] = t
                else:
                    new_active.append(fid)
            active[j] = new_active
    makespan = max(comp.values()) if comp else None
    return comp, makespan
 
def choose_assignments_balanced(flows, capacities):
    ids = [f.get("id", i) for i, f in enumerate(flows)]
    sizes = [float(f["size"]) for f in flows]
    arrs = [float(f["arrival_time"]) for f in flows]
    order = sorted(range(len(ids)), key=lambda i: arrs[i])
    load = [0.0 for _ in capacities]
    assign = {}
    for i in order:
        best_j = None
        best_score = None
        for j in range(len(capacities)):
            score = (load[j] + sizes[i]) / capacities[j]
            if best_score is None or score < best_score:
                best_score = score
                best_j = j
        assign[ids[i]] = best_j
        load[best_j] += sizes[i]
    return assign
 
def choose_assignments_optimal(flows, capacities):
    ids = [f.get("id", i) for i, f in enumerate(flows)]
    n = len(ids)
    m = len(capacities)
    if m != 2 or n > 18:
        return choose_assignments_balanced(flows, capacities)
    best_assign = None
    best_mk = None
    for mask in range(1 << n):
        assign = {}
        for i in range(n):
            j = 1 if (mask >> i) & 1 else 0
            assign[ids[i]] = j
        comp, mk = simulate_fair_completion(flows, capacities, assign)
        if best_mk is None or (mk is not None and mk < best_mk):
            best_mk = mk
            best_assign = assign
    if best_assign is None:
        best_assign = choose_assignments_balanced(flows, capacities)
        comp, best_mk = simulate_fair_completion(flows, capacities, best_assign)
    return best_assign, best_mk

def _simulate_path_completion(active_fids, rem, rmax, capacity):
    t = 0.0
    comp = {}
    fids = [fid for fid in active_fids if rem[fid] > 0.0]
    while True:
        rmax_list = [rmax[fid] for fid in fids]
        rates = _fair_share_rates(capacity, rmax_list) if fids else []
        rate_map = dict(zip(fids, rates))
        next_events = []
        for fid in fids:
            rate = rate_map.get(fid, 0.0)
            if rate > 0.0 and rem[fid] > 0.0:
                next_events.append(rem[fid] / rate)
        if not next_events:
            break
        dt = min(next_events)
        for fid, rate in rate_map.items():
            rem[fid] = max(0.0, rem[fid] - rate * dt)
        t += dt
        new_fids = []
        for fid in fids:
            if rem[fid] <= 0.0 and fid not in comp:
                comp[fid] = t
            else:
                new_fids.append(fid)
        fids = new_fids
    return comp, t

def simulate_switch_assignment_precise(flows, capacities, delays=None):
    ids = [f.get("id", i) for i, f in enumerate(flows)]
    arrivals = {ids[i]: float(flows[i]["arrival_time"]) for i in range(len(ids))}
    rem = {ids[i]: float(flows[i]["size"]) for i in range(len(ids))}
    rmax = {ids[i]: float(flows[i].get("rate_limit", 100.0)) for i in range(len(ids))}
    if delays is None:
        delays = [0.0 for _ in capacities]
    by_path = {j: [] for j in range(len(capacities))}
    active = {j: [] for j in range(len(capacities))}
    assigned = {}
    comp = {}
    t = 0.0
    pending = sorted(ids, key=lambda fid: arrivals[fid])
    idx = 0
    while True:
        while idx < len(pending) and arrivals[pending[idx]] <= t:
            fid = pending[idx]
            best_j = None
            best_path_mk = None
            for j in range(len(capacities)):
                test_rem = {k: rem[k] for k in rem}
                test_active = list(active[j])
                test_active.append(fid)
                cdict, path_mk = _simulate_path_completion(test_active, test_rem, rmax, capacities[j])
                if best_path_mk is None or path_mk < best_path_mk:
                    best_path_mk = path_mk
                    best_j = j
            assigned[fid] = best_j
            by_path[best_j].append(fid)
            if fid not in active[best_j]:
                active[best_j].append(fid)
            idx += 1
        rates_per_path = {}
        for j in range(len(capacities)):
            fids = [fid for fid in active[j] if rem[fid] > 0.0]
            rmax_list = [rmax[fid] for fid in fids]
            rates = _fair_share_rates(capacities[j], rmax_list) if fids else []
            rates_per_path[j] = dict(zip(fids, rates))
        next_completions = []
        for j in range(len(capacities)):
            for fid, rate in rates_per_path[j].items():
                if rate > 0.0 and rem[fid] > 0.0:
                    next_completions.append(rem[fid] / rate)
        next_arrival = None
        if idx < len(pending):
            next_arrival = arrivals[pending[idx]] - t
        if not next_completions and next_arrival is None:
            break
        candidates = []
        if next_completions:
            candidates.append(min(next_completions))
        if next_arrival is not None and next_arrival > 0.0:
            candidates.append(next_arrival)
        if not candidates:
            dt = 0.0
        else:
            dt = min(candidates)
        for j in range(len(capacities)):
            for fid, rate in rates_per_path[j].items():
                rem[fid] = max(0.0, rem[fid] - rate * dt)
        t += dt
        for j in range(len(capacities)):
            new_active = []
            for fid in active[j]:
                if rem[fid] <= 0.0 and fid not in comp:
                    comp[fid] = t + delays[j]
                else:
                    new_active.append(fid)
            active[j] = new_active
    makespan = max(comp.values()) if comp else None
    return assigned, comp, makespan

def _simulate_path_completion_packet(active_fids, rem, rmax, capacity, packet_size):
    t = 0.0
    comp = {}
    next_t = {}
    fids = [fid for fid in active_fids if rem[fid] > 0.0]
    link_free = 0.0
    while True:
        if not fids:
            break
        rmax_list = [rmax[fid] for fid in fids]
        rates = _fair_share_rates(capacity, rmax_list)
        rate_map = dict(zip(fids, rates))
        for fid in fids:
            if fid not in next_t:
                next_t[fid] = t
        candidates = [(fid, next_t[fid]) for fid in fids if rate_map.get(fid, 0.0) > 0.0 and rem[fid] > 0.0]
        if not candidates:
            break
        fid_sel, t_pkt = min(candidates, key=lambda x: x[1])
        start = max(t_pkt, link_free)
        dur = packet_size / capacity
        finish = start + dur
        send_rate = rate_map.get(fid_sel, 0.0)
        if send_rate <= 0.0:
            break
        rem[fid_sel] = max(0.0, rem[fid_sel] - packet_size)
        next_t[fid_sel] = start + (packet_size / send_rate)
        link_free = finish
        t = finish
        new_fids = []
        for fid in fids:
            if rem[fid] <= 0.0 and fid not in comp:
                comp[fid] = t
            else:
                new_fids.append(fid)
        fids = new_fids
    mk = max(comp.values()) if comp else 0.0
    return comp, mk

def simulate_switch_packet_level(flows, capacities, delays=None, packet_size=1e-6):
    ids = [f.get("id", i) for i, f in enumerate(flows)]
    arrivals = {ids[i]: float(flows[i]["arrival_time"]) for i in range(len(ids))}
    rem = {ids[i]: float(flows[i]["size"]) for i in range(len(ids))}
    rmax = {ids[i]: float(flows[i].get("rate_limit", 100.0)) for i in range(len(ids))}
    if delays is None:
        delays = [0.0 for _ in capacities]
    by_path = {j: [] for j in range(len(capacities))}
    active = {j: [] for j in range(len(capacities))}
    assigned = {}
    comp = {}
    t = 0.0
    pending = sorted(ids, key=lambda fid: arrivals[fid])
    idx = 0
    next_t = {j: {} for j in range(len(capacities))}
    link_free = {j: 0.0 for j in range(len(capacities))}
    while True:
        while idx < len(pending) and arrivals[pending[idx]] <= t:
            fid = pending[idx]
            best_j = None
            best_mk = None
            for j in range(len(capacities)):
                test_rem = {k: rem[k] for k in rem}
                test_active = list(active[j]) + [fid]
                cdict, mk = _simulate_path_completion_packet(test_active, test_rem, rmax, capacities[j], packet_size)
                if best_mk is None or mk < best_mk:
                    best_mk = mk
                    best_j = j
            assigned[fid] = best_j
            by_path[best_j].append(fid)
            if fid not in active[best_j]:
                active[best_j].append(fid)
            if fid not in next_t[best_j]:
                next_t[best_j][fid] = t
            idx += 1
        for j in range(len(capacities)):
            fids = [fid for fid in active[j] if rem[fid] > 0.0]
            rmax_list = [rmax[fid] for fid in fids]
            rates = _fair_share_rates(capacities[j], rmax_list) if fids else []
            rate_map = dict(zip(fids, rates))
            for fid in fids:
                if fid not in next_t[j]:
                    next_t[j][fid] = t
                if rate_map.get(fid, 0.0) > 0.0:
                    pass
        next_arrival = None
        if idx < len(pending):
            next_arrival = arrivals[pending[idx]] - t
        next_packet_events = []
        for j in range(len(capacities)):
            for fid, nt in next_t[j].items():
                if rem[fid] > 0.0 and fid in active[j]:
                    next_packet_events.append((j, fid, nt))
        if not next_packet_events and next_arrival is None:
            break
        candidates = []
        if next_arrival is not None and next_arrival > 0.0:
            candidates.append(("arrival", None, None, t + next_arrival))
        for j, fid, nt in next_packet_events:
            candidates.append(("packet", j, fid, nt))
        if not candidates:
            dt_type = None
        else:
            dt_type, j_sel, fid_sel, tevent = min(candidates, key=lambda x: x[3])
        if dt_type == "arrival":
            t = tevent
            continue
        if dt_type == "packet":
            fids = [fid for fid in active[j_sel] if rem[fid] > 0.0]
            rmax_list = [rmax[fid] for fid in fids]
            rates = _fair_share_rates(capacities[j_sel], rmax_list) if fids else []
            rate_map = dict(zip(fids, rates))
            start = max(tevent, link_free[j_sel])
            dur = packet_size / capacities[j_sel]
            finish = start + dur
            send_rate = rate_map.get(fid_sel, 0.0)
            if send_rate <= 0.0:
                next_t[j_sel][fid_sel] = finish
                t = finish
                continue
            rem[fid_sel] = max(0.0, rem[fid_sel] - packet_size)
            next_t[j_sel][fid_sel] = start + (packet_size / send_rate)
            link_free[j_sel] = finish
            t = finish
            new_active = []
            for fid in active[j_sel]:
                if rem[fid] <= 0.0 and fid not in comp:
                    comp[fid] = t + delays[j_sel]
                else:
                    new_active.append(fid)
            active[j_sel] = new_active
        else:
            break
    makespan = max(comp.values()) if comp else None
    return assigned, comp, makespan

def simulate_switch_packet_level_random(flows, capacities, delays=None, packet_size=1e-6, seed=42):
    import random
    random.seed(seed)
    ids = [f.get("id", i) for i, f in enumerate(flows)]
    arrivals = {ids[i]: float(flows[i]["arrival_time"]) for i in range(len(ids))}
    rem = {ids[i]: float(flows[i]["size"]) for i in range(len(ids))}
    rmax = {ids[i]: float(flows[i].get("rate_limit", 100.0)) for i in range(len(ids))}
    if delays is None:
        delays = [0.0 for _ in capacities]
    by_path = {j: [] for j in range(len(capacities))}
    active = {j: [] for j in range(len(capacities))}
    assigned = {}
    comp = {}
    t = 0.0
    pending = sorted(ids, key=lambda fid: arrivals[fid])
    idx = 0
    next_t = {j: {} for j in range(len(capacities))}
    link_free = {j: 0.0 for j in range(len(capacities))}
    while True:
        while idx < len(pending) and arrivals[pending[idx]] <= t:
            fid = pending[idx]
            j = random.randrange(len(capacities))
            assigned[fid] = j
            by_path[j].append(fid)
            if fid not in active[j]:
                active[j].append(fid)
            if fid not in next_t[j]:
                next_t[j][fid] = t
            idx += 1
        next_arrival = None
        if idx < len(pending):
            next_arrival = arrivals[pending[idx]] - t
        next_packet_events = []
        for j in range(len(capacities)):
            for fid, nt in next_t[j].items():
                if rem[fid] > 0.0 and fid in active[j]:
                    next_packet_events.append((j, fid, nt))
        if not next_packet_events and next_arrival is None:
            break
        candidates = []
        if next_arrival is not None and next_arrival > 0.0:
            candidates.append(("arrival", None, None, t + next_arrival))
        for j, fid, nt in next_packet_events:
            candidates.append(("packet", j, fid, nt))
        if not candidates:
            dt_type = None
        else:
            dt_type, j_sel, fid_sel, tevent = min(candidates, key=lambda x: x[3])
        if dt_type == "arrival":
            t = tevent
            continue
        if dt_type == "packet":
            fids = [fid for fid in active[j_sel] if rem[fid] > 0.0]
            rmax_list = [rmax[fid] for fid in fids]
            rates = _fair_share_rates(capacities[j_sel], rmax_list) if fids else []
            rate_map = dict(zip(fids, rates))
            start = max(tevent, link_free[j_sel])
            dur = packet_size / capacities[j_sel]
            finish = start + dur
            send_rate = rate_map.get(fid_sel, 0.0)
            if send_rate <= 0.0:
                next_t[j_sel][fid_sel] = finish
                t = finish
                continue
            rem[fid_sel] = max(0.0, rem[fid_sel] - packet_size)
            next_t[j_sel][fid_sel] = start + (packet_size / send_rate)
            link_free[j_sel] = finish
            t = finish
            new_active = []
            for fid in active[j_sel]:
                if rem[fid] <= 0.0 and fid not in comp:
                    comp[fid] = t + delays[j_sel]
                else:
                    new_active.append(fid)
            active[j_sel] = new_active
        else:
            break
    makespan = max(comp.values()) if comp else None
    return assigned, comp, makespan

def simulate_switch_packet_level_least_queue(flows, capacities, delays=None, packet_size=1e-6):
    ids = [f.get("id", i) for i, f in enumerate(flows)]
    arrivals = {ids[i]: float(flows[i]["arrival_time"]) for i in range(len(ids))}
    rem = {ids[i]: float(flows[i]["size"]) for i in range(len(ids))}
    rmax = {ids[i]: float(flows[i].get("rate_limit", 100.0)) for i in range(len(ids))}
    if delays is None:
        delays = [0.0 for _ in capacities]
    by_path = {j: [] for j in range(len(capacities))}
    active = {j: [] for j in range(len(capacities))}
    assigned = {}
    comp = {}
    t = 0.0
    pending = sorted(ids, key=lambda fid: arrivals[fid])
    idx = 0
    next_t = {j: {} for j in range(len(capacities))}
    link_free = {j: 0.0 for j in range(len(capacities))}
    while True:
        while idx < len(pending) and arrivals[pending[idx]] <= t:
            fid = pending[idx]
            qsizes = []
            for j in range(len(capacities)):
                inflight = packet_size if link_free[j] > t else 0.0
                qsize = (sum(rem[ff] for ff in active[j]) if active[j] else 0.0) + inflight
                qsizes.append((qsize, j))
            _, best_j = min(qsizes, key=lambda x: x[0])
            assigned[fid] = best_j
            by_path[best_j].append(fid)
            if fid not in active[best_j]:
                active[best_j].append(fid)
            if fid not in next_t[best_j]:
                next_t[best_j][fid] = t
            idx += 1
        next_arrival = None
        if idx < len(pending):
            next_arrival = arrivals[pending[idx]] - t
        next_packet_events = []
        for j in range(len(capacities)):
            for fid, nt in next_t[j].items():
                if rem[fid] > 0.0 and fid in active[j]:
                    next_packet_events.append((j, fid, nt))
        if not next_packet_events and next_arrival is None:
            break
        candidates = []
        if next_arrival is not None and next_arrival > 0.0:
            candidates.append(("arrival", None, None, t + next_arrival))
        for j, fid, nt in next_packet_events:
            candidates.append(("packet", j, fid, nt))
        if not candidates:
            dt_type = None
        else:
            dt_type, j_sel, fid_sel, tevent = min(candidates, key=lambda x: x[3])
        if dt_type == "arrival":
            t = tevent
            continue
        if dt_type == "packet":
            fids = [fid for fid in active[j_sel] if rem[fid] > 0.0]
            rmax_list = [rmax[fid] for fid in fids]
            rates = _fair_share_rates(capacities[j_sel], rmax_list) if fids else []
            rate_map = dict(zip(fids, rates))
            start = max(tevent, link_free[j_sel])
            dur = packet_size / capacities[j_sel]
            finish = start + dur
            send_rate = rate_map.get(fid_sel, 0.0)
            if send_rate <= 0.0:
                next_t[j_sel][fid_sel] = finish
                t = finish
                continue
            rem[fid_sel] = max(0.0, rem[fid_sel] - packet_size)
            next_t[j_sel][fid_sel] = start + (packet_size / send_rate)
            link_free[j_sel] = finish
            t = finish
            new_active = []
            for fid in active[j_sel]:
                if rem[fid] <= 0.0 and fid not in comp:
                    comp[fid] = t + delays[j_sel]
                else:
                    new_active.append(fid)
            active[j_sel] = new_active
        else:
            break
    makespan = max(comp.values()) if comp else None
    return assigned, comp, makespan

def simulate_packet_fixed_assignments(flows, capacities, delays=None, packet_size=1e-6, assignments=None):
    ids = [f.get("id", i) for i, f in enumerate(flows)]
    arrivals = {ids[i]: float(flows[i]["arrival_time"]) for i in range(len(ids))}
    sizes = {ids[i]: float(flows[i]["size"]) for i in range(len(ids))}
    rmax = {ids[i]: float(flows[i].get("rate_limit", 100.0)) for i in range(len(ids))}
    if delays is None:
        delays = [0.0 for _ in capacities]
    n_paths = len(capacities)
    assigned = {}
    comp = {}
    to_generate = {fid: sizes[fid] for fid in ids}
    remaining = {fid: sizes[fid] for fid in ids}
    next_in = {fid: arrivals[fid] for fid in ids}
    queues = {j: [] for j in range(n_paths)}
    backlog = {j: 0.0 for j in range(n_paths)}
    in_flight = {j: None for j in range(n_paths)}
    link_free = {j: 0.0 for j in range(n_paths)}
    t = min(arrivals.values()) if arrivals else 0.0
    while True:
        next_gen_time = None
        next_gen_fid = None
        for fid in ids:
            if to_generate[fid] > 0.0:
                ti = next_in.get(fid, None)
                if ti is not None and ti >= t:
                    if next_gen_time is None or ti < next_gen_time:
                        next_gen_time = ti
                        next_gen_fid = fid
        next_tx_time = None
        next_tx_link = None
        for j in range(n_paths):
            if in_flight[j] is not None:
                tf = link_free[j]
                if tf >= t:
                    if next_tx_time is None or tf < next_tx_time:
                        next_tx_time = tf
                        next_tx_link = j
        if next_gen_time is None and next_tx_time is None:
            break
        if next_tx_time is None or (next_gen_time is not None and next_gen_time < next_tx_time):
            t = next_gen_time
            fid = next_gen_fid
            j = assigned.get(fid, None)
            if j is None:
                if assignments and fid in assignments:
                    j = assignments[fid]
                else:
                    j = 0
                assigned[fid] = j
            if queues[j] and queues[j][-1][0] == fid:
                f_old, cnt_old = queues[j][-1]
                queues[j][-1] = (f_old, cnt_old + 1)
            else:
                queues[j].append((fid, 1))
            backlog[j] += packet_size
            to_generate[fid] = max(0.0, to_generate[fid] - packet_size)
            if to_generate[fid] > 0.0:
                next_in[fid] = t + packet_size / rmax[fid]
            else:
                next_in[fid] = None
            if in_flight[j] is None and link_free[j] <= t and backlog[j] >= packet_size:
                head_fid, head_cnt = queues[j][0]
                in_flight[j] = head_fid
                link_free[j] = t + packet_size / capacities[j]
                backlog[j] -= packet_size
                if head_cnt == 1:
                    queues[j].pop(0)
                else:
                    queues[j][0] = (head_fid, head_cnt - 1)
        else:
            t = next_tx_time
            j = next_tx_link
            fid = in_flight[j]
            in_flight[j] = None
            remaining[fid] = max(0.0, remaining[fid] - packet_size)
            if remaining[fid] <= 0.0 and fid not in comp:
                comp[fid] = t + delays[j]
            if queues[j]:
                head_fid, head_cnt = queues[j][0]
                in_flight[j] = head_fid
                link_free[j] = t + packet_size / capacities[j]
                backlog[j] -= packet_size
                if head_cnt == 1:
                    queues[j].pop(0)
                else:
                    queues[j][0] = (head_fid, head_cnt - 1)
    makespan = max(comp.values()) if comp else None
    return comp, makespan

def main():
    import numpy as np
    flows_counts = list(range(3, 13))
    rows = []
    for k in flows_counts:
        rng = np.random.RandomState(42)
        random_ids = list(100 + rng.choice(5000, size=k, replace=False))
        ext_arrivals = list(rng.uniform(0.0, 0.001, size=k))
        arrival_map = {random_ids[i]: ext_arrivals[i] for i in range(k)}
        flows, shared_paths, flow_path_map, H, dlt = build_instance(
            n_inputs=1,
            m_outputs=2,
            flows_per_input=[k],
            size_bits=0.064,
            rate_limit=25.0,
            capacities=[6.5, 15.6],
            delays=[0.0, 0.0],
            delta=None,
            horizon=None,
            input_rate_limits=[25.0],
            flow_ids=random_ids,
            flow_arrival_map=arrival_map,
            desired_slots_per_flow=50,
            max_slots=300
        )
        sol = solve_time_indexed_milp(
            flows, shared_paths, flow_path_map,
            time_horizon=H,
            time_slot_duration=dlt,
            use_fairness=True,
            time_limit=60
        )
        milp_assign = sol.get('assignments')
        milp_comp, milp_mk = simulate_packet_fixed_assignments(
            flows,
            [p['capacity'] for p in shared_paths],
            [p['delay'] for p in shared_paths],
            assignments=milp_assign
        )
        rand_assign, _, _ = simulate_switch_packet_level_random(
            flows,
            [p['capacity'] for p in shared_paths],
            [p['delay'] for p in shared_paths]
        )
        fifo_assign, _, _ = simulate_switch_packet_level_least_queue(
            flows,
            [p['capacity'] for p in shared_paths],
            [p['delay'] for p in shared_paths]
        )
        rand_comp, rand_mk = simulate_packet_fixed_assignments(
            flows,
            [p['capacity'] for p in shared_paths],
            [p['delay'] for p in shared_paths],
            assignments=rand_assign
        )
        fifo_comp, fifo_mk = simulate_packet_fixed_assignments(
            flows,
            [p['capacity'] for p in shared_paths],
            [p['delay'] for p in shared_paths],
            assignments=fifo_assign
        )
        rows.append(
            {
                "Flows": k,
                "MILP": milp_mk,
                "Random": rand_mk,
                "LeastQueue": fifo_mk,
            }
        )

    headers = ["Flows", "MILP", "Random", "LeastQueue"]
    formatted = []
    for r in rows:
        formatted.append(
            {
                "Flows": str(r["Flows"]),
                "MILP": f"{r['MILP']:.6f}" if r["MILP"] is not None else "nan",
                "Random": f"{r['Random']:.6f}" if r["Random"] is not None else "nan",
                "LeastQueue": f"{r['LeastQueue']:.6f}" if r["LeastQueue"] is not None else "nan",
            }
        )
    widths = {h: len(h) for h in headers}
    for r in formatted:
        for h in headers:
            widths[h] = max(widths[h], len(r[h]))
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print("\nResults:")
    print(line)
    print(sep)
    for r in formatted:
        print(" | ".join(r[h].ljust(widths[h]) for h in headers))

if __name__ == "__main__":
    main()

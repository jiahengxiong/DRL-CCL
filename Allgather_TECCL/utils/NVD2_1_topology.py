from decimal import Decimal

import matplotlib.pyplot as plt
import networkx as nx

import config
from generate_topo import gen_topo


# 模拟 topologies.topology 模块中的 Topology 类
class Topology:
    def __init__(self):
        self.chunk_size = 1  # 假设 chunk_size 为 1
        self.chassis = config.chassis


class NDv2(Topology):
    def __init__(self):
        super().__init__()
        self.node_per_chassis = 8
        self.construct_topology()

    def construct_topology(self):
        chassis = self.chassis

        # 定义 conversion_map，确保没有重复赋值
        conversion_map = {}
        conversion_map[0] = 0
        conversion_map[23] = 50 / self.chunk_size  # 使用最终赋值
        conversion_map[46] = 50 / (2 * self.chunk_size)
        conversion_map[107] = 12.5 / self.chunk_size

        self.switch_indices = [0, 1, 2, 3]
        if chassis == 2:
            # connectivity: 0.5, GPU: 6
            if config.connectivity == "test":
                import simplejson as json
                topology_path = '/Users/xiongjiaheng/RDMA/CCL/network_topology.json'# ===== 读文件 =====
                with open(topology_path, "r") as f:
                    data = json.load(f, parse_float=Decimal, parse_int=Decimal)

                capacity = data["capacity"]
                self.pro = data["propagation"]
                self.TECCL_pro = data["float_propagation"]
            if config.connectivity == 0.5:
                capacity = [[0, 0, 107, 107, 107, 0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 107, 107, 107, 107, 0, 0, 0, 107, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 107, 107, 0, 107],
                            [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 0, 0, 107],
                            [107, 107, 0, 0, 0, 46, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 107, 0, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 46, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0],
                            [107, 0, 0, 0, 0, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0],
                            [107, 0, 0, 0, 46, 0, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0],
                            [0, 107, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 0, 0],
                            [0, 0, 0, 107, 0, 0, 0, 0, 0, 0, 46, 0, 46, 0, 0, 46],
                            [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46, 0, 0],
                            [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46],
                            [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46, 0]]
                self.pro = [[-1, -1, Decimal('5E-5'), Decimal('5E-5'), Decimal('7E-7'), -1, -1, Decimal('7E-7'),
                             Decimal('7E-7'), -1,
                             -1, -1, -1, -1, -1, -1],
                            [-1, -1, Decimal('5E-5'), Decimal('5E-5'), Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1,
                             Decimal('7E-7'),
                             -1, -1, -1, -1, -1, -1],
                            [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                             Decimal('7E-7'),
                             Decimal('7E-7'), -1, Decimal('7E-7')],
                            [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'),
                             -1, -1, -1,
                             Decimal('7E-7')],
                            [Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), -1,
                             Decimal('7E-7'), -1,
                             -1, -1, -1, -1, -1, -1],
                            [-1, Decimal('7E-7'), -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1,
                             -1, -1, -1],
                            [-1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1,
                             -1, -1, -1],
                            [Decimal('7E-7'), -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1,
                             -1, -1, -1],
                            [Decimal('7E-7'), -1, -1, -1, Decimal('7E-7'), -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'),
                             -1, -1, -1,
                             -1, -1, -1],
                            [-1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, -1, -1, -1],
                            [-1, -1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'),
                             -1, -1,
                             Decimal('7E-7')],
                            [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                             Decimal('7E-7'), -1, -1],
                            [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                             Decimal('7E-7'),
                             Decimal('7E-7')],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7')],
                            [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                             Decimal('7E-7'), Decimal('7E-7'), -1]]
                self.TECCL_pro = [[-1, -1, 5E-5, 5E-5, 7e-07, -1, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, 5E-5, 5E-5, 7e-07, 7e-07, -1, -1, -1, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [5E-5, 5E-5, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, -1, 7e-07],
                                  [5E-5, 5E-5, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, -1, -1, 7e-07],
                                  [7e-07, 7e-07, -1, -1, -1, 7e-07, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, 7e-07, -1, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, 7e-07, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [7e-07, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [7e-07, -1, -1, -1, 7e-07, -1, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [-1, 7e-07, -1, -1, -1, -1, -1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, -1, -1, -1],
                                  [-1, -1, -1, 7e-07, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, -1, 7e-07],
                                  [-1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, -1],
                                  [-1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07],
                                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07],
                                  [-1, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, -1]]
            elif config.connectivity == 0.7:

                # connectivity = 0.7, GPU:6
                capacity = [[0, 0, 107, 107, 107, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 107, 107, 0, 107, 107, 107, 107, 107, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 107, 0, 107, 107],
                            [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 107, 0, 107],
                            [107, 0, 0, 0, 0, 46, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0],
                            [0, 107, 0, 0, 46, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 0, 46, 0, 46, 46, 46, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 46, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 107, 0, 0, 0, 46, 46, 46, 0, 46, 0, 0, 0, 0, 0, 0],
                            [0, 107, 0, 0, 46, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 46, 0],
                            [0, 0, 0, 107, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46, 46, 46],
                            [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46, 0],
                            [0, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 46, 46, 0, 46, 46],
                            [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 46, 46, 46, 46, 0, 46],
                            [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46, 0]]
                self.pro = [
                    [-1, -1, Decimal('5E-5'), Decimal('5E-5'), Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'),
                     -1, -1,
                     -1, -1, -1, -1, -1, -1],
                    [-1, -1, Decimal('5E-5'), Decimal('5E-5'), -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'),
                     Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                     Decimal('7E-7'),
                     -1, Decimal('7E-7'), Decimal('7E-7')],
                    [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                     Decimal('7E-7'), -1, Decimal('7E-7')],
                    [Decimal('7E-7'), -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1,
                     -1,
                     -1, -1, -1],
                    [-1, Decimal('7E-7'), -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1,
                     -1,
                     -1, -1, -1],
                    [Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'),
                     Decimal('7E-7'),
                     Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [Decimal('7E-7'), Decimal('7E-7'), -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1,
                     Decimal('7E-7'), -1,
                     -1, -1, -1, -1, -1, -1],
                    [-1, Decimal('7E-7'), -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1,
                     Decimal('7E-7'),
                     -1, -1, -1, -1, -1, -1],
                    [-1, Decimal('7E-7'), -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1,
                     -1,
                     -1, -1, -1],
                    [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, -1, Decimal('7E-7'),
                     -1],
                    [-1, -1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'),
                     Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7')],
                    [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'),
                     Decimal('7E-7'), -1],
                    [-1, -1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), -1,
                     Decimal('7E-7'), Decimal('7E-7')],
                    [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'),
                     Decimal('7E-7'),
                     Decimal('7E-7'), -1, Decimal('7E-7')],
                    [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                     Decimal('7E-7'), Decimal('7E-7'), -1]]
                self.TECCL_pro = [[-1, -1, 5E-5, 5E-5, 7e-07, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, 5E-5, 5E-5, -1, 7e-07, 7e-07, 7e-07, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [5E-5, 5E-5, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, 7e-07, 7e-07],
                                  [5E-5, 5E-5, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, 7e-07],
                                  [7e-07, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [-1, 7e-07, -1, -1, 7e-07, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [7e-07, 7e-07, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [7e-07, 7e-07, -1, -1, 7e-07, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, 7e-07, -1, -1, -1, 7e-07, 7e-07, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [-1, 7e-07, -1, -1, 7e-07, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, -1, 7e-07, -1],
                                  [-1, -1, -1, 7e-07, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, 7e-07, 7e-07],
                                  [-1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, -1],
                                  [-1, -1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, -1, 7e-07, 7e-07],
                                  [-1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, 7e-07, 7e-07, -1, 7e-07],
                                  [-1, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, -1]]


            # # connectivity = 0.9, GPU = 6
            elif config.connectivity == 0.9:
                capacity = [[0, 0, 107, 107, 107, 107, 107, 0, 107, 107, 0, 0, 0, 0, 0, 0],
                            [0, 0, 107, 107, 0, 107, 107, 107, 107, 107, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 107, 107, 107, 107, 107, 107],
                            [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 107, 107, 107, 0, 107, 107],
                            [107, 0, 0, 0, 0, 46, 0, 46, 46, 46, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 46, 0, 46, 46, 46, 46, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 0, 46, 0, 46, 46, 46, 0, 0, 0, 0, 0, 0],
                            [0, 107, 0, 0, 46, 46, 46, 0, 46, 46, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 46, 46, 46, 46, 0, 46, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 46, 46, 46, 46, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0, 46, 46, 46, 46, 0],
                            [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46, 0, 46],
                            [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 46, 46, 0, 46, 46, 46],
                            [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 46, 46, 46, 0, 46, 46],
                            [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46, 0, 46],
                            [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0, 46, 46, 46, 46, 0]]
                self.pro = [
                    [-1, -1, Decimal('5E-5'), Decimal('5E-5'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1,
                     Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [-1, -1, Decimal('5E-5'), Decimal('5E-5'), -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'),
                     Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'),
                     Decimal('7E-7'),
                     Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7')],
                    [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'),
                     Decimal('7E-7'),
                     Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7')],
                    [Decimal('7E-7'), -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'),
                     Decimal('7E-7'),
                     -1, -1, -1, -1, -1, -1],
                    [Decimal('7E-7'), Decimal('7E-7'), -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'),
                     Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'),
                     Decimal('7E-7'),
                     Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [-1, Decimal('7E-7'), -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1,
                     Decimal('7E-7'),
                     Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [Decimal('7E-7'), Decimal('7E-7'), -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'),
                     Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [Decimal('7E-7'), Decimal('7E-7'), -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'),
                     Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'),
                     Decimal('7E-7'),
                     Decimal('7E-7'), Decimal('7E-7'), -1],
                    [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                     Decimal('7E-7'),
                     Decimal('7E-7'), -1, Decimal('7E-7')],
                    [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'),
                     -1,
                     Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7')],
                    [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'),
                     Decimal('7E-7'),
                     -1, Decimal('7E-7'), Decimal('7E-7')],
                    [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1,
                     Decimal('7E-7'),
                     Decimal('7E-7'), -1, Decimal('7E-7')],
                    [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'),
                     Decimal('7E-7'),
                     Decimal('7E-7'), Decimal('7E-7'), -1]]
                self.TECCL_pro = [[-1, -1, 5E-5, 5E-5, 7e-07, 7e-07, 7e-07, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, 5E-5, 5E-5, -1, 7e-07, 7e-07, 7e-07, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [5E-5, 5E-5, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, 7e-07, 7e-07, 7e-07,
                                   7e-07],
                                  [5E-5, 5E-5, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, 7e-07, -1, 7e-07, 7e-07],
                                  [7e-07, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [7e-07, 7e-07, -1, -1, 7e-07, -1, 7e-07, 7e-07, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [7e-07, 7e-07, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [-1, 7e-07, -1, -1, 7e-07, 7e-07, 7e-07, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [7e-07, 7e-07, -1, -1, 7e-07, 7e-07, 7e-07, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [7e-07, 7e-07, -1, -1, 7e-07, 7e-07, 7e-07, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, 7e-07, 7e-07, -1],
                                  [-1, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, -1, 7e-07],
                                  [-1, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, -1, 7e-07, 7e-07, 7e-07],
                                  [-1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, 7e-07, -1, 7e-07, 7e-07],
                                  [-1, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, 7e-07, -1, 7e-07],
                                  [-1, -1, 7e-07, 7e-07, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, 7e-07, 7e-07, -1]]
            elif config.connectivity == 0.3:
                capacity = [[0, 0, 107, 107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 107, 107, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 107, 0, 0, 0, 0, 0],
                            [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 107],
                            [107, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 46, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0],
                            [0, 107, 0, 0, 0, 0, 46, 0, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 0, 46, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 46],
                            [0, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0]]
                self.pro = [
                    [-1, -1, Decimal('5E-5'), Decimal('5E-5'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1,
                     -1],
                    [-1, -1, Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, Decimal('7E-7'), -1, -1, -1, -1,
                     -1,
                     -1],
                    [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, -1, -1,
                     -1,
                     -1], [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                           Decimal('7E-7')],
                    [Decimal('7E-7'), -1, -1, -1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1,
                     -1], [-1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1],
                    [-1, Decimal('7E-7'), -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1,
                     -1],
                    [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1,
                     -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7'), -1,
                     -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7')],
                    [-1, -1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1]]
                self.TECCL_pro = [[-1, -1, 5E-5, 5E-5, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, 5E-5, 5E-5, -1, -1, -1, -1, -1, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [5E-5, 5E-5, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, -1, -1, -1, -1],
                                  [5E-5, 5E-5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07],
                                  [7e-07, -1, -1, -1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1],
                                  [-1, 7e-07, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, -1, -1, -1],
                                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1, -1, -1],
                                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, 7e-07, -1, 7e-07, -1, -1],
                                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07, -1],
                                  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1, 7e-07],
                                  [-1, -1, -1, 7e-07, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7e-07, -1]]

            # self.pro = [[-1, -1, Decimal('5E-5'), Decimal('5E-5'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1], [-1, -1, Decimal('5E-5'), Decimal('5E-5'), -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1], [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7')], [Decimal('5E-5'), Decimal('5E-5'), -1, -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7')], [Decimal('7E-7'), -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1], [Decimal('7E-7'), Decimal('7E-7'), -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1], [Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1], [-1, Decimal('7E-7'), -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1], [Decimal('7E-7'), Decimal('7E-7'), -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1], [Decimal('7E-7'), Decimal('7E-7'), -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1], [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1], [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7')], [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7')], [-1, -1, Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7')], [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, Decimal('7E-7'), -1, Decimal('7E-7'), Decimal('7E-7'), -1, Decimal('7E-7')], [-1, -1, Decimal('7E-7'), Decimal('7E-7'), -1, -1, -1, -1, -1, -1, -1, Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), Decimal('7E-7'), -1]]
            # capacity = [[0, 0, 107, 107, 107, 107, 107, 0, 107, 107, 0, 0, 0, 0, 0, 0], [0, 0, 107, 107, 0, 107, 107, 107, 107, 107, 0, 0, 0, 0, 0, 0], [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 107, 107, 107, 107, 107, 107], [107, 107, 0, 0, 0, 0, 0, 0, 0, 0, 107, 107, 107, 0, 107, 107], [107, 0, 0, 0, 0, 46, 0, 46, 46, 46, 0, 0, 0, 0, 0, 0], [107, 107, 0, 0, 46, 0, 46, 46, 46, 46, 0, 0, 0, 0, 0, 0], [107, 107, 0, 0, 0, 46, 0, 46, 46, 46, 0, 0, 0, 0, 0, 0], [0, 107, 0, 0, 46, 46, 46, 0, 46, 46, 0, 0, 0, 0, 0, 0], [107, 107, 0, 0, 46, 46, 46, 46, 0, 46, 0, 0, 0, 0, 0, 0], [107, 107, 0, 0, 46, 46, 46, 46, 46, 0, 0, 0, 0, 0, 0, 0], [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0, 46, 46, 46, 46, 0], [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46, 0, 46], [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 46, 46, 0, 46, 46, 46], [0, 0, 107, 0, 0, 0, 0, 0, 0, 0, 46, 46, 46, 0, 46, 46], [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 46, 0, 46, 46, 0, 46], [0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0, 46, 46, 46, 46, 0]]







        else:
            self.switch_indices = [0]
            single_capacity = [
                [0, 23, 46, 46, 23, 0, 0, 0],
                [23, 0, 46, 23, 0, 46, 0, 0],
                [46, 46, 0, 23, 0, 0, 23, 0],
                [46, 23, 23, 0, 0, 0, 0, 46],
                [23, 0, 0, 0, 0, 23, 46, 46],
                [0, 46, 0, 0, 23, 0, 46, 23],
                [0, 0, 23, 0, 46, 46, 0, 23],
                [0, 0, 0, 46, 46, 23, 23, 0]
            ]
            capacity = [[0] * (8 * chassis + 1)]
            for i in range(chassis):
                for j in single_capacity:
                    cap = [0] * (8 * chassis)
                    for k in range(8):
                        cap[8 * i + k] = j[k]
                    capacity.append([0] + cap)
            for i in range(chassis):
                capacity[0][i * 8 + 1] = 107
                capacity[i * 8 + 2][0] = 107
        capacity = list(map(list, zip(*capacity)))
        self.capacity = [list(map(lambda x: conversion_map[x], r))
                         for r in capacity]
        self.topology = [list(map(lambda x: int(x > 0), r))
                         for r in self.capacity]
        # print("Capacity:", self.capacity)
        # print("Topology:", self.topology)
        self.alpha = []
        for r in capacity:
            row = []
            for i in r:
                if i:
                    if i == 107:
                        row.append(0.0 * pow(10, -6))
                    else:
                        row.append(0.0 * pow(10, -6))
                else:
                    row.append(-1)
            self.alpha.append(row)
        # print("Alpha:", self.alpha)


class NVD2_1_topology(NDv2):
    def __init__(self, packet_size, num_chunk):
        super().__init__()
        # print("Initializing NVD2_1_topology")
        self.capacity = [
            [Decimal(str(x)) for x in row] for row in self.capacity
        ]  # 转化为高精度
        self.nodes = list(range(len(self.capacity)))
        self.packet_size = Decimal(str(packet_size)) / Decimal(str(num_chunk))
        self.num_chunk = num_chunk
        self.num_gpu = len(self.capacity) - 4

        self.topology = self.get_topology()
        # for node in self.topology.nodes:
        #     print(node, self.topology.nodes[node]['memory'])

    def get_topology(self):
        G = nx.DiGraph()  # 使用有向图
        DC_1 = [0, 1]
        DC_2 = [2, 3]
        # print(self.num_gpu / 2)
        DC_1_GPU = list(range(4, 4 + int(self.num_gpu / 2)))
        DC_2_GPU = list(range(4 + self.num_gpu, self.num_gpu + 4))
        self.DC_1 = DC_1_GPU + DC_1
        self.DC_2 = DC_2_GPU + DC_2

        for node in self.nodes:
            if node in DC_1_GPU + DC_1:
                DC = 0
            else:
                DC = 1
            if node not in self.switch_indices:
                G.add_node(node, memory=self.initial_buffer(node, DC), type='GPU', job={}, added_job={}, policy=[],
                           DC=DC, receive_buffer=[])
            else:
                G.add_node(node, memory=self.initial_buffer(node, DC), type='switch', job={}, added_job={}, policy=[],
                           in_queue=[], out_queue=[], DC=DC, receive_buffer=[], buffer_limitation=0, right=0, left=0)
        for i in range(len(self.capacity)):
            for j in range(len(self.capacity[i])):
                if self.capacity[i][j] == 25 or self.capacity[i][j] == 50:  # 如果有链路
                    propagation_latency = self.pro[self.nodes[i]][self.nodes[j]]
                    transmission_latency = self.packet_size / self.capacity[i][j]
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency=self.pro[self.nodes[i]][self.nodes[j]],  # Decimal(0.7e-7),
                               transmission_latency=self.packet_size / self.capacity[i][j],
                               state='free',
                               job=[],
                               type='NVlink',
                               weight=propagation_latency + transmission_latency,
                               num_chunk = 0,
                               connect=False)
                if self.capacity[i][j] == 12.5:
                    propagation_latency = self.pro[self.nodes[i]][self.nodes[j]]
                    transmission_latency = self.packet_size / self.capacity[i][j]
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency=self.pro[self.nodes[i]][self.nodes[j]],
                               # Decimal(1.3e-7),self.pro[self.nodes[i]][self.nodes[j]],
                               transmission_latency=self.packet_size / self.capacity[i][j],
                               state='free',
                               job=[],
                               type='Switch',
                               weight=propagation_latency + transmission_latency,
                               num_chunk = 0,
                               connect=True)

                # 新增功能：根据 self.chassis 删除节点 0 并调整其他节点编号
        if self.chassis == 1:
            if 0 in G:
                G.remove_node(0)
            mapping = {old_node: old_node - 1 for old_node in list(G.nodes)}
            G = nx.relabel_nodes(G, mapping, copy=True)
        # if self.chassis == 2:
        #     G.nodes[0]['type'] = 'switch'
        #     for node in list(G.nodes):
        #         G.nodes[node]['memory'].pop(0, None)
        for node in G.nodes:
            for next_node in list(G.successors(node)):
                G.nodes[node]['job'][(node, next_node)] = []
                G.nodes[node]['added_job'][(node, next_node)] = []
                G.nodes[node][f'sender to {next_node}'] = 'free'
                G.nodes[node][f'sender to {next_node} job'] = None
            for pre_node in list(G.predecessors(node)):
                G.nodes[node][f'receiver from {pre_node}'] = 'free'

        return G

    def initial_buffer(self, current_node, DC):
        buffer = {}
        if config.collective == 'ALLGATHER':
            config.buffer_constant = 1
        else:
            config.buffer_constant = self.num_gpu
        for node in range(self.num_gpu * self.num_chunk * config.buffer_constant):
            buffer[node] = {'buffer': None, 'send_time': None, 'received_time': None, 'DC': DC}
        if self.chassis == 2:
            switch_num = 0
            self.switch_indices = [0, 1, 2, 3]
        else:
            switch_num = 1
            self.switch_indices = [0, 1, 2, 3]

        switch_num = 4
        if current_node not in self.switch_indices:
            for i in range(self.num_chunk * config.buffer_constant):
                buffer[(current_node - switch_num) * self.num_chunk * config.buffer_constant + i] = {
                    'buffer': (current_node - switch_num) * self.num_chunk * config.buffer_constant + i, 'send_time': 0,
                    'received_time': 0}
        return buffer


if __name__ == '__main__':
    config.packet_size = Decimal(str(0.003906252))
    config.num_chunk = 1
    config.chassis = 2
    config.collective = 'ALLGATHER'
    config.topology_name = 'NVD2'
    config.connect_matrix = []
    config.connectivity = 0.3
    topo = NVD2_1_topology(num_chunk=4, packet_size=1)
    G = topo.topology
    for node in G.nodes:
        print(node, G.nodes[node]['memory'])
    for node in G.nodes:
        if G.nodes[node]['type'] == 'switch':
            print(list(G.successors(node)), list(G.predecessors(node)))

    # 设置图形布局
    pos = nx.spring_layout(G)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrows=True)  # 添加箭头表示方向

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # 绘制边的权重
    edge_labels = nx.get_edge_attributes(G, 'link_capcapacity')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # 显示图形
    plt.title("Directed Topology Visualization")
    plt.axis('off')
    plt.show()

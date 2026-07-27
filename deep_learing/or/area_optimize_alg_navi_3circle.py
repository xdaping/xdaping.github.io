#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2026/6/26 16:44

# -*- coding: utf-8 -*-

"""
@software: PyCharm
@file: area_optimize_alg_navi_3circle.py
@time: 2023/7/18 10:06
"""

import time
from ortools.linear_solver import pywraplp

from utils import logging_util


class OptimizeAlg(object):

    def __init__(self, data):

        self.data = data
        self.treatment = [0, 1]

        self.i_range = range(len(self.data))
        self.j_range = range(len(self.treatment))

        # 预存每个AOI-poi订单信息
        self.aoi_all_poi_idx = {}
        # 每个aoi的基础商家列表长度
        self.aoi_poi_len = {}
        # 城市粒度基础流向数量
        self.city_base_len = 0

        # 创建求解器
        self.solver = pywraplp.Solver.CreateSolver('SCIP')

        # 预处理aoi-poi列表长度
        self._pre_compute()

        # 决策变量占位符
        self.max_flag = 10000000

        # 决策变量
        self.v = {}
        self._init_var()

        # 目标函数
        self._set_obj()

        # 约束条件
        self._add_constraint()

        # 求解结果
        self.status = None

    def _pre_compute(self):
        """
        提前计算好的or求解需要的aoi-poi列表数据
        :return:
        """
        # for i in self.i_range:
        #     if self.data[i]['aoi_id'] not in self.aoi_base_idx and self.data[i]['base_set'] == 1:
        #         self.aoi_base_idx[self.data[i]['aoi_id']] = [i]
        #     elif self.data[i]['aoi_id'] in self.aoi_base_idx and self.data[i]['base_set'] == 1:
        #         self.aoi_base_idx[self.data[i]['aoi_id']].append(i)
        # 给每个aoi建立一个poi-list
        for i in self.i_range:
            if self.data[i]['aoi_id'] not in self.aoi_all_poi_idx:
                self.aoi_all_poi_idx[self.data[i]['aoi_id']] = [i]
            else:
                self.aoi_all_poi_idx[self.data[i]['aoi_id']].append(i)
        # 给每个aoi存基础商家列表长度
        for i in self.i_range:
            if self.data[i]['aoi_id'] not in self.aoi_poi_len and self.data[i]['base_set'] == 1:
                self.aoi_poi_len[self.data[i]['aoi_id']] = 1
            elif self.data[i]['aoi_id'] in self.aoi_poi_len and self.data[i]['base_set'] == 1:
                self.aoi_poi_len[self.data[i]['aoi_id']] += 1
        # 存取城市粒度基础流向数量
        for i in self.i_range:
            if self.data[i]['base_set'] == 1:
                self.city_base_len += 1

    def _init_var(self):
        """
        设置决策变量
        :return:
        """
        # for i in self.i_range:
        #     for j in self.j_range:
        #         self.v[i, j] = self.solver.BoolVar(f"v_{(i * len(self.treatment) + j + self.max_flag)}")
        for i in self.i_range:
            # 基础集合固定都选，不需要决策变量
            if self.data[i]['base_set'] == 1 and self.data[i]['candidate_set'] == 0 and self.data[i]['navi_cls'] <= 2:
                self.v[i, 0] = 0
                self.v[i, 1] = 1
            # 不在基础集合和候选集合固定都不选，不需要决策变量
            elif self.data[i]['base_set'] == 0 and self.data[i]['candidate_set'] == 0 and self.data[i]['navi_cls'] > 3:
                self.v[i, 0] = 0
                self.v[i, 1] = 0
            # elif self.data[i]['navi_cls'] > 3:
            #     self.v[i, 0] = 0
            #     self.v[i, 1] = 0
            else:
                for j in self.j_range:
                    self.v[i, j] = self.solver.BoolVar(f"v_{(i * len(self.treatment) + j + self.max_flag)}")

    def _set_obj(self):
        """
        设定目标函数
        :return:
        """
        self.solver.Minimize(
            sum([sum([self.data[i]['prediction_map']['navi_distance'][j] * self.v[i, j] for j in self.j_range]) for i
                 in self.i_range]))

    def _add_constraint(self):
        """
        添加约束
        :return:
        """

        # 每个流向约束
        for i in self.i_range:
            # # 只有一个treatment（选或者不选）
            # self.solver.Add(sum([self.v[i, j] for j in self.j_range]) == 1)
            # # 只属于基础集合的商家必选
            # if self.data[i]['base_set'] == 1 and self.data[i]['candidate_set'] == 0:
            #     for j in self.j_range:
            #         if self.treatment[j] > 0:
            #             self.solver.Add(self.v[i, j] == 1)
            # 除了内圈和最外圈以外只有一个treatment（选或者不选）
            if not ((self.data[i]['base_set'] == 1 and self.data[i]['candidate_set'] == 0 and self.data[i]['navi_cls'] <= 2) or
                    (self.data[i]['base_set'] == 0 and self.data[i]['candidate_set'] == 0 and self.data[i]['navi_cls'] > 3)):
                self.solver.Add(sum([self.v[i, j] for j in self.j_range]) == 1)
            # 只属于基础集合的商家必选
            # if self.data[i]['base_set'] == 1 and self.data[i]['candidate_set'] == 0:
            #     for j in self.j_range:
            #         if self.treatment[j] > 0:
            #             self.solver.Add(self.v[i, j] == 1)

        # 单量约束
        order_threshold = -0.005
        origin_order_cnt = sum(
            [self.data[i]['prediction_map']['order'][1] if self.data[i]['base_set'] == 1 else 0.0 for i in self.i_range])
        adjust_order_cnt = sum(
            [sum([self.data[i]['prediction_map']['order'][j] * self.v[i, j] for j in self.j_range]) for i in
             self.i_range])
        order_change_rate = 1 - adjust_order_cnt / origin_order_cnt
        # 小于等于增加的阈值
        # self.solver.Add(order_change_rate >= -order_threshold)
        # 大于等于下降的阈值
        self.solver.Add(order_change_rate <= order_threshold)

        # 候选集合更改数量限制
        # aoi_d = {}
        # poi_adjust_threshold = 0.8
        # for i in self.i_range:
        #     if self.data[i]['candidate_set'] == 1:
        #         if self.data[i]['aoi_id'] not in aoi_d:
        #             aoi_d[self.data[i]['aoi_id']] = [i]
        #         else:
        #             aoi_d[self.data[i]['aoi_id']].append(i)
        # for aoi_id, idx_list in aoi_d.items():
        #     adj_len = int(len(idx_list) * poi_adjust_threshold)
        #     self.solver.Add(sum([self.v[idx, 1] for idx in idx_list]) <= adj_len)

        # 商家列表更改数量限制aoi粒度
        # poi_min_len_threshold = 0.90
        # poi_max_len_threshold = 1.10
        # for aoi_id, idx_list in self.aoi_all_poi_idx.items():
        #     if aoi_id in self.aoi_poi_len.keys():
        #         adj_min_len = round(self.aoi_poi_len[aoi_id] * poi_min_len_threshold)
        #         adj_max_len = round(self.aoi_poi_len[aoi_id] * poi_max_len_threshold)
        #         self.solver.Add(sum([self.v[idx, 1] for idx in idx_list]) <= adj_max_len)
        #         self.solver.Add(sum([self.v[idx, 1] for idx in idx_list]) >= adj_min_len)

        # 商家列表更改数量限制城市粒度
        city_min_len_threshold = 0.95
        city_max_len_threshold = 1.10
        adj_city_len = 0
        city_adj_min_len = round(self.city_base_len * city_min_len_threshold)
        city_adj_max_len = round(self.city_base_len * city_max_len_threshold)
        if self.city_base_len > 0:
            for i in self.i_range:
                adj_city_len += self.v[i, 1]
            # adj_city_len = sum([self.v[i, 1] for i in self.i_range])
            self.solver.Add(adj_city_len <= city_adj_max_len)
            self.solver.Add(adj_city_len >= city_adj_min_len)
        # for aoi_id, idx_list in self.aoi_all_poi_idx.items():
        #     if aoi_id in self.aoi_poi_len.keys():
        #         adj_min_len = round(self.aoi_poi_len[aoi_id] * poi_min_len_threshold)
        #         adj_max_len = round(self.aoi_poi_len[aoi_id] * poi_max_len_threshold)
        #         self.solver.Add(sum([self.v[idx, 1] for idx in idx_list]) <= adj_max_len)
        #         self.solver.Add(sum([self.v[idx, 1] for idx in idx_list]) >= adj_min_len)

    def solve(self, max_time_limit=120):
        """
        :param max_time_limit: 最长求解时间，单位分钟
        :return:
        """
        self.solver.SetTimeLimit(int(max_time_limit * 60 * 1000))  # 最长x分钟，求不出最优解可行解也行。
        self.status = self.solver.Solve()
        return self.status

    def get_adjust_treatment(self):
        """
        根据求解结果，返回每个流向的选择情况
        :return:
        """
        # return [round(sum([self.v[i, j].solution_value() * self.treatment[j] for j in self.j_range])) for i in self.i_range]
        return [round(self.v[i, 1] if type(self.v[i, 0]) == int else sum([self.v[i, j].solution_value() * self.treatment[j] for j in self.j_range])) for i in self.i_range]


class OptimizeMapPartitionCircle(object):
    @staticmethod
    def scip_optimize(partition_data):

        data = [item for item in partition_data]
        cur_logger = logging_util.Logger()

        start_time = time.time()
        cur_logger.info(f"正在求解区域{data[0]['city_id']},流向个数为：{len(data)}")

        optimize_alg = OptimizeAlg(data)
        status = optimize_alg.solve()

        cur_logger.info(f"当前区域{data[0]['city_id']} 求解完成, status：{status}, 流向个数：{len(data)},"
                        f"耗时：{round(time.time() - start_time, 1)} 秒")

        # 写入结果
        result = [[row.da_id, row.city_id, row.poi_id, row.aoi_id, row.base_set, row.candidate_set, int(-1)] for row in data]
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            adjust_treatment = optimize_alg.get_adjust_treatment()
            for i in optimize_alg.i_range:
                result[i][-1] = adjust_treatment[i]

        return result.__iter__()

# encoding=utf-8
import argparse
import time
from collections import defaultdict,deque

import shapely
from pyproj import Geod
from pyspark.sql.types import *

from jobs.spark_job import SparkJob
from shapely import ops
from utils.base_util import BaseUtil
from pyspark.rdd import Partitioner
import shapely
from rtree import index
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon, LineString
import h3
from shapely.ops import nearest_points

precisions = 9
geod = Geod(ellps="WGS84")


class KeyCountPartitioner(Partitioner):
    """根据键数量动态分区的正确实现"""

    def __init__(self, keys):
        self.keys_dict = {k: i for i, k in enumerate(sorted(set(keys)))}
        self._numPartitions = len(self.keys_dict)

    def numPartitions(self):
        return max(1, self._numPartitions)  # 确保至少1个分区

    def getPartition(self, key):
        return self.keys_dict.get(key, 0)  # 未知键分配到0号分区


def st_split(polygon_wkt, line_wkt):
    """Split a Polygon with a LineString"""

    boundary = polygon_wkt.boundary
    union = ops.unary_union([boundary, line_wkt])

    collection = []
    for geometry in list(shapely.ops.polygonize(union)):
        if polygon_wkt.contains(geometry.representative_point()):
            collection.append(geometry)
    return collection


def h3_to_polygon(h3_index):
    temp = h3.cell_to_boundary(h3_index)
    corrected_coords = [(lon, lat) for lat, lon in temp]
    corrected_coords.append(corrected_coords[0])
    polygon = Polygon(corrected_coords)
    return polygon


def get_rtree(polygon_dict):
    rtree_index = index.Index()
    for h3_id, polygon in polygon_dict.items():
        rtree_index.insert(h3.str_to_int(h3_id), polygon.bounds)
    return rtree_index


def seq_op(acc, item):
    acc.add(item[0])
    return acc


def comb_op(acc1, acc2):
    return acc1.union(acc2)


class DataProcess(object):

    def __init__(self, sc, sql_context):
        self.sc = sc
        self.sqlContext = sql_context

    def data_to_hive(self, results, output_table, dt):
        print('************data_to_hive **************', output_table, dt)
        print('type bill_time:', type(results))

        fields = [
            StructField("shadow_id", LongType(), True),
            StructField("h3_code", StringType(), True),
            StructField("shadow_p_wkt", StringType(), True),
            StructField("area", DoubleType(), True),
            StructField("blockid", StringType(), True),
            StructField("detailed_category", StringType(), True),
            StructField("code", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("level", IntegerType(), True),
            StructField("parentcode", IntegerType(), True)
        ]

        dataf = self.sqlContext.createDataFrame(results, schema=StructType(fields))
        #print(dataf.take(1))
        print()
        dataf.registerTempTable("tmp")
        self.sqlContext.sql("set hive.exec.dynamic.partition=true")
        self.sqlContext.sql("set hive.exec.parallel = true")
        self.sqlContext.sql("set hive.exec.dynamic.partition.mode = nonstrict")
        self.sqlContext.sql("set spark.sql.shuffle.partitions = 4000")
        self.sqlContext.sql("set spark.shuffle.compress=true")
        self.sqlContext.sql("set spark.io.compression.codec=snappy")

        self.sqlContext.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {output_table}
            (
                `shadow_id`                bigint        COMMENT '',
                `h3_code`                  string        COMMENT '',
                `shadow_p_wkt`             string        COMMENT '',
                `area`                     double        COMMENT '',
                `blockid`                  string        COMMENT '',
                `detailed_category`        string        COMMENT '',
                `code`                     bigint        COMMENT '',
                `name`                     string        COMMENT '',
                `level`                    bigint        COMMENT '',
                `parentcode`               bigint        COMMENT ''          
            ) COMMENT ''
            PARTITIONED BY (dt string)
            ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
            STORED AS ORC
            """)

        self.sqlContext.sql(
            f"""insert  overwrite table {output_table} partition(dt='{dt}')
                    select * from tmp""")
        return

    def process(self, raw_shadow_sql_query, output_table, ver):
        print(f"*****任务开始*****")
        start = time.time()

        def convert_to_simple_polygon(polygon):
            exterior = polygon.exterior
            interiors = list(polygon.interiors)

            for interior in interiors:
                point1, point2 = nearest_points(interior, exterior)
                line_b = LineString([(point1.x, point1.y), (point2.x, point2.y)]).buffer(0.00001, resolution=2)
                polygon = polygon.difference(line_b)
            return polygon

        def remove_contain_data(shadow_h3_dict):  #shadow_h3_dict[h3_id] = [geom, row.blockid, row.detailed_category, row.code, row.name, row.level, row.parentcode]
            rtree_index = index.Index()
            new_shadow_h3_dict = {}
            for key, item in shadow_h3_dict.items():
                value = item[0]
                i = h3.str_to_int(key)
                rtree_index.insert(i, value.bounds)
                # temp_shadow_h3_dict[i] = value

            # for key, value in temp_shadow_h3_dict.items():
            for key, item in shadow_h3_dict.items():
                value = item[0]
                match_shadows = set(rtree_index.intersection(value.bounds))
                for match_shadow_id in match_shadows:
                    match_shadow_id = h3.int_to_str(match_shadow_id)
                    if match_shadow_id != key:
                        match_polygon = shadow_h3_dict.get(match_shadow_id)[0]
                        if match_polygon.contains(value):
                            #print(value)
                            break
                        if match_polygon.intersection(value).area / value.area > 0.95:
                            #print(value)
                            break

                else:
                    new_shadow_h3_dict[key] = item
            return new_shadow_h3_dict

        def union_small(new_shadow_h3_dict):
            union_dict = defaultdict(list)
            for key, item in new_shadow_h3_dict.items():
                # 获取对应的9级ID
                #value = item[0]
                h3_9_index = h3.cell_to_parent(key, 9)
                union_dict[h3_9_index].append(item)
            return union_dict

        def is_not_small(geom):
            if geom.length * 1.0 / geom.area * (geom.length / 20) > 10 or geom.area * 1e10 < 1000:
                return False
            return True

        def fix_invalid_geometry(geom):
            """修复无效的几何图形"""
            if not geom.is_valid:
                try:
                    # 尝试使用buffer(0)修复
                    fixed_geom = geom.buffer(0)
                    if fixed_geom.is_valid:
                        return fixed_geom
                except:
                    pass
                try:
                    # 如果是MultiPolygon，尝试分别修复每个Polygon
                    if isinstance(geom, MultiPolygon):
                        valid_polygons = []
                        for poly in geom.geoms:
                            if poly.is_valid:
                                valid_polygons.append(poly)
                            else:
                                fixed_poly = poly.buffer(0)
                                if fixed_poly.is_valid:
                                    valid_polygons.append(fixed_poly)
                        if valid_polygons:
                            return MultiPolygon(valid_polygons)
                except:
                    pass
            return geom

        def union_polygons(union_dict, shadow_h3_dict):
            # 合并多边形
            new_union_dict = {}
            for key, item_list in union_dict.items():
                if len(item_list) > 1:
                    # 按面积排序
                    sorted_polygons = sorted(item_list, key=lambda x: x[0].area)

                    for i, item in enumerate(sorted_polygons):
                        current_poly = fix_invalid_geometry(item[0])
                        if not current_poly.is_valid:
                            continue

                        current_indices = {i}
                        current_perimeter = current_poly.length * 1e5

                        # 找到相交比例最大的polygon
                        max_ratio = 0
                        max_ratio_index = -1
                        max_ratio_polygon = None

                        # 计算与其他polygon的相交线长度
                        for j, other_item in enumerate(sorted_polygons[i + 1:], start=i + 1):
                            other_poly = fix_invalid_geometry(other_item[0])
                            if not other_poly.is_valid:
                                continue

                            try:
                                inter_line = current_poly.buffer(0.00005, resolution=2).intersection(
                                    other_poly.buffer(0.00005, resolution=2))
                                if not inter_line.is_empty:
                                    inter_line_len = inter_line.length * 1e5
                                    # 计算相交线长度与当前polygon周长的比例
                                    ratio = inter_line_len / current_perimeter

                                    if ratio > max_ratio:
                                        max_ratio = ratio
                                        max_ratio_index = j
                                        max_ratio_polygon = other_poly
                            except Exception as e:
                                print(f"计算相交时出错: {e}")
                                continue

                        # 如果找到相交比例最大的polygon且满足阈值条件，则进行合并
                        if max_ratio > 0.3:  # 相交线长度超过周长的30%
                            try:
                                # 更新other_polygon的形状
                                other_poly = ops.unary_union([max_ratio_polygon.buffer(0.0001, resolution=2),
                                                              current_poly.buffer(0.0001, resolution=2)]).buffer(
                                    -0.0001,
                                    resolution=2)

                                other_poly = fix_invalid_geometry(other_poly)

                                if not other_poly.is_valid:
                                    continue

                                # 对合并后的polygon进行坐标点简化
                                if isinstance(other_poly, MultiPolygon):
                                    simplified_geoms = []
                                    for geom in other_poly.geoms:
                                        simplified_geom = geom.simplify(0.00001, preserve_topology=True)
                                        if simplified_geom.is_valid and simplified_geom.area > 0:
                                            simplified_geoms.append(simplified_geom)
                                    if simplified_geoms:
                                        other_poly = MultiPolygon(simplified_geoms)
                                    else:
                                        continue
                                else:
                                    other_poly = other_poly.simplify(0.00001, preserve_topology=True)
                                    if not other_poly.is_valid or other_poly.area <= 0:
                                        continue

                                sorted_polygons[max_ratio_index][0] = other_poly
                                current_indices.add(max_ratio_index)
                            except Exception as e:
                                print(f"合并多边形时出错: {e}")
                                continue

                        # 如果没有被合并，则保留current_poly
                        if len(current_indices) == 1:
                            try:
                                if isinstance(current_poly, MultiPolygon):
                                    for poly in current_poly.geoms:
                                        if not poly.is_valid:
                                            continue
                                        if poly.interiors.__len__() > 0:
                                            poly = convert_to_simple_polygon(poly)
                                        if not is_not_small(poly):
                                            continue
                                        h3_id = h3.latlng_to_cell(poly.centroid.y, poly.centroid.x, precisions)
                                        if h3_id in shadow_h3_dict:
                                            h3_id = h3.latlng_to_cell(poly.centroid.y, poly.centroid.x, precisions + 1)
                                            if h3_id in shadow_h3_dict:
                                                h3_id = h3.latlng_to_cell(poly.centroid.y, poly.centroid.x,
                                                                          precisions + 2)
                                        new_union_dict[h3_id] = [poly, item[1], item[2], item[3], item[4], item[5], item[6]]
                                else:
                                    if current_poly.interiors.__len__() > 0:
                                        current_poly = convert_to_simple_polygon(current_poly)
                                    if not is_not_small(current_poly):
                                        continue
                                    h3_id = h3.latlng_to_cell(current_poly.centroid.y, current_poly.centroid.x,
                                                              precisions)
                                    if h3_id in shadow_h3_dict:
                                        h3_id = h3.latlng_to_cell(current_poly.centroid.y, current_poly.centroid.x,
                                                                  precisions + 1)
                                        if h3_id in shadow_h3_dict:
                                            h3_id = h3.latlng_to_cell(current_poly.centroid.y, current_poly.centroid.x,
                                                                      precisions + 2)
                                    new_union_dict[h3_id] = [current_poly, item[1], item[2], item[3], item[4], item[5], item[6]]
                            except Exception as e:
                                print(f"处理单个多边形时出错: {e}")
                                continue
                else:
                    try:
                        current_poly = fix_invalid_geometry(item_list[0][0])
                        if current_poly.is_valid and is_not_small(current_poly):
                            new_union_dict[key] = item_list[0]
                    except Exception as e:
                        print(f"处理列表中单个多边形时出错: {e}")
                        continue

            return new_union_dict

        def get_shaodow_h3(iter):  # blockid,raw_shadow_p_wkt,code, name, level, parentcode, detailed_category
            # 最终结果
            shadow_h3_dict = {}

            for block_id, row in iter:  #blockid,raw_shadow_p_wkt,code, name, level, parentcode, detailed_category

                shadow_polygon = wkt.loads(row.raw_shadow_p_wkt)

                if shadow_polygon.area * 1e10 < 5e4:  # 面积太小不切割
                    # print(BaseUtil.geometry2string(shadow_polygon))
                    h3_id = h3.latlng_to_cell(shadow_polygon.centroid.y, shadow_polygon.centroid.x, precisions)
                    if h3_id in shadow_h3_dict:
                        h3_id = h3.latlng_to_cell(shadow_polygon.centroid.y, shadow_polygon.centroid.x, precisions + 1)
                        if h3_id in shadow_h3_dict:
                            h3_id = h3.latlng_to_cell(shadow_polygon.centroid.y, shadow_polygon.centroid.x,
                                                      precisions + 2)
                    if shadow_polygon.interiors.__len__() > 0:
                        shadow_polygon = convert_to_simple_polygon(shadow_polygon)

                    shadow_h3_dict[h3_id] = [shadow_polygon, row.blockid, row.detailed_category, row.code, row.name, row.level, row.parentcode]
                    continue

                # shadow_area_dict[shadow_id] = shadow_polygon.area*1e10

                # block_match_h3s = set(h3_rtree_index.intersection(shadow_polygon.bounds))
                # 获取shadow范围内的h3对应polygon
                shadow_match_h3s = h3.geo_to_cells(shadow_polygon.buffer(0.003), precisions)
                # h3_polygon_dict = {h3_id: h3_to_polygon(h3_id) for h3_id in shadow_match_h3s}
                m_h3_line_list = [LineString(h3_to_polygon(h3_id).exterior.coords) for h3_id in shadow_match_h3s]
                m_h3_line_union = ops.unary_union(m_h3_line_list)
                geom_of_shadow = st_split(shadow_polygon, m_h3_line_union)

                polygon_of_shadow_list = []
                for geom in geom_of_shadow:
                    if geom.interiors.__len__() > 0:  # 有内环的，挖线转成无内环
                        geom = convert_to_simple_polygon(geom)
                    polygon_of_shadow_list.append(geom)

                s_h3_polygon_dict = {}  # 每个阴影h3的polygon
                s_h3_area_dict = {}  # 每个阴影h3的面积
                for poly in polygon_of_shadow_list:
                    # print(BaseUtil.geometry2string(poly))
                    p = poly.centroid
                    h3_index = h3.latlng_to_cell(p.y, p.x, precisions)

                    if h3_index in s_h3_polygon_dict:
                        # print(h3_index)
                        h3_index = h3.latlng_to_cell(p.y, p.x, precisions + 1)
                        # print("==",h3_index)
                    s_h3_polygon_dict[h3_index] = poly
                    s_h3_area_dict[h3_index] = poly.area * 1e10

                    # print(h3_index, BaseUtil.geometry2string(poly))

                #     print(BaseUtil.geometry2string(poly))
                # print(s_h3_area_dict)

                sorted_s_h3_area_items = sorted(s_h3_area_dict.items(), key=lambda x: x[1])
                # print(sorted_s_h3_area_items)
                # print(s_h3_polygon_dict.keys())

                s_h3_polygon_rtree = get_rtree(s_h3_polygon_dict)
                for item in sorted_s_h3_area_items:

                    raw_s_h3_index = item[0]
                    h3_polygon = s_h3_polygon_dict[raw_s_h3_index]

                    # print("------------------", raw_s_h3_index)

                    if h3_polygon.area * 1e10 > 6e4:  # 面积超过xxxxx的不处理
                        # print(h3_polygon.area*1e10)
                        # print("bububub")
                        continue

                    # print('------------------', raw_s_h3_index, h3_polygon.area*1e10)

                    # h3_9_index = raw_s_h3_index
                    # if h3.get_resolution(raw_s_h3_index) == 10:
                    #     print("bububub")
                    #     h3_9_index = h3.cell_to_parent(raw_s_h3_index, 9)

                    neighbor_h3s = [h3.int_to_str(i) for i in list(s_h3_polygon_rtree.intersection(h3_polygon.bounds))
                                    if h3.int_to_str(i) != raw_s_h3_index]

                    neighbor_h3s = sorted(neighbor_h3s)

                    inter_line_len_list = []  # 与周边的交线长度
                    for n_index in neighbor_h3s:
                        if n_index in s_h3_polygon_dict:
                            n_h3_polygon = s_h3_polygon_dict[n_index]
                            inter_line = n_h3_polygon.intersection(h3_polygon)
                            if inter_line.is_empty:
                                inter_line_len = -1.0
                            else:
                                inter_line_len = inter_line.length * 1e5
                        else:
                            inter_line_len = -1.0
                        inter_line_len_list.append(inter_line_len)

                    # print(neighbor_h3s)
                    # print(inter_line_len_list)
                    # print(neighbor_h3s)
                    if len(inter_line_len_list) == 0 or max(inter_line_len_list) < 0:  # 剩余不能合并的不处理
                        # print(h3_polygon)
                        continue

                    max_inter_h3_id = neighbor_h3s[inter_line_len_list.index(max(inter_line_len_list))]
                    # print(max_inter_h3_id,"----")
                    # print(s_h3_polygon_dict.keys())
                    # print(max_inter_h3_id)
                    # print(inter_line_len_list.index(min(inter_line_len_list)))
                    # print(shadow_id)

                    max_inter_h3_polygon = s_h3_polygon_dict[max_inter_h3_id]
                    union_polygon = ops.unary_union([h3_polygon, max_inter_h3_polygon])
                    # print(BaseUtil.geometry2string(union_polygon))

                    s_h3_polygon_dict[max_inter_h3_id] = union_polygon
                    s_h3_polygon_dict.pop(raw_s_h3_index)

                # print(s_h3_polygon_dict.keys())

                for key, geom in s_h3_polygon_dict.items():  # 最终结果
                    geom = geom.buffer(-0.00005)

                    if geom.is_empty:
                        continue

                    if isinstance(geom, MultiPolygon):
                        for poly in geom.geoms:  # 遍历所有子多边形
                            if not poly.is_valid or poly.is_empty:
                                continue
                            # print(BaseUtil.geometry2string(poly))
                            if poly.area * 1e10 < 1000:  # 面积过小的不要
                                continue
                            if poly.interiors.__len__() > 0:  # 有内环的，取外环部分
                                # poly = BaseUtil.string2geometry(BaseUtil.geometry2string(poly.exterior, choose_type='linearring'))
                                poly = convert_to_simple_polygon(poly)

                            h3_id = h3.latlng_to_cell(poly.centroid.y, poly.centroid.x, precisions)
                            if h3_id in shadow_h3_dict:
                                h3_id = h3.latlng_to_cell(poly.centroid.y, poly.centroid.x, precisions + 1)
                                if h3_id in shadow_h3_dict:
                                    h3_id = h3.latlng_to_cell(poly.centroid.y, poly.centroid.x, precisions + 2)
                            shadow_h3_dict[h3_id] = [poly, row.blockid, row.detailed_category, row.code, row.name, row.level, row.parentcode]
                    else:
                        if not geom.is_valid or geom.is_empty:
                            continue
                        if geom.area * 1e10 < 1000:  # 面积过小的不要
                            continue
                        if geom.interiors.__len__() > 0:  # 有内环的，取外环部分
                            geom = convert_to_simple_polygon(geom)
                        h3_id = h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, precisions)
                        if h3_id in shadow_h3_dict:
                            h3_id = h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, precisions + 1)
                            if h3_id in shadow_h3_dict:
                                h3_id = h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, precisions + 2)
                        shadow_h3_dict[h3_id] = [geom, row.blockid, row.detailed_category, row.code, row.name, row.level, row.parentcode]
            print(len(shadow_h3_dict))

            new_shadow_h3_dict = remove_contain_data(shadow_h3_dict)
            print("remove_contain_data", len(new_shadow_h3_dict))

            union_dict = union_small(new_shadow_h3_dict)
            print("union_small", len(union_dict))

            new_union_dict = union_polygons(union_dict, shadow_h3_dict)
            print("union_polygons", len(new_union_dict))

            shadow_h3_list = [[h3.str_to_int(h3_id), h3_id, str(item[0].buffer(0)),
                               round(abs(geod.geometry_area_perimeter(item[0])[0]), 2),
                               item[1], item[2], item[3], item[4], item[5], item[6]] for h3_id, item in new_union_dict.items()
                              if isinstance(item[0], Polygon) and len(item[0].interiors) == 0]

            if len(shadow_h3_list) > 0:
                return shadow_h3_list
            else:
                return []

        raw_shadow_rdd = self.sqlContext.sql(raw_shadow_sql_query).rdd.map(lambda row: (row.code, row)).cache()

        # 初始化空集合（注意需要返回新对象）
        initial_set = set()
        distinct_keys = raw_shadow_rdd.aggregate(initial_set, seq_op, comb_op)

        # 初始化分区器
        partitioner = KeyCountPartitioner(distinct_keys)
        print(f"实际使用的分区数: {partitioner.numPartitions()}")  # 实际使用的分区数: 3118

        # 正确使用分区器的方式
        shadow_h3_rdd = raw_shadow_rdd.partitionBy(
            numPartitions=partitioner.numPartitions(),
            partitionFunc=lambda k: partitioner.getPartition(k)  # 关键修正点
        ).mapPartitions(get_shaodow_h3).filter(lambda row: row != [])

        shadow_h3_rdd.take(2)

        # ========== 新增：按shadow_id聚合处理 ===========
        def merge_shadow_group(rows):
            # rows: list of [shadow_id, h3_id, shadow_p_wkt, area, blockid, detailed_category, code, name, level, parentcode]
            if len(rows) == 1:
                row = rows[0]  # 直接返回里面的第一条
                h3_id = h3.latlng_to_cell(wkt.loads(row[2]).centroid.y,  wkt.loads(row[2]).centroid.x, precisions + 2)
                row[0] = h3.str_to_int(h3_id)
                row[1] = h3_id
                return row
            polygons = [wkt.loads(row[2]) for row in rows]
            # 直接做并集
            merged_poly = ops.unary_union(polygons)
            # 处理并集结果
            def get_largest_polygon(geom):
                if isinstance(geom, MultiPolygon):
                    # 取面积最大的
                    largest = max(geom.geoms, key=lambda g: g.area)
                else:
                    largest = geom
                # 有内环则取外环
                if largest.interiors.__len__() > 0:  # 有内环的，取外环部分
                    largest = convert_to_simple_polygon(largest)
                return largest

            merged_poly = get_largest_polygon(merged_poly)
            # 检查有效性，若无效则修复
            if not merged_poly.is_valid:
                try:
                    fixed = merged_poly.buffer(0)
                    if fixed.is_valid:
                        merged_poly = fixed
                    else:
                        # 修复失败，直接取原polygons中面积最大的polygon
                        merged_poly = max(polygons, key=lambda g: g.area)
                except:
                    merged_poly = max(polygons, key=lambda g: g.area)
            # 取面积最大的那条的其他字段
            areas = [poly.area for poly in polygons]
            max_idx = areas.index(max(areas))
            row = rows[max_idx]
            # 重新计算area
            area = round(abs(geod.geometry_area_perimeter(merged_poly)[0]), 2)

            h3_id = h3.latlng_to_cell(merged_poly.centroid.y,  merged_poly.centroid.x, precisions + 2)
            # 返回格式
            return [
                h3.str_to_int(h3_id), #row[0],  # shadow_id
                h3_id,  #row[1],  # h3_id
                str(merged_poly),  # shadow_p_wkt
                area,
                row[4], row[5], row[6], row[7], row[8], row[9]
            ]
        # ========== 聚合处理结束 ===========

        shadow_h3_rdd = shadow_h3_rdd \
            .groupBy(lambda row: row[0]) \
            .mapValues(list) \
            .map(lambda kv: merge_shadow_group(kv[1]))

        self.data_to_hive(shadow_h3_rdd, output_table, ver)

        end = time.time()
        print("end - start", end - start)

    def run_city_county(self, spark_job, dt):

        raw_shadow_sql_query = f"""select blockid, raw_shadow_p_wkt,
                 if(code < 0, -cast(floor(rand()*100)-1 as bigint),  code) as code,
                 name, level, parentcode, detailed_category
                from mart_peisongpa.raw_shadow_of_code_block_aoi_daily
                where dt='{dt}'"""

        print(raw_shadow_sql_query)

        output_table = "mart_peisongpa.shadow_h3_of_block_aoi_daily"
        #spark_job.drop_table(output_table)

        self.process(raw_shadow_sql_query, output_table, dt)


def process(args):
    """Spark任务接口 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=str, default='20250707', required=True)
    args, unknown = parser.parse_known_args()

    spark_job = SparkJob()
    job = DataProcess(spark_job.sc, spark_job.sqlContext)

    dt = args.dt

    job.run_city_county(spark_job, dt)

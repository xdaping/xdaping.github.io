#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/8/4 18:55

import math
import base64
import io
import numpy as np
from PIL import Image
import shapely.wkt as wkt
from shapely.ops import transform
import pyproj
from rtree import index
from coord_convert.transform import wgs2gcj, gcj2wgs
from shapely import ops
from shapely.geometry import MultiPolygon, Polygon, MultiPoint, Point
import geohash
import cn_digits_norm as normer
import datetime

LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05


class BaseUtil:
    project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), pyproj.CRS('EPSG:32618'), always_xy=True).transform
    reverse = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:32618'), pyproj.CRS('EPSG:4326'), always_xy=True).transform

    @staticmethod
    def normalize_text(text):
        # 汉字转数字
        x = normer.change_chinese_num2arab(text)
        # 去除文本中的标点符号，只提取中文英文数字
        x = normer.get_rid_of_punc(x)
        # 全角转半角
        norm_text = normer.str_full_width2half_width(x)
        return norm_text

    @staticmethod
    def get_diff_row_col(lng, lat, bounds, pixel_length=1):
        """
        给定经纬度，计算距中心点的行列数
        :param lng:
        :param lat:
        :return:
        """
        centroid = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
        col_gap = math.ceil(abs(lng - centroid[0]) / (pixel_length * LNG_PER_METER) - 0.5)
        col_gap = col_gap if lng > centroid[0] else -col_gap
        row_gap = math.ceil(abs(lat - centroid[1]) / (pixel_length * LAT_PER_METER) - 0.5)
        row_gap = row_gap if lat > centroid[1] else -row_gap
        return row_gap, col_gap

    @staticmethod
    def get_matrix_idx(lng, lat, bounds, row_num, col_num):
        """
        根据经纬度获取二维坐标,超出AOI范围返回 nan
        :param lng:
        :param lat:
        :return:
        """
        row_gap, col_gap = BaseUtil.get_diff_row_col(lng, lat, bounds)
        row_index = row_num // 2 + row_gap
        col_index = col_num // 2 + col_gap
        if 0 <= row_index < row_num and 0 <= col_index < col_num:
            return row_index, col_index
        else:
            return np.nan, np.nan

    @staticmethod
    def get_coordinate_by_idx(x, y, bounds, row_num, col_num):
        """
        根据matrix 下标 计算经纬度
        :param x: row_index
        :param y: col_index
        :return:
        """
        centroid = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
        if x < 0 or x >= row_num or y < 0 or y >= col_num:
            return -1, -1
        row_gap = row_num // 2 - x
        col_gap = col_num // 2 - y
        lng = centroid[0] - col_gap * LNG_PER_METER
        lat = centroid[1] - row_gap * LAT_PER_METER
        return lng, lat

    @staticmethod
    def wgs2gcj_for_point(min_lon, max_lat):
        line_wgs84 = Point((min_lon, max_lat))
        line_gcj02 = ops.transform(wgs2gcj, line_wgs84)
        point = BaseUtil.geometry2string(line_gcj02, 'point').split(',')
        min_lon, max_lat = float(point[0]), float(point[1])
        return min_lon, max_lat

    @staticmethod
    def gcj2wgs_for_point(min_lon, max_lat):
        line_gcj02 = Point((min_lon, max_lat))
        line_wgs84 = ops.transform(gcj2wgs, line_gcj02)
        point = BaseUtil.geometry2string(line_wgs84, 'point').split(',')
        min_lon, max_lat = float(point[0]), float(point[1])
        return min_lon, max_lat

    @staticmethod
    def numpy2bytes(im, format, image_depth):
        img = Image.fromarray(im).convert(image_depth)
        with io.BytesIO() as image_bytes:
            img.save(image_bytes, format=format)
            image_bytes = image_bytes.getvalue()
        return image_bytes

    @staticmethod
    def bytes2base64(image_bytes):
        image_base64 = base64.b64encode(image_bytes)
        return bytearray(image_base64)

    @staticmethod
    def numpy2base64(im, format='png', image_depth='L'):
        image_bytes = BaseUtil.numpy2bytes(im, format, image_depth)
        image_base64 = BaseUtil.bytes2base64(image_bytes)
        return bytearray(image_base64)

    @staticmethod
    def base64Tobytes(image_base64):
        image_bytes = base64.b64decode(image_base64)
        return image_bytes

    @staticmethod
    def bytes2numpy(image_bytes):
        img = np.array(Image.open(io.BytesIO(image_bytes)))
        return img

    @staticmethod
    def base64ToNumpy(image_base64):
        image_bytes = BaseUtil.base64Tobytes(image_base64)
        img = BaseUtil.bytes2numpy(image_bytes)
        return img

    @staticmethod
    def GoogleLonLatToXYZ(lng, lat, zoom=18):
        n = 2 ** zoom
        tx = ((lng + 180) / 360) * n
        ty = (1 - (math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi)) / 2 * n
        return int(tx), int(ty)

    @staticmethod
    def GoogleXYZToLonLat(tx, ty, zoom=18):
        n = 2 ** zoom
        lon = tx / n * 360 - 180
        lat = math.atan(math.sinh(math.pi * (1 - 2 * ty / n))) * 180 / math.pi
        return lon, lat

    @staticmethod
    def get_max_polygon_from_multiarea_polygon(polygon):
        max_area = 0
        max_polygon = Polygon()
        p_ex_str = BaseUtil.geometry2string(polygon.exterior, choose_type='linearring')
        p = BaseUtil.string2geometry(p_ex_str)
        if p.area > max_area:
            max_area = p.area
            max_polygon = p
        for in_area in polygon.interiors:
            p = BaseUtil.string2geometry(BaseUtil.geometry2string(in_area, choose_type='linearring'))
            if p.area > max_area:
                max_area = p.area
                max_polygon = p
        return max_polygon

    @staticmethod
    def get_max_polygon_from_multipolygon(multipolygon):
        max_area = 0
        max_polygon = Polygon()
        for p in multipolygon:
            if p.area > max_area:
                max_area = p.area
                max_polygon = p
        return max_polygon

    @staticmethod
    def get_polygon_for_one_geohash(g):
        one_point_set_list = []
        (min_lng, min_lat, max_lng, max_lat) = geohash.bbox(g)
        one_point_set_list.append((min_lng, min_lat))
        one_point_set_list.append((min_lng, max_lat))
        one_point_set_list.append((max_lng, max_lat))
        one_point_set_list.append((max_lng, min_lat))
        # one_point_set_list.append((min_lng, min_lat))
        return Polygon(one_point_set_list)

    @staticmethod
    def get_all_area_of_polygon(pid, polygon):
        area_list = []
        if polygon.interiors.__len__() > 0:
            p_ex_str = BaseUtil.geometry2string(polygon.exterior, choose_type='linearring')
            p = BaseUtil.string2geometry(p_ex_str)
            area_list.append([pid, str(p)])
            for in_area in polygon.interiors:
                p = BaseUtil.string2geometry(BaseUtil.geometry2string(in_area, choose_type='linearring'))
                area_list.append([pid, str(p)])
        else:
            area_list.append([pid, str(polygon)])
        return area_list

    @staticmethod
    def get_rotated_rectangle_ratio(small_grid_ploygon):
        """
        得到最小外接矩形占比
        """
        small_grid_rectangle = small_grid_ploygon.minimum_rotated_rectangle
        rotated_rectangle_ratio = small_grid_ploygon.area / small_grid_rectangle.area if small_grid_rectangle.area > 0 else -1.0
        return rotated_rectangle_ratio

    @staticmethod
    def get_rtree_index():
        """
        初始化rtree
        """
        idx = index.Index()
        return idx

    @staticmethod
    def get_rtree_from_link_list(link_dict, rtree_index, link_list):
        """
        为路网建立rtree
        """
        for link_info in link_list.split('&#&'):
            # 示例：784293903&_&LINESTRING (120.059313 43.868825, 120.060193 43.868842)&_&{"road_grade": "7", ...}
            link_id, geom, link_flag = link_info.split('&_&')
            link_coors = geom[12:-1].replace(', ', ';').replace(' ', ',')
            link_dict[int(link_id)] = link_coors + "\t" + link_flag
            rtree_index.insert(int(link_id), wkt.loads(geom).bounds)
        return rtree_index, link_dict

    @staticmethod
    def point2lng_lat(aoi_shape):
        """
        POINT格式转lng，lat
        """
        point = str(aoi_shape.centroid).split('POINT (')[1].split(')')[0].split(' ')
        return float(point[0]), float(point[1])

    @staticmethod
    def get_inter_percent_of_addr_pair(hull1, hull2):
        """
        根据两个AOI形状计算，交集/AOI1、交集/AOI2、交并比
        """
        inter_area = hull1.intersection(hull2).area
        union_area = hull1.union(hull2).area
        inter_percent = inter_area * 1.0 / union_area
        inter_hull1 = inter_area * 1.0 / hull1.area
        inter_hull2 = inter_area * 1.0 / hull2.area
        return inter_hull1, inter_hull2, inter_percent

    @staticmethod
    def geodistance(lng1, lat1, lng2, lat2):
        """
        计算两个经纬度点之间的距离，单位米
        """
        diff_lat = (lat1 - lat2) / 2.0
        diff_lng = (lng1 - lng2) / 2.0
        coors_sum = math.pow(math.sin((math.pi * diff_lat) / 180.0), 2) + \
                    math.cos((math.pi * lat1) / 180.0) * math.cos((math.pi * lat2) / 180.0) * \
                    math.pow(math.sin(math.pi * diff_lng) / 180.0, 2)
        result = 2 * 6378137 * math.asin(math.sqrt(coors_sum))
        return result

    @staticmethod
    def geo_dist_point(point1, point2):
        """
        计算两个经纬度点之间的距离，单位米
        """
        lng1, lat1 = point1.coords[0]
        lng2, lat2 = point2.coords[0]

        diff_lat = (lat1 - lat2) / 2.0
        diff_lng = (lng1 - lng2) / 2.0
        coors_sum = math.pow(math.sin((math.pi * diff_lat) / 180.0), 2) + \
                    math.cos((math.pi * lat1) / 180.0) * math.cos((math.pi * lat2) / 180.0) * \
                    math.pow(math.sin(math.pi * diff_lng) / 180.0, 2)
        result = 2 * 6378137 * math.asin(math.sqrt(coors_sum))
        return result

    @staticmethod
    def buffer_m(geom, meter, resolution=1, join_style=2, cap_style=1):
        """
        开buffer，米为单位
        """
        trans_geom = transform(BaseUtil.project, geom)
        dist_buffer = trans_geom.buffer(meter, resolution=resolution, join_style=join_style, cap_style=cap_style)
        geom_buffer = transform(BaseUtil.reverse, dist_buffer)
        return geom_buffer

    @staticmethod
    def grid_line_buffer_m(line, meter1, meter2):
        """
        开buffer，米为单位
        """
        distline = transform(BaseUtil.project, line)
        dist_buffer = distline.buffer(meter1, resolution=1, cap_style=2)
        dist_buffer = dist_buffer.buffer(meter2, resolution=1, join_style=2, cap_style=3)
        dist_buffer = transform(BaseUtil.reverse, dist_buffer)
        return dist_buffer

    @staticmethod
    def string2geometry(string, choose_type='polygon'):
        """
        字符串转不同类型geometry
        """
        if choose_type == 'polygon':
            return wkt.loads('POLYGON ((' + string.replace(',', ' ').replace(';', ',') + '))')
        if choose_type == 'linestring':
            return wkt.loads('LINESTRING (' + string.replace(',', ' ').replace(';', ',') + ')')
        if choose_type == 'point':
            return wkt.loads('POINT (' + string.replace(',', ' ') + ')')
        return None

    @staticmethod
    def geometry2string(geometry, choose_type='polygon'):
        """
        不同类型geometry转字符串
        """
        if choose_type == 'polygon':
            return str(geometry).replace('POLYGON ', 'POLYGON').replace('POLYGON', 'POLYGON ')[10:-2] \
                .replace(', ', ',').replace(',', ', ').replace(', ', ';').replace(' ', ',')
        if choose_type == 'linestring':
            return str(geometry)[12:-1].replace(', ', ',').replace(',', ', ').replace(', ', ';').replace(' ', ',')
        if choose_type == 'point':
            return str(geometry)[7:-1].replace(' ', ',')
        if choose_type == 'linearring':
            return str(geometry)[12:-1].replace(', ', ',').replace(',', ', ').replace(', ', ';').replace(' ', ',')
        if choose_type == 'mutipoint':
            return str(geometry)[12:-1].replace(', ', ',').replace(',', ', ').replace(', ', ';').replace(' ', ',')
        return None

    @staticmethod
    def choose_max_area_from_polygon(new_p):
        """
        从包含多个区域的polygon中找到最大的区域
        """
        p_words = eval(str(new_p)[9:-1].replace('), (', '","').replace(
            '(', '("').replace(')', '")'))
        split_ploygons_areas = [wkt.loads('POLYGON ((' + i + '))').area
                                for i in p_words]
        max_index = split_ploygons_areas.index(max(split_ploygons_areas))
        new_p = wkt.loads('POLYGON ((' + p_words[max_index] + '))')
        return new_p

    @staticmethod
    def get_raw_ploygon_list(pid, split_ploygons):
        """
        根据pid与切割得到的区域统计其中的polygon
        """
        raw_ploygon_list = []
        if type(split_ploygons) == MultiPolygon:
            for grid_p in split_ploygons:
                # 处理polygon有多个部分
                raw_ploygon_list += BaseUtil.get_all_area_of_polygon(pid, grid_p)
        else:
            # 处理polygon有多个部分
            raw_ploygon_list += BaseUtil.get_all_area_of_polygon(pid, split_ploygons)
        return raw_ploygon_list

    @staticmethod
    def remove_return_point(line):
        """
        移除返回的点
        """
        words = line.split(";")
        point_num = len(words)

        point_list = []
        point_pair_list = []

        for i in range(point_num):
            if not point_list.__contains__(words[i]):
                point_list.append(words[i])
            else:
                index = words.index(words[i])
                point_pair_list.append((index, i))

        point_pair_list = sorted(point_pair_list)

        remove_point = set()
        for point_pair in point_pair_list:
            if remove_point.__contains__(point_pair[0]):
                continue
            diff = point_pair[1] - point_pair[0]
            if diff == 2:
                remove_point.add(point_pair[1])
                remove_point.add(point_pair[1] - 1)
            else:
                remove_ploygon = wkt.loads(
                    'POLYGON ((' + ';'.join(words[point_pair[0]: point_pair[1] + 1]).replace(',', ' ').replace(';',
                                                                                                               ',') + '))')
                if remove_ploygon.area <= 1e-10:
                    for i in range(point_pair[0] + 1, point_pair[1] + 1):
                        remove_point.add(i)

        point_pair_list.reverse()
        for point_pair in point_pair_list:
            if remove_point.__contains__(point_pair[1]):
                continue

            return_points = words[point_pair[1]:] + words[:point_pair[0] + 1]
            if len(return_points) <= 3:
                for i in range(point_pair[1] + 1, point_num):
                    remove_point.add(i)
                for i in range(0, point_pair[0]):
                    remove_point.add(i)
            else:
                remove_ploygon = wkt.loads(
                    'POLYGON ((' + ';'.join(return_points).replace(',', ' ').replace(';', ',') + '))')
                if remove_ploygon.area <= 1e-10:
                    for i in range(point_pair[1] + 1, point_num):
                        remove_point.add(i)
                    for i in range(0, point_pair[0]):
                        remove_point.add(i)

        words = [words[i] for i in range(point_num) if not remove_point.__contains__(i)]

        return ';'.join(words)

    @staticmethod
    def simple_remove_near_repeat_point(line, accuracy=5):
        """
        移除相距很近的点
        """
        # 除了起始点与终点，其余点精度5下相等则remove
        words = line.split(";")
        new_points_list = [words[0]]
        temp = [round(float(i), accuracy) for i in words[0].split(',')]
        # print(len(words))
        for i in range(1, len(words)):
            t_i = [round(float(i), accuracy) for i in words[i].split(',')]
            if temp == t_i and i != len(words) - 1:
                continue
            else:
                new_points_list.append(words[i])
                temp = t_i
        polygon_str = ';'.join(new_points_list)
        return polygon_str

    @staticmethod
    def simplify_and_remove_near_repeat_point(line, accuracy=5):
        """
        移除相距很近的点
        """
        # 除了起始点与终点，其余点精度5下相等则remove
        words = line.split(";")
        new_points_list = [words[0]]
        temp = [round(float(i), accuracy) for i in words[0].split(',')]
        # print(len(words))
        for i in range(1, len(words)):
            t_i = [round(float(i), accuracy) for i in words[i].split(',')]
            if temp == t_i and i != len(words) - 1:
                continue
            else:
                new_points_list.append(words[i])
                temp = t_i
        polygon_str = ';'.join(new_points_list)
        if len(new_points_list) > 1000:
            polygon = BaseUtil.string2geometry(polygon_str)
            simplified_polygon = polygon.simplify(tolerance=1e-6)
            polygon_str = BaseUtil.geometry2string(simplified_polygon)
            # print(len(polygon_str.split(';')))
        return polygon_str

    @staticmethod
    def remove_repeat_point(line):
        """
        移除重复的点
        """
        words = line.split(";")
        point_list = []
        for i in range(len(words)):
            if not point_list.__contains__(words[i]):
                point_list.append(words[i])
            else:
                index = point_list.index(words[i])
                # 最后一个点前面出现过 且 不是起点，那么前面这个出现过的点作为起点
                if i == len(words) - 1 and index != 0:
                    point_list.append(words[i])
                    point_list = point_list[index:]

                # 当前点不是最后一个点，但是是起点，那么到此作为终点
                if i != len(words) - 1 and index == 0:
                    point_list.append(words[i])
                    break
        return ';'.join(point_list)

    @staticmethod
    def get_ring(line):
        """
        取得封闭形状
        """
        words = line.split(";")
        size = len(words)

        if size < 3 or words[0] == words[-1]:
            return line, False

        else:
            base_is_ring = False

            word_s = [float(i) for i in words[0].split(',')]
            word_e = [float(i) for i in words[-1].split(',')]

            # 起终点距离
            min_dist = BaseUtil.geodistance(word_s[0], word_s[1], word_e[0], word_e[1])
            s_index = size - 1
            # 从终点前一个点向前开始遍历
            for i in range(size - 2, 0, -1):
                word_i = [float(i) for i in words[i].split(',')]

                # 起点到该点的距离
                dist = BaseUtil.geodistance(word_s[0], word_s[1], word_i[0], word_i[1])

                # 与起点计算距离，如果两点距离比前面最短距离大就break，记录该点的后一个点（也就是最近的点）
                if dist > min_dist:
                    s_index = i + 1
                    break
                else:
                    # 如果两点距离比起终点距离近，就更新当前距离
                    min_dist = dist

            # 起终点距离
            min_dist = BaseUtil.geodistance(word_s[0], word_s[1], word_e[0], word_e[1])
            e_index = 0
            # 从起点后一个点向后开始遍历
            for i in range(1, size - 2, 1):
                word_i = [float(i) for i in words[i].split(',')]

                # 终点到该点的距离
                dist = BaseUtil.geodistance(word_e[0], word_e[1], word_i[0], word_i[1])

                # 与终点计算距离，如果两点距离比前面最短距离大就break，记录该点的前一个点（也就是最近的点）
                if dist > min_dist:
                    e_index = i - 1
                    break
                else:
                    # 如果两点距离比起终点距离近，就更新当前距离
                    min_dist = dist

            # 说明最短距离仍然是起终点距离
            if s_index == size - 1 and e_index == 0:
                # 补上起点形成闭合
                words.append(words[0])
                return ';'.join(words), base_is_ring

            s_points = (';'.join(words[:s_index + 1]) + ';' + words[0])
            e_points = (';'.join(words[e_index:]) + ';' + words[e_index])
            s_ploygon = BaseUtil.string2geometry(s_points)
            e_ploygon = BaseUtil.string2geometry(e_points)

            return [s_points, base_is_ring] if s_ploygon.area > e_ploygon.area else [e_points, base_is_ring]

    @staticmethod
    def fill_n_meter_point(points, n_meter, point_delimiter=';', xy_delimiter=','):
        """
        对于AOI边框插入点加密
        """
        aoi_center_points = ''
        words = points.split(point_delimiter)

        for i in range(len(words) - 1):

            [x1, y1] = [float(i) for i in words[i].strip().split(xy_delimiter)]
            [x2, y2] = [float(i) for i in words[i + 1].strip().split(xy_delimiter)]

            # 当前点与下一个点的距离
            dist = BaseUtil.geodistance(x1, y1, x2, y2)

            # 给定n_meter，则需要插入center_n个点
            center_n = math.ceil(dist / n_meter)  # ceil返回 >= x 的最小整数

            dx = x1 - x2
            dy = y1 - y2
            aoi_center_points += words[i].strip() + point_delimiter

            for j in range(center_n - 1):
                aoi_center_points += str(x1 - round((j + 1) * dx / center_n, 12)) + xy_delimiter + str(
                    y1 - round((j + 1) * dy / center_n, 12)) + point_delimiter

            if i == len(words) - 2:
                aoi_center_points += words[i + 1]
        return aoi_center_points

    @staticmethod
    def get_retrieve_links_of_polygon(_polygon, _rtree_index, _link_dict):
        """
        根据polygon检索得到路网。返回高等级路网集合中的link_id。
        """
        # 自动审核使用的高等级路网集合
        high_way_set = ['motorway', 'motorway_link', 'primary', 'primary_link', 'secondary', 'secondary_link',
                        'trunk', 'trunk_link']
        high_way_ids = set()

        match_link_dict = {}
        grid_match_links = set(_rtree_index.intersection(_polygon.bounds))

        for link_id in grid_match_links:
            link_info = _link_dict[link_id].split('\t')
            link_flag = eval(link_info[1].strip())

            if link_flag.get('highway', '') in high_way_set:
                high_way_ids.add(link_id)

            match_link_dict[link_id] = link_flag.get('highway', '') + " " + \
                                       link_flag.get('name', '') + " " + \
                                       link_flag.get('oneway', 'no') + " " + \
                                       link_flag.get('divider', 'no') + " " + \
                                       link_info[0]  # 道路等级、道路名、坐标(坐标放最后面)

        match_link_set = set()
        for k in match_link_dict:
            match_link_set.add(str(k) + ' ' + match_link_dict[k])  # 坐标放最后面
        return match_link_set, high_way_ids

    @staticmethod
    def get_retrieve_links_of_ploygon(_ploygon, _rtree_index, _link_dict, need_link_set=[]):
        """
        根据polygon检索得到路网
        """
        match_link_dict = {}
        grid_match_links = set(_rtree_index.intersection(_ploygon.bounds))

        for link_id in grid_match_links:
            link_info = _link_dict[link_id].split('\t')
            link_flag = eval(link_info[1].strip())

            if len(need_link_set) != 0 and link_flag.get('highway', '') not in need_link_set:
                continue

            match_link_dict[link_id] = link_flag.get('highway', '') + " " + \
                                       link_flag.get('name', '') + " " + \
                                       link_flag.get('oneway', 'no') + " " + \
                                       link_flag.get('divider', 'no') + " " + \
                                       link_info[0]  # 道路等级、道路名、坐标(坐标放最后面)

        match_link_set = set()
        for k in match_link_dict:
            match_link_set.add(str(k) + ' ' + match_link_dict[k])  # 坐标放最后面
        return match_link_set

    @staticmethod
    def get_convex_hull_from_geohash_set(geohash_set):
        point_set_list = []
        for g in geohash_set:
            (min_lng, min_lat, max_lng, max_lat) = geohash.bbox(g)
            point_set_list.append((min_lng, min_lat))
            point_set_list.append((min_lng, max_lat))
            point_set_list.append((max_lng, max_lat))
            point_set_list.append((max_lng, min_lat))
        points = MultiPoint(point_set_list)
        hull = points.convex_hull
        return hull

    @staticmethod
    def get_former_days(today, persist_days=1):

        def get_former_day(today, diff_day=0):
            return str(datetime.datetime.strptime(today, '%Y%m%d').date() - datetime.timedelta(days=diff_day)).replace(
                "-", "")

        result = []
        for i in range(persist_days):
            result.append(get_former_day(today, i))
        return result


if __name__ == '__main__':
    print('daping')
# arr = np.zeros((5, 5), np.uint8)
# arr[1][1] = 100
# base64 = BaseUtil.numpy2base64(arr)
# print(base64)
# bytearray(b'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAAAAACoBHk5AAAAC0lEQVR4nGNgwAcAAB4AAfb96ZYAAAAASUVORK5CYII=')
# bytearray(b'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAAAAACoBHk5AAAAD0lEQVR4nGNggIAUBmwAAAi2AGVwaoTXAAAAAElFTkSuQmCC')

# x, y = 184594, 99826
# lng1, lat1 = BaseUtil.GoogleXYZToLonLat(x, y)
#
# x, y = 184594, 99827
# lng2, lat2 = BaseUtil.GoogleXYZToLonLat(x, y)
#
# x, y = 184595, 99826
# lng3, lat3 = BaseUtil.GoogleXYZToLonLat(x, y)
#
# print(lng2 - lng1, lat2 - lat1)
# print(lng3 - lng1, lat3 - lat1)

# s = "116.381146,39.955325;116.381064,39.956527;116.381059,39.956664;116.381741,39.956661;116.381751,39.956588;116.382495,39.956623;116.382527,39.955598;116.381855,39.955574;116.381839,39.955338;116.381146,39.955325"
# p = BaseUtil.string2geometry(s)
# print(p.bounds)
#
# l = BaseUtil.string2geometry(s, 'linestring')
# print(l.coords.xy)
#
# p1 = Point((1, 2))
# p2 = Point((2, 3))
# print(p1.distance(p2))
#
# print(BaseUtil.remove_repeat_point('1;2;3;4;1;2;5'))
# print(BaseUtil.remove_repeat_point('1;2;3;4;1;2'))
# print(BaseUtil.remove_repeat_point('1;2;3;4;5;6;7;8;6;5;1;2'))


# print(BaseUtil.geometry2string(p))

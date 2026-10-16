# -*- coding: utf-8 -*-
"""生成 dynasty_world.html：瓦片底图（高德 / OSM / 谷歌卫星）+ 历代地名 + 世界同期政权。

读入 map_data.json 中的中国轮廓、省界、江河（平面投影坐标），反算回经纬度；
地名取自 cities_data.py，世界同期政权取自 world_data.py，历史疆域取自 borders_data.json
（由 fetch_borders.py 生成）；注入 template_world.html。

    python3 build_world.py
"""

import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

from cities_data import DYNASTIES, DYNASTY_ORDER
from world_data import WORLD, REGIONS

# 与 build_map.py 的 project() 互逆
K = math.cos(math.radians(35.0))
SCALE = 9.0


def unproject(x, y):
    return [round(-y / SCALE, 3), round(x / (K * SCALE), 3)]   # [lat, lon]


def path_to_rings(d):
    """'Mx,yLx,y...Z' -> [[[lat, lon], ...], ...]"""
    rings = []
    for part in d.split('Z'):
        part = part.strip().lstrip('M')
        if not part:
            continue
        pts = [unproject(*map(float, xy.split(','))) for xy in part.split('L')]
        rings.append(pts)
    return rings


base = json.load(open('map_data.json', encoding='utf-8'))
# 历史疆域（fetch_borders.py 生成）；缺失时页面照常可用，只是没有疆域图层
borders = json.load(open('borders_data.json', encoding='utf-8')) if os.path.exists('borders_data.json') else {}

dynasties = {}
for key in DYNASTY_ORDER:
    d = DYNASTIES[key]
    cities = []
    for c in d['cities']:
        lat, lon = unproject(c['x'], c['y'])
        cities.append({"name": c['name'], "full": c['full'], "tier": c['tier'],
                       "note": c['note'], "lat": lat, "lon": lon})
    world = WORLD.get(key, {"summary": "", "states": []})
    dynasties[key] = {"label": d['label'], "years": d['years'], "summary": d['summary'],
                      "cities": cities, "worldSummary": world['summary'], "world": world['states'],
                      "borders": borders.get(key)}

out = {
    "china": path_to_rings(base['chinaPath']),
    "provinces": [path_to_rings(pr['d']) for pr in base['provinces']],
    "rivers": [{"name": r['name'], "points": [unproject(x, y) for x, y in r['points']]} for r in base['rivers']],
    "regions": REGIONS,
    "dynastyOrder": DYNASTY_ORDER,
    "dynasties": dynasties,
}

blob = json.dumps(out, ensure_ascii=False, separators=(',', ':'))
tpl = open('template_world.html', encoding='utf-8').read()
open('dynasty_world.html', 'w', encoding='utf-8').write(tpl.replace('%%MAP_DATA%%', blob))

n_city = sum(len(d['cities']) for d in dynasties.values())
n_world = sum(len(d['world']) for d in dynasties.values())
print(f'{len(dynasties)} 朝，地名 {n_city} 处，世界同期政权 {n_world} 条；数据 {len(blob) // 1024} KB')
print('已写出 dynasty_world.html')

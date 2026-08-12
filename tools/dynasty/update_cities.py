# -*- coding: utf-8 -*-
"""只更新地名，不重建底图。

build_map.py 需要 /tmp/china100000.json 与 /tmp/ne110.geojson 两个外部地理数据文件
才能重跑；若只改了 cities_data.py 里的地名，用本脚本即可：读入既有 map_data.json，
替换其中的 dynasties / dynastyOrder，按需放大 viewBox，再把数据注入模板产出成品页面。

    python3 update_cities.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

from cities_data import DYNASTIES, DYNASTY_ORDER

data = json.load(open('map_data.json', encoding='utf-8'))
data['dynasties'] = DYNASTIES
data['dynastyOrder'] = DYNASTY_ORDER

# viewBox 至少要能容下所有地名点
xs, ys = [], []
for dyn in DYNASTIES.values():
    for c in dyn['cities']:
        xs.append(c['x']); ys.append(c['y'])
vx, vy, vw, vh = data['viewBox']
pad = 18
x0 = min(vx, min(xs) - pad); y0 = min(vy, min(ys) - pad)
x1 = max(vx + vw, max(xs) + pad); y1 = max(vy + vh, max(ys) + pad)
data['viewBox'] = [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]

json.dump(data, open('map_data.json', 'w', encoding='utf-8'), ensure_ascii=False)

tpl = open('template.html', encoding='utf-8').read()
blob = json.dumps(data, ensure_ascii=False)
open('dynasty_map.html', 'w', encoding='utf-8').write(tpl.replace('%%MAP_DATA%%', blob))

total = sum(len(d['cities']) for d in DYNASTIES.values())
print(f'viewBox {data["viewBox"]}   共 {len(DYNASTIES)} 朝 {total} 处地名')
for k in DYNASTY_ORDER:
    print(f'  {DYNASTIES[k]["label"]}: {len(DYNASTIES[k]["cities"])}')
print('已写出 map_data.json 与 dynasty_map.html')

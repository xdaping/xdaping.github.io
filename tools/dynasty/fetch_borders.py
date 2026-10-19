# -*- coding: utf-8 -*-
"""下载并精简各朝对应年份的世界历史疆域，生成 borders_data.json（供 build_world.py 使用）。

数据来源：aourednik/historical-basemaps（GPL-3.0）
    https://github.com/aourednik/historical-basemaps
该数据为若干年份的世界政区快照，精度有限（部分边界为约略示意）。
本脚本为每个朝代选一个最接近的年份快照，并把包含该朝都城的多边形标为「本朝」。

    python3 fetch_borders.py [缓存目录]       # 默认缓存到系统临时目录
"""

import json
import math
import os
import sys
import tempfile
import urllib.request

from shapely.geometry import shape, Point, MultiPolygon, Polygon
from shapely.ops import unary_union, polylabel
from shapely.validation import make_valid

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
from cities_data import DYNASTIES, DYNASTY_ORDER

URL = "https://raw.githubusercontent.com/aourednik/historical-basemaps/master/geojson/world_{}.geojson"
CACHE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "historical-basemaps")

# 朝代 → 快照年份（数据集中可用年份里最贴近该朝的一张）
SNAPSHOT = {
    "xia": "bc2000", "shang": "bc1500", "dongzhou": "bc500", "chunqiu": "bc700", "zhanguo": "bc300",
    "qin": "bc200", "xihan": "bc100", "donghan": "100", "sanguo": "200", "xijin": "300",
    "sui": "600", "tang": "800", "beisong": "1100", "nansong": "1200", "yuan": "1300",
    "ming": "1500", "qing": "1880",   # 1800 年快照把澳洲、北美等切成数百个部落，改用 1880
}

TOLERANCE = 0.06      # 简化容差（度）
MIN_AREA = 0.15       # 丢弃小于此面积（平方度）的碎片，岛屿除非属于本朝

from border_names_zh import ZH
from border_overrides import OVERRIDES, RENAME, DROP


def fetch(year):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"world_{year}.geojson")
    if not os.path.exists(path):
        print("下载", year)
        urllib.request.urlretrieve(URL.format(year), path)
    return json.load(open(path, encoding="utf-8"))


def to_latlng(geom):
    """shapely (Multi)Polygon -> [[[ [lat,lon], ... ] rings ] polygons]，坐标保留两位小数"""
    polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    out = []
    for p in polys:
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        rings = [p.exterior] + list(p.interiors)
        rr = []
        for r in rings:
            pts, last = [], None
            for x, y in r.coords:
                q = [round(y, 2), round(x, 2)]
                if q != last:
                    pts.append(q); last = q
            if len(pts) >= 4:
                rr.append(pts)
        if rr:
            out.append(rr)
    return out


# 先秦列国：与都城不在同一多边形、但属华夏诸侯的政区也标为「本朝」
EXTRA_CHINA = {
    "chunqiu": {"Wu"}, "dongzhou": {"Qin", "Wu", "Yue"}, "zhanguo": {"Qin", "Yue"},
}

K = math.cos(math.radians(35.0)); S = 9.0
result = {}
missing = set()
for key in DYNASTY_ORDER:
    year = SNAPSHOT[key]
    caps = [Point(c["x"] / (K * S), -c["y"] / S) for c in DYNASTIES[key]["cities"] if c["tier"] == "capital"]
    groups = {}   # 中文名 -> {"names": [...], "geoms": [...], "china": bool}
    raw = fetch(year)["features"]
    # 手绘修正：先裁成陆地，再按顺序互相扣除，最后从数据集各政区中挖掉
    ov_union = None
    if key in OVERRIDES:
        land = unary_union([make_valid(shape(ft["geometry"])).buffer(0) for ft in raw if ft["geometry"]])
        taken = None
        for o in OVERRIDES[key]["features"]:
            g = Polygon(o["ring"]).buffer(0).intersection(land)
            if taken is not None:
                g = g.difference(taken)
            taken = g if taken is None else unary_union([taken, g])
            groups[("china" if o["china"] else "other", o["zh"])] = {"names": [o["en"]], "geoms": [g], "china": o["china"]}
        ov_union = taken.buffer(0.01)
    rename = RENAME.get(key, {})
    for ft in raw:
        name = ((ft["properties"] or {}).get("NAME") or "").strip()
        if not name or name == "?" or not ft["geometry"] or name in DROP.get(key, ()):
            continue
        g = make_valid(shape(ft["geometry"]))
        if g.geom_type == "GeometryCollection":
            g = g.buffer(0)
        if ov_union is not None:
            g = g.difference(ov_union)
        if g.is_empty:
            continue
        zh = rename.get(name) or ZH.get(name)
        if zh is None:
            missing.add(name); zh = name
        is_china = (ov_union is None and any(g.contains(pt) for pt in caps)) or name in EXTRA_CHINA.get(key, ())
        # 本朝多边形单独成组，避免被同名的其他块（如“汉”）吞并或反向
        gk = ("china", zh) if is_china else ("other", zh)
        grp = groups.setdefault(gk, {"names": [], "geoms": [], "china": is_china})
        if name not in grp["names"]:
            grp["names"].append(name)
        grp["geoms"].append(g)
    feats = []
    for (_, zh), grp in groups.items():
        g = unary_union(grp["geoms"]) if len(grp["geoms"]) > 1 else grp["geoms"][0]
        g = g.simplify(TOLERANCE, preserve_topology=True)
        if not grp["china"]:
            parts = [p for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]) if p.geom_type == "Polygon" and p.area >= MIN_AREA]
            if not parts:
                continue
            g = MultiPolygon(parts) if len(parts) > 1 else parts[0]
        coords = to_latlng(g)
        if not coords:
            continue
        # 标注点：最大一块多边形的“最远离边界点”；a 为该块面积（平方度），页面据此决定何时显示名称
        big = max(g.geoms, key=lambda q: q.area) if g.geom_type == "MultiPolygon" else g
        lp = polylabel(big, tolerance=0.1)
        f = {"n": " / ".join(grp["names"][:3]), "z": zh, "g": coords,
             "p": [round(lp.y, 2), round(lp.x, 2)], "a": round(big.area, 1)}
        if grp["china"]: f["c"] = 1
        feats.append(f)
    feats.sort(key=lambda f: f.get("c", 0))   # 本朝画在最上面
    result[key] = {"year": year, "features": feats}
    china = [f["z"] for f in feats if f.get("c")]
    print(f"{DYNASTIES[key]['label']:<4} {year:>7}  {len(feats):>3} 个政区  本朝: {china}")

if missing:
    print("未翻译（请补入 border_names_zh.py）:", sorted(missing))
json.dump(result, open("borders_data.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("borders_data.json", os.path.getsize("borders_data.json") // 1024, "KB")

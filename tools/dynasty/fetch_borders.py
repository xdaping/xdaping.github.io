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

from shapely.geometry import shape, Point, MultiPolygon
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
    "sui": "600", "tang": "800", "beisong": "1000", "nansong": "1200", "yuan": "1300",
    "ming": "1500", "qing": "1800",
}

TOLERANCE = 0.06      # 简化容差（度）
MIN_AREA = 0.15       # 丢弃小于此面积（平方度）的碎片，岛屿除非属于本朝

ZH = {
    # 东亚
    "Xia": "夏", "Zhoa": "商·周", "Zhou states": "周列国", "Qin": "秦", "Han Empire": "汉", "Han": "汉", "Jin": "晋",
    "Sui Empire": "隋", "Tang Empire": "唐", "Song Empire": "宋", "Liao": "辽", "Xixia": "西夏", "Great Khanate": "元（大汗国）",
    "Ming Chinese Empire": "明", "Manchu Empire": "清", "Qing Empire": "清", "Wu": "吴", "Yue": "越", "Min-Yue": "闽越",
    "Nan-Yue": "南越", "Yueban": "悦般", "Han Zhao": "汉赵", "Nan Chao": "南诏", "Tibet": "吐蕃", "Tibetans": "藏地诸部",
    "Zhangzhung Kingdom": "象雄", "Xiongnu": "匈奴", "Southern Xiongnu": "南匈奴", "Yuezhi": "月氏", "Tungus": "通古斯诸部",
    "Koguryo": "高句丽", "Paekche": "百济", "Silla": "新罗", "Gaya": "伽倻", "Korea": "朝鲜", "Koreans": "古朝鲜诸部",
    "Paleo-Koreans": "古朝鲜诸部", "Japan": "日本", "Imperial Japan (Fujiwara)": "日本（藤原摄关）", "Yayoi": "弥生",
    "Late Jomon culture": "绳文晚期", "Jōmon": "绳文", "Ainu": "虾夷（阿伊努）", "Ainus": "虾夷（阿伊努）", "Hainan": "海南诸部",
    "Đại Việt": "大越", "Champa": "占城", "Chen-La": "真腊", "Khmer Empire": "吴哥（高棉）", "Cambodia": "柬埔寨",
    "Srivijaya Empire": "室利佛逝", "Mataram": "马打兰", "East Java": "东爪哇", "Dvaravati": "堕罗钵底", "Pyu state": "骠国",
    "Mon state": "孟人国", "Burmese": "缅人", "Laos": "老挝", "Thai": "傣泰诸部", "Philippines": "菲律宾诸岛",
    "Aceh": "亚齐", "Malays": "马来诸邦", "Austronesians": "南岛语族",
    "Chagatai Khanate": "察合台汗国", "Proto-Altaic pastoralists": "阿尔泰草原牧民", "Turcik tribes": "突厥诸部",
    # 南亚·中亚
    "Indus valley civilization": "印度河文明", "Vedic Aryans": "吠陀雅利安人", "Magadha": "摩揭陀", "Mauryan Empire": "孔雀王朝",
    "Kushan Empire": "贵霜", "Gupta Empire": "笈多王朝", "Palas": "波罗王朝", "Chola Empire": "朱罗王朝", "Chola": "朱罗",
    "Sultanate of Delhi": "德里苏丹国", "Rajput kingdoms": "拉其普特诸国", "Hindu kingdoms": "印度教诸国", "Simhala": "僧伽罗",
    "Sinhalese kingdoms": "僧伽罗诸国", "Kalinga": "羯陵伽", "Satavahanihara": "百乘王朝", "Nepal": "尼泊尔", "Bhutan": "不丹",
    "Bactria": "巴克特里亚", "Oxus": "阿姆河文明", "Saka Kingdom": "塞人王国", "Scythians": "斯基泰", "Saces": "塞人",
    "Ghaznavid Emirate": "伽色尼", "central Asian khanates": "中亚诸汗国", "Andronovo": "安德罗诺沃文化",
    # 西亚·北非
    "Egypt": "埃及", "Ur": "乌尔", "Elam": "埃兰", "Hittites": "赫梯", "Assyria": "亚述", "Babylonia": "巴比伦",
    "Achaemenid Empire": "阿契美尼德波斯", "Seleucid Kingdom": "塞琉古王国", "Ptolemaic Kingdom": "托勒密埃及",
    "Parthian Empire": "安息（帕提亚）", "Parthia": "安息", "Armenia": "亚美尼亚", "Urartu": "乌拉尔图", "Kush": "库什",
    "Meroe": "麦罗埃", "Kerma": "凯尔玛", "Axum": "阿克苏姆", "Saba": "萨巴", "Himyarite Kingdom": "希木叶尔",
    "Nabatean Kingdom": "纳巴泰", "Byzantine Empire": "拜占庭", "Fatimid Caliphate": "法蒂玛王朝",
    "Buwayhid Emirates": "白益王朝", "Mamluke Sultanate": "马穆鲁克", "Ottoman Empire": "奥斯曼帝国", "Arabs": "阿拉伯诸部",
    "Kingdom of David and Solomon": "以色列王国", "Carthaginian Empire": "迦太基", "Carthage": "迦太基", "Numidia": "努米底亚",
    "Mauretania": "毛里塔尼亚", "Berbers": "柏柏尔人", "Morocco": "摩洛哥", "Hafsid Caliphate": "哈夫斯王朝", "Yemen": "也门",
    "Georgia": "格鲁吉亚", "Muscat": "马斯喀特", "Ethiopia": "埃塞俄比亚", "Makkura": "马库里亚", "Alwa": "阿勒瓦",
    # 欧洲
    "Minoan": "米诺斯", "Greek city-states": "希腊城邦", "Macedon and Hellenic League": "马其顿", "Roman Republic": "罗马共和国",
    "Rome": "罗马", "Roman Empire": "罗马帝国", "Rome (Constantinus)": "罗马（君士坦提乌斯辖区）",
    "Rome (Maximian)": "罗马（马克西米安辖区）", "Rome (Galerius)": "罗马（伽列里乌斯辖区）",
    "Rome (Diocletianus)": "罗马（戴克里先辖区）", "Celts": "凯尔特人", "La Tène culture": "拉坦诺文化", "Etrurians": "伊特鲁里亚",
    "Thrace": "色雷斯", "Illyrians": "伊利里亚人", "Dacia": "达契亚", "Dacians": "达契亚人", "Sarmates": "萨尔马泰人",
    "Ostrogoths": "东哥特", "Visigothic Kingdom": "西哥特王国", "Avars": "阿瓦尔", "Bulgars": "保加尔人", "Khazars": "可萨汗国",
    "Carolingian Empire": "加洛林帝国", "Holy Roman Empire": "神圣罗马帝国", "Kingdom of France": "法兰西王国", "France": "法国",
    "England": "英格兰", "Scotland": "苏格兰", "United Kingdom": "英国", "Portugal": "葡萄牙", "Castilla": "卡斯蒂利亚",
    "Aragón": "阿拉贡", "Navarre": "纳瓦拉", "León": "莱昂", "Papal States": "教皇国", "Venice": "威尼斯", "Poland": "波兰",
    "Hungary": "匈牙利", "Sweden": "瑞典", "Norway": "挪威", "Denmark": "丹麦", "Denmark-Norway": "丹麦-挪威",
    "Grand Duchy of Moscow": "莫斯科大公国", "Prussia": "普鲁士", "Teutonic Knights": "条顿骑士团", "Magyars": "马扎尔人",
    "Volga Bulgars": "伏尔加保加尔", "Croatia": "克罗地亚", "Northmen": "北欧诺斯人", "Norsemen": "北欧诺斯人",
    # 非洲·美洲
    "Ghana": "加纳", "Empire of Ghana": "加纳帝国", "Mali": "马里", "Songhai": "桑海", "Kanem": "加涅姆", "Kanem-Bornu": "加涅姆-博尔努",
    "Great Zimbabwe": "大津巴布韦", "Congo": "刚果", "Benin": "贝宁", "Oyo": "奥约", "Hausa States": "豪萨诸邦", "Bantu": "班图人",
    "Bantu peoples": "班图诸族", "Madagascar": "马达加斯加", "Khoisan": "科伊桑人",
    "Olmec": "奥尔梅克", "Chavin": "查文", "Paracas": "帕拉卡斯", "Nazca": "纳斯卡", "Moche": "莫切", "Teotihuacan": "特奥蒂瓦坎",
    "Monte Albán": "阿尔班山（萨波特克）", "Maya chiefdoms and states": "玛雅诸邦", "Maya city-states": "玛雅城邦", "Maya states": "玛雅诸邦",
    "Huari Empire": "瓦里帝国", "Tiahuanaco Empire": "蒂瓦纳科", "Toltec Empire": "托尔特克", "Chimú Empire": "奇穆",
    "Mixtec Empire": "米斯特克", "Norte Chico": "小北文明", "Hopewell Culture": "霍普韦尔文化", "Polynesians": "波利尼西亚人",
    "Australian aboriginal hunter-gatherers": "澳洲原住民", "Maori": "毛利人",
}


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
for key in DYNASTY_ORDER:
    year = SNAPSHOT[key]
    caps = [Point(c["x"] / (K * S), -c["y"] / S) for c in DYNASTIES[key]["cities"] if c["tier"] == "capital"]
    feats = []
    for ft in fetch(year)["features"]:
        name = (ft["properties"] or {}).get("NAME")
        if not name or not ft["geometry"] or name == "?":
            continue
        g = make_valid(shape(ft["geometry"]))
        if g.geom_type == "GeometryCollection":
            g = g.buffer(0)
        is_china = any(g.contains(pt) for pt in caps) or name in EXTRA_CHINA.get(key, ())
        g = g.simplify(TOLERANCE, preserve_topology=True)
        if g.geom_type == "MultiPolygon" and not is_china:
            parts = [p for p in g.geoms if p.area >= MIN_AREA]
            if not parts:
                continue
            g = MultiPolygon(parts)
        elif g.area < MIN_AREA and not is_china:
            continue
        coords = to_latlng(g)
        if not coords:
            continue
        f = {"n": name, "g": coords}
        if name in ZH: f["z"] = ZH[name]
        if is_china: f["c"] = 1
        feats.append(f)
    # 本朝多边形画在最上面
    feats.sort(key=lambda f: f.get("c", 0))
    result[key] = {"year": year, "features": feats}
    china = [f["n"] for f in feats if f.get("c")]
    print(f"{DYNASTIES[key]['label']:<4} {year:>7}  {len(feats):>3} 个政区  本朝: {china}")

json.dump(result, open("borders_data.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("borders_data.json", os.path.getsize("borders_data.json") // 1024, "KB")

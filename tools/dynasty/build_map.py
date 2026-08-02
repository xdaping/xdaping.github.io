import json, math
from shapely.geometry import shape, box
from shapely.ops import unary_union, transform as shp_transform

REF_LAT = 35.0
K = math.cos(math.radians(REF_LAT))
SCALE = 9.0

def project(lon, lat):
    return (lon * K * SCALE, -lat * SCALE)
def project_geom(g):
    return shp_transform(lambda x, y, z=None: (x * K * SCALE, -y * SCALE), g)
def ring_to_path(coords):
    return "M" + "L".join(f"{x:.2f},{y:.2f}" for x, y in coords) + "Z"
def polygon_to_d(poly):
    d = ring_to_path(list(poly.exterior.coords))
    for interior in poly.interiors:
        d += ring_to_path(list(interior.coords))
    return d
def geom_to_d(g):
    if g.geom_type == "Polygon": return polygon_to_d(g)
    if g.geom_type == "MultiPolygon": return "".join(polygon_to_d(p) for p in g.geoms)
    return ""

# ---------- China outline ----------
china_raw = json.load(open('/tmp/china100000.json'))
geoms, provinces = [], []
for f in china_raw['features']:
    props = f['properties']
    if props.get('adcode') == '100000_JD': continue
    g = shape(f['geometry']).buffer(0)
    geoms.append(g); provinces.append((props.get('name'), g))
china_proj = project_geom(unary_union(geoms).simplify(0.02, preserve_topology=True))
china_d = geom_to_d(china_proj)
province_entries = [{"name": n, "d": geom_to_d(project_geom(g.simplify(0.08, preserve_topology=True)))} for n, g in provinces]

# ---------- Neighboring countries ----------
NEIGHBORS = {"Afghanistan","Bhutan","India","Japan","Kazakhstan","Kyrgyzstan","Laos",
             "Mongolia","Myanmar","Nepal","North Korea","Pakistan","Russia","South Korea",
             "Tajikistan","Vietnam"}
CLIP_BOX = box(68, 10, 142, 55)
ne = json.load(open('/tmp/ne110.geojson'))
n_geoms = []
for f in ne['features']:
    if f['properties'].get('NAME') in NEIGHBORS:
        g = shape(f['geometry']).buffer(0).intersection(CLIP_BOX)
        if not g.is_empty: n_geoms.append(g)
neighbors_union = unary_union(n_geoms).simplify(0.05, preserve_topology=True)
neighbors_d = geom_to_d(project_geom(neighbors_union))

# ---------- Rivers ----------
YELLOW_RIVER = [(34.9,97.0),(35.0,100.3),(36.06,103.79),(38.47,106.27),(40.6,109.8),(37.8,111.0),(34.6,110.3),(34.8,113.6),(36.65,115.0),(37.75,119.15)]
YANGTZE_RIVER = [(33.4,91.1),(33.0,97.0),(26.85,100.2),(26.58,101.72),(29.56,106.55),(30.7,111.3),(30.58,114.3),(29.7,116.0),(32.06,118.8),(31.4,121.9)]
def river_points(w): return [project(lon, lat) for lat, lon in w]
rivers = [
    {"name": "黄河（示意）", "points": river_points(YELLOW_RIVER)},
    {"name": "长江（示意）", "points": river_points(YANGTZE_RIVER)},
]
minx, miny, maxx, maxy = china_proj.bounds

# ---------- 地名数据（今地坐标 + 各朝地名）----------
from cities_data import COORDS, p, DYNASTIES, DYNASTY_ORDER

# ---------- Satellite / terrain basemap (NASA Blue Marble topo+bathy, public domain) ----------
from PIL import Image
import base64, io
SAT_LON0, SAT_LON1 = 70.0, 141.0
SAT_LAT0, SAT_LAT1 = 3.0, 54.0
bm = Image.open('bluemarble.jpg')          # 5400x2700 equirectangular (lon -180..180, lat 90..-90)
IW, IH = bm.size
def _px(lon): return (lon + 180) / 360 * IW
def _py(lat): return (90 - lat) / 180 * IH
crop = bm.crop((round(_px(SAT_LON0)), round(_py(SAT_LAT1)), round(_px(SAT_LON1)), round(_py(SAT_LAT0))))
tw = 1280
crop = crop.resize((tw, round(crop.height * tw / crop.width)), Image.LANCZOS)
buf = io.BytesIO(); crop.save(buf, format='JPEG', quality=84, optimize=True)
sat_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
xL = SAT_LON0 * K * SCALE; xR = SAT_LON1 * K * SCALE
yT = -SAT_LAT1 * SCALE;     yB = -SAT_LAT0 * SCALE
satellite = {"uri": sat_uri, "x": round(xL,2), "y": round(yT,2), "w": round(xR-xL,2), "h": round(yB-yT,2)}
print("sat crop", crop.size, "jpeg KB", round(len(buf.getvalue())/1024), "uri KB", round(len(sat_uri)/1024))

# ---------- overall bounds (all projected) ----------
npx = project_geom(neighbors_union).bounds  # projected neighbor bounds
all_x = [minx, maxx, npx[0], npx[2], satellite["x"], satellite["x"]+satellite["w"]]
all_y = [miny, maxy, npx[1], npx[3], satellite["y"], satellite["y"]+satellite["h"]]
for dyn in DYNASTIES.values():
    for c in dyn["cities"]:
        all_x.append(c["x"]); all_y.append(c["y"])
pad = 18
vb = [round(min(all_x)-pad,1), round(min(all_y)-pad,1), round(max(all_x)-min(all_x)+pad*2,1), round(max(all_y)-min(all_y)+pad*2,1)]

out = {
    "viewBox": vb, "chinaPath": china_d, "neighborsPath": neighbors_d,
    "provinces": province_entries, "rivers": rivers, "satellite": satellite,
    "dynastyOrder": DYNASTY_ORDER,
    "dynasties": DYNASTIES,
}
json.dump(out, open('map_data.json','w'), ensure_ascii=False)
print("num provinces", len(province_entries), "viewBox", vb)
for k in DYNASTY_ORDER:
    print(f"  {DYNASTIES[k]['label']}: {len(DYNASTIES[k]['cities'])} 城")
print("total json size", len(json.dumps(out, ensure_ascii=False)))

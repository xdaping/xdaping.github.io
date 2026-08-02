# 华夏舆图 · 历代疆域地名

一个交互式的中国历史地图可视化：左侧竖向时间轴选择朝代，右侧在 NASA 卫星地形底图上标注该朝代主要城市/地名。

## 直接使用

用浏览器打开 **`dynasty_map.html`** 即可（单文件、自包含、离线可用，卫星影像已内嵌）。

覆盖 17 个时期（按时间顺序）：夏、商、东周、春秋、战国、秦、西汉、东汉、三国、西晋、隋、唐、北宋、南宋、元、明、清，**共 2372 处地名，每朝 94–168 处**。

每朝的标注分四级：

| 级别 | 含义 | 样式 |
|---|---|---|
| `capital` | 都城 | 红点，最大字号 |
| `secondary` | 陪都、别都、行都、霸主之国 | 黄点 |
| `city` | 郡 / 州 / 府 / 路 / 行省 / 布政使司等正式治所 | 青点 |
| `town` | 关隘要塞、著名战场、经济文化名镇、考古遗址、羁縻与周边并存政权据点 | 小点，标注优先级最低 |

功能：

- 左侧时间轴切换朝代
- 卫星地形底图 + 中国轮廓高亮 + 周边区域压暗
- 地名字号恒定屏幕大小（如电子地图，缩放不改变字号）
- **标注随缩放动态重排**：按 都城 → 陪都 → 州府 → 关隘小邑 的优先级贪心占位，放不下的暂时隐去文字（圆点仍在，仍可悬浮/点击）；放大后空间变宽，被隐去的地名会陆续显现
- 悬浮/点击城市查看历史沿革
- 图层开关：卫星底图、周边压暗、江河示意、现代省界、地名文字
- 滚轮/双指缩放、拖动平移

## 文件说明

| 文件 | 作用 |
|---|---|
| `dynasty_map.html` | **最终成品**，可直接打开 |
| `template.html` | 页面模板（HTML/CSS/JS，含 `%%MAP_DATA%%` 占位符） |
| `cities_data.py` | **地名数据源**：`COORDS`（718 条今地经纬度）+ `DYNASTIES`（各朝地名表） |
| `build_map.py` | 完整构建脚本（重算地图路径、省界、江河、卫星底图） |
| `update_cities.py` | 只更新地名的快捷脚本（不动底图） |
| `map_data.json` | 构建产物（地图路径 + 各朝地名 + 卫星 data URI） |
| `bluemarble.jpg` | NASA Blue Marble 卫星原图（裁剪用，公有领域） |

## 重新生成

### 只改了地名或样式（常见情形）

`update_cities.py` 复用 `map_data.json` 里已有的地图路径与卫星底图，无需外部地理数据：

```bash
python3 update_cities.py     # 更新 map_data.json 并产出 dynasty_map.html
```

- **改地名/城市**：编辑 `cities_data.py` 中的 `COORDS`（今地经纬度）与 `DYNASTIES`（各朝地名列表），再跑上面这条命令。
- **改样式/交互**：编辑 `template.html`，再跑上面这条命令。

### 重建底图（改省界、江河、投影或卫星裁剪范围时）

`build_map.py` 需要两个外部地理数据文件：`/tmp/china100000.json`（阿里云 DataV 全国 GeoJSON）与 `/tmp/ne110.geojson`（Natural Earth 1:110m 国界）。

```bash
pip install Pillow shapely     # 首次
python3 build_map.py           # 生成 map_data.json
python3 - <<'PY'
t = open('template.html', encoding='utf-8').read()
d = open('map_data.json', encoding='utf-8').read()
open('dynasty_map.html', 'w', encoding='utf-8').write(t.replace('%%MAP_DATA%%', d))
PY
```

## 数据约定

- `COORDS[今地名] = (纬度, 经度)`，是古地名唯一允许的落点白名单。
- `p(古地名, 今地键, tier, 沿革注释)` 生成一个标注点；古地名中全角括号后的补充说明只出现在悬浮提示里，地图上只显示括号前的部分。
- **同一朝代内同一个今地键只能用一次**，否则两点完全重合、互相遮挡。改数据后可用下面这段自查：

```bash
python3 -c "
import collections, cities_data as C
for k, d in C.DYNASTIES.items():
    keys = [c['x'], ] and [(c['x'], c['y']) for c in d['cities']]
    names = [c['name'] for c in d['cities']]
    dupk = [x for x, v in collections.Counter(keys).items() if v > 1]
    dupn = [x for x, v in collections.Counter(names).items() if v > 1]
    if dupk or dupn: print(d['label'], '重合坐标', len(dupk), '重名', dupn)
print('检查完毕')
"
```

## 说明

地图边界与地名仅为历史地理教学示意，古地名以约略对应之今地标注（少数关隘、战场、遗址取邻近今地为落点），边界未随各朝逐年精绘，不代表现代国界或任何主权主张。并存政权（如辽、西夏、金、大理、吐蕃、渤海、高丽、交趾等）在相应朝代中一并标出，注释里已注明其归属。地理轮廓数据来自公开数据集（阿里云 DataV、Natural Earth），底图为 NASA Blue Marble（公有领域）。

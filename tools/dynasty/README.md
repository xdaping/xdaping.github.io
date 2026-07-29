# 华夏舆图 · 历代疆域地名

一个交互式的中国历史地图可视化：左侧竖向时间轴选择朝代，右侧在 NASA 卫星地形底图上标注该朝代主要城市/地名。

## 直接使用

用浏览器打开 **`dynasty_map.html`** 即可（单文件、自包含、离线可用，卫星影像已内嵌）。

覆盖 17 个时期（按时间顺序）：夏、商、东周、春秋、战国、秦、西汉、东汉、三国、西晋、隋、唐、北宋、南宋、元、明、清。每个时期标注 10–102 处都邑 / 诸侯国都 / 州郡府治。

功能：
- 左侧时间轴切换朝代
- 卫星地形底图 + 中国轮廓高亮 + 周边区域压暗
- 地名字号恒定屏幕大小（如电子地图，缩放不变，放大自动散开解决拥挤）
- 悬浮/点击城市查看历史沿革
- 图层开关：卫星底图、周边压暗、江河示意、现代省界、地名文字
- 滚轮/双指缩放、拖动平移

## 文件说明

| 文件 | 作用 |
|---|---|
| `dynasty_map.html` | **最终成品**，可直接打开 |
| `template.html` | 页面模板（HTML/CSS/JS，含 `%%MAP_DATA%%` 占位符） |
| `build_map.py` | 数据构建脚本 |
| `map_data.json` | 构建产物（地图路径 + 九朝地名 + 卫星 data URI） |
| `bluemarble.jpg` | NASA Blue Marble 卫星原图（裁剪用，公有领域） |

## 重新生成

修改地名数据或样式后重新构建：

```bash
# 1. 安装依赖（首次）
pip install Pillow shapely

# 2. 生成 map_data.json（读取 bluemarble.jpg 裁剪卫星底图）
python3 build_map.py

# 3. 把数据注入模板，产出 dynasty_map.html
python3 - <<'PY'
t = open('template.html', encoding='utf-8').read()
d = open('map_data.json', encoding='utf-8').read()
open('dynasty_map.html', 'w', encoding='utf-8').write(t.replace('%%MAP_DATA%%', d))
PY
```

- **改地名/城市**：编辑 `build_map.py` 中的 `COORDS`（今地经纬度）与 `DYNASTIES`（各朝城市列表），重新执行上面 2、3 步。
- **改样式/交互**：直接编辑 `template.html`，重新执行第 3 步。

## 说明

地图边界与地名仅为历史地理教学示意，古地名以约略对应之今地标注，边界未随各朝逐年精绘，不代表现代国界或任何主权主张。地理轮廓数据来自公开数据集（阿里云 DataV、Natural Earth），底图为 NASA Blue Marble（公有领域）。

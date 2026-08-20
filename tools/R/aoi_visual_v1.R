library(shiny)
library(leaflet)
library(leaflet.extras)   # 测量/全屏/复位控件
library(readr)
library(dplyr)
library(DT)
library(sf)
library(stringr)

options(shiny.maxRequestSize = 1024 * 1024^2)

setwd("/Users/daping/xiongdaping/WorkSpaces/RstudioProject")

DEFAULT_FILE <- "aoi_data_p_100.txt"

# WKT -> sfc，解析失败或为空一律返回 NULL
wkt_to_sfc <- function(w) {
  if (is.null(w) || length(w) != 1 || is.na(w) || !nzchar(trimws(w))) return(NULL)
  g <- tryCatch(sf::st_as_sfc(w, crs = 4326), error = function(e) NULL)
  if (is.null(g) || length(g) == 0 || isTRUE(sf::st_is_empty(g)[1])) return(NULL)
  g
}

# ---------------- UI ----------------
ui <- fluidPage(
  fileInput("segment_file", "TSV文件", buttonLabel = "浏览",
            placeholder = "没有文件被选中", accept = c(".txt", ".tsv"), width = 1200),
  hr(),
  div(style = "color:#666;margin-bottom:8px;", textOutput("data_status")),
  fluidRow(
    column(4, DT::DTOutput("dt")),
    column(8, leafletOutput("segment_plot", height = 600))
  )
)

# ---------------- Server ----------------
server <- function(input, output, session) {
  
  # 数据源：上传了就用上传的，否则用默认文件
  data_path <- reactive({
    if (!is.null(input$segment_file)) input$segment_file$datapath else DEFAULT_FILE
  })
  
  # 文件格式：\t 分隔，列为 aoi_id / aoi_name / fix / polygon_str
  segment_trace <- reactive({
    path <- data_path()
    validate(need(file.exists(path), paste0("找不到文件：", path)))
    txt <- readLines(path, warn = FALSE)          # 绕开末行无换行的问题
    validate(need(length(txt) > 1, "文件里只有表头，没有数据行"))
    
    df <- read_tsv(I(txt), col_types = cols(.default = col_character()), progress = FALSE)
    if (!"fix" %in% names(df)) df$fix <- "0"
    df %>% mutate(polygon_str = str_trim(polygon_str))
  })
  
  output$data_status <- renderText({
    paste0("数据文件：", data_path(), " | 行数：", nrow(segment_trace()))
  })
  
  selected_row <- reactive({
    df  <- segment_trace()
    rid <- input$dt_rows_selected
    if (is.null(rid) || length(rid) == 0 || rid > nrow(df)) rid <- 1
    df %>% slice(rid)
  })
  
  current_sfc <- reactive({
    g <- wkt_to_sfc(selected_row()$polygon_str[1])
    validate(need(!is.null(g), "该行的 polygon_str 无法解析为有效几何"))
    g
  })
  
  output$dt <- DT::renderDT({
    DT::datatable(
      segment_trace() %>% select(aoi_id, aoi_name, fix),
      selection = list(mode = "single", selected = 1),
      filter = "top",
      options = list(stateSave = TRUE)
    )
  })
  
  output$segment_plot <- renderLeaflet({
    g  <- current_sfc()
    bb <- sf::st_bbox(g)
    
    leaflet() %>%
      addTiles('https://mt.google.com/vt/lyrs=s&hl=zh-CN&gl=cn&x={x}&y={y}&z={z}',
               options = tileOptions(tileSize = 256, minZoom = 3, maxZoom = 21),
               attribution = '&copy; <a href="https://mt.google.com/">谷歌卫星</a>',
               group = '谷歌卫星') %>%
      addTiles('https://mt.google.com/vt/lyrs=m&hl=zh-CN&gl=cn&x={x}&y={y}&z={z}',
               options = tileOptions(tileSize = 256, minZoom = 3, maxZoom = 21),
               attribution = '&copy; <a href="https://mt.google.com/">谷歌路网</a>',
               group = '谷歌路网') %>%
      addPolygons(data = g, group = "raw_polygon",
                  color = "red", fill = TRUE, fillOpacity = 0.2) %>%
      addLayersControl(
        baseGroups = c('谷歌卫星', '谷歌路网'),
        overlayGroups = c("raw_polygon"),
        options = layersControlOptions(collapsed = FALSE)
      ) %>%
      # 右侧：测量工具（点两下量距离，闭合成面量面积）
      addMeasure(
        position = "topright",
        primaryLengthUnit = "meters",   secondaryLengthUnit = "feet",
        primaryAreaUnit   = "sqmeters", secondaryAreaUnit   = "acres",
        activeColor = "#f2a12b", completedColor = "#d9541e",
        localization = "zh"
      ) %>%
      addFullscreenControl(position = "topright") %>%
      addResetMapButton() %>%
      fitBounds(as.numeric(bb["xmin"]), as.numeric(bb["ymin"]),
                as.numeric(bb["xmax"]), as.numeric(bb["ymax"]))
  })
}

shinyApp(ui = ui, server = server)
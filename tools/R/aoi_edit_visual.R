
library(shiny)
library(shinyjs)
library(leaflet)
library(leaflet.extras)  # 添加leaflet扩展包
library(readr)
library(dplyr)
library(tidyr)
library(splitstackshape)
library(DT)
library(sf)  # 替代空间数据处理
library(terra)  # 替代栅格数据处理
library(stringr)

options(shiny.maxRequestSize=1024*1024^2) 

setwd("/Users/daping/xiongdaping/WorkSpaces/RstudioProject")

# 定义配色方案 ----
ui <- fluidPage(
  useShinyjs(),
  #helpText("文件"),
  #fileInput(inputId="segment_file", label="TSV文件", buttonLabel="浏览", placeholder="没有文件被选中", width = 1200),
  hr(),
  fluidRow(
    column(4,dataTableOutput("dt",height = 600)),
    column(8,leafletOutput("segment_plot", height = 600))
  ),
  textOutput("fix_p"),
  
  div(
    style = "margin-top: 20px;",
    textInput("save_path", "保存路径", value = getwd(), width = "100%"),
    actionButton("save_polygon", "保存修改后的polygon", class = "btn-primary")
  )
)

# 服务端功能 ----
server <- function(input, output, session) {
  # 在server函数顶部添加表格状态存储
  table_state <- reactiveValues(
    page = 1,
    search = "",
    order = NULL
  )
  
  # 新增：文件更新触发器 ----
  file_update_trigger <- reactiveVal(0)
  
  # 新增：存储保存时的行ID ----
  last_save_row_id <- reactiveVal(1)  # 默认初始值为1
  
  # 创建dataTable代理
  dt_proxy <- dataTableProxy('dt')
  
  # 存储当前选中的多边形数据
  current_polygon <- reactiveVal(NULL)
  
  # 获取当前工作目录
  observe({
    updateTextInput(session, "save_path", value = getwd())
  })
  
  segment_trace <- reactive({
    # 添加对触发器的依赖
    file_update_trigger()
    
    
    "
    # 文件格式，以\t为分隔符，最后一行要换行
    aoi_id	aoi_name	fix	polygon_str	fix_polygon_str
    1	测试	1	POLYGON((-118.36 33.82,-118.37 33.82,-118.37 33.83,-118.36 33.82))
    "
    segment_trace <- read_tsv("aoi_data_p_100.txt") %>%
      mutate(
        polygon_str = str_trim(polygon_str),  # 去除可能的空白字符
        fix_polygon_str = ifelse(!is.na(fix_polygon_str), str_trim(fix_polygon_str), NA)  # 处理fix_polygon_str
      )
    return(segment_trace)
  })
  
  
  # 修改has_changes响应式表达式
  has_changes <- reactive({
    # 确保关键依赖项就绪
    req(
      current_polygon(), 
      segment_trace(),
      cancelOutput = TRUE  # 当依赖项缺失时静默返回
    )
    
    tryCatch({
      # 获取当前行ID（带默认值处理）
      row_id <- if (is.null(input$dt_row_last_clicked)) 1 else input$dt_row_last_clicked
      
      # 安全获取原始数据
      original_data <- tryCatch(
        segment_trace() %>% slice(row_id),
        error = function(e) {
          message("原始数据获取失败：", e$message)
          return(NULL)
        }
      )
      if (is.null(original_data)) return(FALSE)
      
      # 将原始数据转换为SF对象
      original_polygon_sf <- original_data %>%
        sf::st_as_sf(wkt="polygon_str") %>%
        sf::st_set_crs(4326)  %>%
        sf::st_cast('POLYGON')
      
      print("=======")
      
      # 检查原始多边形是否有足够的点
      coords <- st_coordinates(original_polygon_sf)
      if (nrow(coords) < 4) {
        message("原始多边形点数不足4个")
        return(FALSE)
      }
      
      print("原始多边形数据：")
      print(head(coords))
      print("原始多边形行数：")
      print(nrow(coords))
      
      print("当前多边形数据：")
      print(head(current_polygon()))
      print("当前多边形行数：")
      print(nrow(current_polygon()))
      
      # 将当前多边形转换为SF对象
      current_polygon_sf <- current_polygon() %>%
        st_as_sf(coords = c("lng", "lat"), crs = 4326) %>%
        summarise(geometry = st_combine(geometry)) %>%
        st_cast("POLYGON")
      
      print(original_polygon_sf)
      print(current_polygon_sf)
      
      # 计算两个多边形的面积
      original_area <- st_area(original_polygon_sf)
      current_area <- st_area(current_polygon_sf)
      
      # 计算交集面积
      intersection_area <- st_area(st_intersection(original_polygon_sf, current_polygon_sf))
      
      # 计算并集面积
      union_area <- st_area(st_union(original_polygon_sf, current_polygon_sf))
      
      # 计算IoU
      iou <- as.numeric(intersection_area / union_area)
      
      # 如果IoU小于0.99，则认为有变化
      iou < 0.99
      
    }, error = function(e) {
      message("比较过程出错：", e$message)
      return(FALSE)  # 确保返回逻辑值
    })
  })
  
  # 修改按钮状态控制逻辑
  observe({
    tryCatch({
      # 强制转换为逻辑值
      changes_exist <- isTRUE(has_changes())
      
      if (changes_exist) {
        shinyjs::enable("save_polygon")
      } else {
        shinyjs::disable("save_polygon")
      }
    }, error = function(e) {
      shinyjs::disable("save_polygon")
      message("按钮状态控制出错：", conditionMessage(e))
    })
  })
  
  # 新增：当多边形数据变化时重新验证
  observeEvent(current_polygon(), {
    has_changes()  # 主动触发验证
  })
  
  
  output$dt <- renderDataTable({
    segment_df <- segment_trace()  %>% select(aoi_id,aoi_name,fix)
    datatable(segment_df, selection="single", filter="top", 
              options = list(
                stateSave = TRUE,
                displayStart = table_state$page - 1,
                search = list(search = table_state$search),
                order = table_state$order
              ))# 启用状态保存
  })
  
  output$segment_plot <- renderLeaflet({
    row_id <- input$dt_row_last_clicked
    if(is.null(row_id)) row_id <- 1
    target_data <- segment_trace()
    
    selected_row <- target_data %>% slice(row_id)
    
    # 修改多边形数据的处理方式
    raw_polygon_sf <- selected_row %>%
      sf::st_as_sf(wkt="polygon_str") %>%
      sf::st_set_crs(4326) %>%
      sf::st_cast('POLYGON')
    
    # 存储当前多边形数据
    current_polygon_data <- raw_polygon_sf %>%
      st_coordinates() %>%
      as_tibble() %>%
      rename(lng = X, lat = Y)
    
    current_polygon(current_polygon_data)
    
    m <- leaflet() %>% 
      addTiles(
        'https://mt.google.com/vt/lyrs=s&hl=zh-CN&gl=cn&x={x}&y={y}&z={z}',
        options = tileOptions(tileSize=256, minZoom=3, maxZoom=21),
        attribution = '&copy; <a href="https://mt.google.com/">谷歌卫星</a>',
        group='谷歌卫星'
      ) %>%
      addTiles(
        'https://mt.google.com/vt/lyrs=m&hl=zh-CN&gl=cn&x={x}&y={y}&z={z}',
        options = tileOptions(tileSize=256, minZoom=3, maxZoom=21),
        attribution = '&copy; <a href="https://mt.google.com/">谷歌路网</a>',
        group='谷歌路网'
      )
    
    # 处理fix_polygon_str
    if (!is.na(selected_row$fix_polygon_str) && nzchar(trimws(selected_row$fix_polygon_str))) {
      fix_polygon_sf <- selected_row %>%
        sf::st_as_sf(wkt="fix_polygon_str") %>%
        sf::st_set_crs(4326) %>%
        sf::st_cast('POLYGON')
    } else {
      fix_polygon_sf <- raw_polygon_sf
    }
    
    m <- m %>%
      addPolygons(data = raw_polygon_sf, color = "blue", fill = TRUE, fillOpacity = 0.2, group = "editable_polygon") %>%
      addPolygons(data = fix_polygon_sf, color = "red", fill = TRUE, fillOpacity = 0.2, group = "good_polygon") %>%
      addLayersControl(
        baseGroups = c('谷歌卫星', '谷歌路网'),
        overlayGroups = c("editable_polygon","good_polygon"),
        options = layersControlOptions(collapsed = FALSE)
      ) %>%
      hideGroup(c()) %>%
      addDrawToolbar(
        targetGroup = "editable_polygon",
        editOptions = editToolbarOptions(
          selectedPathOptions = selectedPathOptions(),
          remove = TRUE,  # 启用删除功能
          edit = TRUE,    # 启用编辑功能
          allowIntersection = TRUE  # 允许路径相交
        ),
        position = "topright"
      )
    
    m
  })
  
  # 监听多边形编辑事件
  observeEvent(input$segment_plot_draw_edited_features, {
    print("检测到编辑事件")
    edited_polygon <- input$segment_plot_draw_edited_features
    if (!is.null(edited_polygon)) {
      tryCatch({
        # 提取多边形坐标
        print("提取多边形坐标")
        coords <- edited_polygon$features[[1]]$geometry$coordinates[[1]]  # 注意这里多了一层坐标
        print(coords)
        print("多边形坐标数量")
        print(length(coords))
        
        # 提取坐标并创建数据框，确保数据格式正确
        edited_df <- tibble(
          lng = unlist(sapply(coords, function(x) x[1])),
          lat = unlist(sapply(coords, function(x) x[2]))
        )
        print(edited_df)
        print(nrow(edited_df))
        
        current_polygon(edited_df)
        print("正确提取的坐标：")
        print(head(edited_df))
        
      }, error = function(e) {
        showNotification(paste("坐标提取失败:", e$message), type = "error")
        print(paste("DEBUG - 错误详情:", e$message))
      })
    }
  })
  
  output$fix_p <- renderText({
    tryCatch({
      if (is.null(current_polygon())) {
        return("")
      }
      
      # 将修改后的多边形坐标转换为WKT格式
      modified_polygon <- current_polygon() %>%
        st_as_sf(coords = c("lng", "lat"), crs = 4326) %>%
        st_geometry() %>%  # 只保留几何信息
        st_combine() %>%   # 组合几何
        st_cast("POLYGON") %>%
        st_as_text()
      
      return(modified_polygon)
    }, error = function(e) {
      message("多边形坐标转换错误：", e$message)
      return("")
    })
  })
  
  
  
  # 保存修改后的多边形
  observeEvent(input$save_polygon, {
    if (!is.null(current_polygon())) {
      print("开始保存多边形")
      
      row_id <- input$dt_row_last_clicked
      if(is.null(row_id)) row_id <- 1
      last_save_row_id(row_id)
      
      original_data <- segment_trace() %>% slice(row_id)
      
      # 将修改后的多边形坐标转换为WKT格式
      modified_polygon <- current_polygon() %>%
        st_as_sf(coords = c("lng", "lat"), crs = 4326) %>%
        st_geometry() %>%  # 只保留几何信息
        st_combine() %>%   # 组合几何
        st_cast("POLYGON") %>%
        st_as_text()
      
      print("修改后的多边形字符串：")
      print(modified_polygon)
      
      # 准备保存的数据
      save_data <- data.frame(
        aoi_id = original_data$aoi_id,
        aoi_name = original_data$aoi_name,
        polygon_str = original_data$polygon_str,
        fix_polygon_str = modified_polygon
      )
      
      # 构建完整的文件路径
      save_dir <- input$save_path
      if (!dir.exists(save_dir)) {
        dir.create(save_dir, recursive = TRUE)
      }
      
      #file_name <- paste0("modified_polygon_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
      full_path <- file.path(save_dir, "modified_polygons.csv")
      
      # 判断文件是否存在
      file_exists <- file.exists(full_path)
      
      # 保存到文件
      #write.csv(save_data, file = full_path, row.names = FALSE)
      # 写入文件（追加模式）
      write.table(
        save_data,
        file = full_path,
        sep = ",",
        append = file_exists,  # 存在则追加
        col.names = !file_exists,  # 不存在时写入列名
        row.names = FALSE,
        quote = TRUE
      )
      showNotification(paste("polygon已保存到文件：", full_path), type = "message")
      
      
      
      # 更新原始数据文件中的fix字段
      original_file <- "aoi_data_p_100.txt"
      original_data_path <- "aoi_data_p_100.txt"  # 直接使用本地文件路径
      
      # 读取原始数据
      original_data <- read_tsv(original_data_path)
      
      # 更新polygon_str
      original_data[row_id, "fix_polygon_str"] <- modified_polygon  # 更新修复后的多边形
      original_data[row_id, "fix"] <- 1  # 标记为已修复
      
      # 构建保存路径
      save_dir <- input$save_path
      original_file_path <- file.path(save_dir, original_file)
      
      # 确保目录存在
      if (!dir.exists(save_dir)) {
        dir.create(save_dir, recursive = TRUE)
      }
      
      # 写入更新后的数据（保留列名）
      write_tsv(original_data, original_file_path)
      
      showNotification(paste("原始数据文件已更新：", original_file_path), type = "message")
      
      # 使用代理更新数据（保持当前状态）
      segment_df <- segment_trace()  %>% select(aoi_id,aoi_name,fix)
      replaceData(dt_proxy, 
                  data = segment_df,
                  resetPaging = FALSE,  # 保持分页状态
                  clearSelection = "none")  # 保持选中状态
      
      # 触发数据刷新 ----
      file_update_trigger(file_update_trigger() + 1)  # 递增触发器值
      
      # 显式保持表格状态
      isolate({
        if (!is.null(input$dt_state)) {
          table_state$page <- input$dt_state$start + 1
          table_state$search <- input$dt_state$search$search
          table_state$order <- input$dt_state$order
        }
      })
    }
  })
}

# 运行App
shinyApp(ui = ui, server = server)
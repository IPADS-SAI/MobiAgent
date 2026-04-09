@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ==========================================================
:: Auto Search 运行参数模板 (Windows .bat 版本)
:: 使用方式：直接双击运行，或在 CMD/PowerShell 中执行此文件
:: ==========================================================

:: 基础参数配置
set APP_NAME=美团
set DEPTH=2
set BREADTH=10
set DEVICE=Android
set SERVICE_IP=166.111.53.96
set DECIDER_PORT=7003
set SJTU_BASE_URL=https://models.sjtu.edu.cn/api/v1
set SJTU_API_KEY=sk-7PSBcDuOjYh9VPOK05-e_w
set SJTU_MODEL=qwen3vl
set EXPLORER_MODEL=%SJTU_MODEL%
set OPENROUTER_BASE_URL=%SJTU_BASE_URL%
set OPENROUTER_API_KEY=%SJTU_API_KEY%
set DECIDER_BASE_URL=%SJTU_BASE_URL%
set DECIDER_API_KEY=%SJTU_API_KEY%
set DECIDER_MODEL=%SJTU_MODEL%

:: 运行模式配置
set USE_QWEN3=on
set DATA_DIR=
set ALLOW_HIERARCHY_TEXT_DECIDER=off
set ENABLE_UI_SEMANTIC_COLLECT=on

:: 页面加载等待（加载慢的 App 可调大）
:: PAGE_LOAD_WAIT_SEC:          动作后固定等待秒数
:: PAGE_LOAD_STABLE_MAX_POLLS:  最多再轮询几次（每次0.5s）等 hierarchy 稳定
set PAGE_LOAD_WAIT_SEC=1.5
set PAGE_LOAD_STABLE_MAX_POLLS=6

:: BBox 精炼阈值（换模型/换手机时调整）
:: BBOX_IOU_THRESHOLD:      IoU >= 此值则用 XML 元素边框（模型越不准确 → 调低，如 0.1）
:: BBOX_CENTER_DIST_RATIO:  中心距/对角线 <= 此值才匹配（偏差大 → 调高，如 0.15）
:: BBOX_AREA_RATIO_MIN/MAX: 候选元素面积/模型bbox面积的允许范围
set BBOX_IOU_THRESHOLD=0.1
set BBOX_CENTER_DIST_RATIO=0.15
set BBOX_AREA_RATIO_MIN=0.3
set BBOX_AREA_RATIO_MAX=3.0

:: 弹窗自动关闭（Explorer VLM 检测到广告/弹窗时自动点关闭按钮）
:: POPUP_DISMISS_MAX_ATTEMPTS: 最多尝试几次（0 表示禁用，建议 2）
set POPUP_DISMISS_MAX_ATTEMPTS=2

:: UI 采集模块配置
set UI_COLLECT_ASYNC=on
set UI_COLLECT_QUEUE_SIZE=8
set UI_COLLECT_DRAIN_ON_EXIT=on
set UI_COLLECT_DRAIN_TIMEOUT_SEC=180
set UI_COLLECT_USE_VLM=on
set UI_COLLECT_VLM_TEXT_ONLY=off
set UI_COLLECT_VLM_MODEL=qwen/qwen3-vl-30b-a3b-instruct
set UI_COLLECT_BASE_URL=%OPENROUTER_BASE_URL%
set UI_COLLECT_API_KEY=%OPENROUTER_API_KEY%
set UI_COLLECT_MAX_ITEMS=32
set UI_COLLECT_MAX_VLM_CALLS=12
set UI_COLLECT_MIN_AREA=16
set PYTHONPATH=%CD%;%PYTHONPATH%

echo Running auto-search with app=%APP_NAME% depth=%DEPTH% breadth=%BREADTH% ...

:: 构建命令字符串
set CMD=python -m runner.mobiagent.auto-search ^
 --app_name "%APP_NAME%" ^
 --depth "%DEPTH%" ^
 --breadth "%BREADTH%" ^
 --device "%DEVICE%" ^
 --service_ip "%SERVICE_IP%" ^
 --decider_port "%DECIDER_PORT%" ^
 --decider_base_url "%DECIDER_BASE_URL%" ^
 --decider_api_key "%DECIDER_API_KEY%" ^
 --decider_model "%DECIDER_MODEL%" ^
 --openrouter_base_url "%OPENROUTER_BASE_URL%" ^
 --openrouter_api_key "%OPENROUTER_API_KEY%" ^
 --explorer_model "%EXPLORER_MODEL%" ^
 --use_qwen3 "%USE_QWEN3%" ^
 --allow_hierarchy_text_decider "%ALLOW_HIERARCHY_TEXT_DECIDER%" ^
 --enable_ui_semantic_collect "%ENABLE_UI_SEMANTIC_COLLECT%" ^
 --ui_collect_async "%UI_COLLECT_ASYNC%" ^
 --ui_collect_queue_size "%UI_COLLECT_QUEUE_SIZE%" ^
 --ui_collect_drain_on_exit "%UI_COLLECT_DRAIN_ON_EXIT%" ^
 --ui_collect_drain_timeout_sec "%UI_COLLECT_DRAIN_TIMEOUT_SEC%" ^
 --ui_collect_use_vlm "%UI_COLLECT_USE_VLM%" ^
 --ui_collect_vlm_text_only "%UI_COLLECT_VLM_TEXT_ONLY%" ^
 --ui_collect_vlm_model "%UI_COLLECT_VLM_MODEL%" ^
 --ui_collect_base_url "%UI_COLLECT_BASE_URL%" ^
 --ui_collect_api_key "%UI_COLLECT_API_KEY%" ^
 --ui_collect_max_items "%UI_COLLECT_MAX_ITEMS%" ^
 --ui_collect_max_vlm_calls "%UI_COLLECT_MAX_VLM_CALLS%" ^
 --ui_collect_min_area "%UI_COLLECT_MIN_AREA%" ^
 --page_load_wait_sec "%PAGE_LOAD_WAIT_SEC%" ^
 --page_load_stable_max_polls "%PAGE_LOAD_STABLE_MAX_POLLS%" ^
 --bbox_iou_threshold "%BBOX_IOU_THRESHOLD%" ^
 --bbox_center_dist_ratio "%BBOX_CENTER_DIST_RATIO%" ^
 --bbox_area_ratio_min "%BBOX_AREA_RATIO_MIN%" ^
 --bbox_area_ratio_max "%BBOX_AREA_RATIO_MAX%" ^
 --popup_dismiss_max_attempts "%POPUP_DISMISS_MAX_ATTEMPTS%"

:: 如果 DATA_DIR 不为空，则添加参数
if not "%DATA_DIR%"=="" set CMD=%CMD% --data_dir "%DATA_DIR%"

:: 执行命令
%CMD%

paused:\cdl\code\MobiAgent\MobiAgent\runner
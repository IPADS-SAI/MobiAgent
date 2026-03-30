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
set BREADTH=2
set DEVICE=Android
set SERVICE_IP=166.111.53.96
set DECIDER_PORT=7003
set EXPLORER_MODEL=qwen3-30b-a3b-instruct
set OPENROUTER_BASE_URL=http://166.111.53.96:7002/v1
set OPENROUTER_API_KEY=local

:: 运行模式配置
set USE_QWEN3=on
set DATA_DIR=
set ALLOW_HIERARCHY_TEXT_DECIDER=off
set ENABLE_UI_SEMANTIC_COLLECT=on

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
 --ui_collect_min_area "%UI_COLLECT_MIN_AREA%"

:: 如果 DATA_DIR 不为空，则添加参数
if not "%DATA_DIR%"=="" set CMD=%CMD% --data_dir "%DATA_DIR%"

:: 执行命令
%CMD%

pause
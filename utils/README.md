# 工具重构说明

## 概览

本次重构将原本分散在 `MobiFlow/tools/` 目录下的OCR引擎和图标检测工具代码统一整理到 `utils/` 目录下，并将模型权重文件统一管理到 `weights/` 目录。

## 主要变化

### 1. 代码迁移

#### OCR引擎
- **原路径**: `MobiFlow/tools/app_trajectory_analyzer/src/analyzer/ocr_engine.py`
- **新路径**: `utils/ocr_engine.py`
- **改进**:
  - 简化了导入依赖
  - 优化了日志系统集成
  - 统一了错误处理
  - 支持配置管理

#### 图标检测
- **原路径**: `MobiFlow/tools/Icon_detection/`
- **新路径**: `utils/icon_detection.py`
- **改进**:
  - 整合了多个模块到单个文件
  - 统一了接口设计
  - 自动路径查找功能

### 2. 新增模块

#### 统一配置管理 (`utils/config.py`)
- 提供 OCR 和图标检测的统一配置管理
- 支持从文件、环境变量加载配置
- 提供配置验证和默认值

#### 权重管理器 (`utils/weights_manager.py`)
- 统一管理各种模型权重文件
- 提供模型路径解析和验证
- 支持动态添加新模型

#### 工具入口 (`utils/tools.py`)
- 提供统一的工具导入接口
- 简化外部调用

### 3. 权重文件迁移

#### 迁移的模型
- **OwlViT模型**: `MobiFlow/tools/app_trajectory_analyzer/owlvit-base-patch32` → `weights/owlvit-base-patch32`

#### 现有模型
- `weights/icon_detect/` - 图标检测模型
- `weights/icon_caption_florence/` - 图标标注模型

### 4. 导入路径更新

#### 旧的导入方式
```python
# OCR引擎
from tools.app_trajectory_analyzer.src.analyzer.ocr_engine import OCREngine

# 图标检测
from tools.Icon_detection import get_icon_detection_service
```

#### 新的导入方式
```python
# 通过utils.tools统一导入
from utils.tools import OCREngine, get_icon_detection_service

# 或直接导入
from utils.ocr_engine import OCREngine
from utils.icon_detection import get_icon_detection_service
```

## 使用方法

### OCR引擎使用示例

```python
from utils.tools import OCREngine, ocr_image

# 方式1: 使用便捷函数
result = ocr_image("path/to/image.png")
print(result.get_text())

# 方式2: 使用引擎实例
engine = OCREngine()
result = engine.run("path/to/image.png")
for word in result.words:
    print(f"文字: {word.text}, 位置: {word.bbox}, 置信度: {word.conf}")
```

### 图标检测使用示例

```python
from utils.tools import detect_single_icon, get_icon_detection_service

# 方式1: 简单检测
found = detect_single_icon("screenshot.png", "search_icon")

# 方式2: 详细检测
service = get_icon_detection_service()
result = service.detect_icons(
    image="screenshot.png",
    icon_names=["search_icon", "menu_icon"],
    match_mode='any'
)
print(f"检测成功: {result['success']}")
print(f"匹配的图标: {result['matched_icons']}")
```

### 配置管理示例

```python
from utils.tools import get_config_manager

# 获取配置管理器
config_manager = get_config_manager()

# 更新OCR配置
config_manager.update_ocr_config(use_paddle=True)

# 添加图标搜索路径
config_manager.add_icon_path("/path/to/custom/icons")

# 保存配置
config_manager.save_config()
```

### 权重管理示例

```python
from utils.weights_manager import get_weights_manager, get_model_path

# 获取模型路径
icon_model_path = get_model_path("icon_detect")
owlvit_path = get_model_path("owlvit_base_patch32")

# 验证所有模型
manager = get_weights_manager()
validation = manager.validate_models()
for model_name, exists in validation.items():
    print(f"{model_name}: {'✓' if exists else '✗'}")
```

## 兼容性

- 保持了原有API的兼容性
- 支持渐进式迁移
- 提供了兼容性函数和别名

## 配置文件

重构后会生成配置文件 `config/tools_config.json`，包含所有工具的配置参数：

```json
{
  "ocr": {
    "lang": "chi_sim+eng",
    "use_paddle": null,
    "paddle_config": {},
    "tesseract_config": {}
  },
  "icon_detection": {
    "icon_base_paths": ["/path/to/task_configs/icons"],
    "default_threshold": 0.8,
    "scale_range": [0.5, 2.0],
    "scale_step": 0.1,
    "nms_threshold": 0.3
  },
  "weights_dir": "/path/to/weights"
}
```

## 环境变量支持

可通过环境变量配置：

```bash
export OCR_LANG="chi_sim+eng"
export OCR_USE_PADDLE="true"
export ICON_BASE_PATHS="/path1:/path2"
export ICON_DEFAULT_THRESHOLD="0.8"
export WEIGHTS_DIR="/path/to/weights"
```

## 迁移建议

1. **逐步迁移**: 建议先更新新代码使用新接口，旧代码保持不变
2. **测试验证**: 确认新接口功能正常后再删除旧代码
3. **配置同步**: 将现有配置迁移到新的配置管理系统
4. **路径检查**: 确认权重文件路径正确，模型能正常加载
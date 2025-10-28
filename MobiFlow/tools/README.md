# Tools 目录说明

## 重构通知

原本位于此目录的工具代码已经重构并迁移：

- **OCR引擎**: `app_trajectory_analyzer/` → `../../utils/ocr_engine.py`
- **图标检测**: `Icon_detection/` → `../../utils/icon_detection.py`

## 新的使用方式

```python
# 新的导入方式
from utils.tools import OCREngine, get_icon_detection_service
from utils.tools import ocr_image, detect_single_icon

# 统一配置管理
from utils.tools import get_config_manager, get_weights_dir
```

## 备份

原始代码已备份到 `tools_backup_YYYYMMDD_HHMMSS/` 目录。

## 详细信息

请查看 `../../utils/README.md` 了解详细的重构说明和使用方法。
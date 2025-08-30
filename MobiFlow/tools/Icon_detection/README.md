# 图标检测工具

## 概述

这是一个基于OpenCV模板匹配的图标检测工具，用于在手机应用截图中检测UI图标的存在。该工具已完全集成到验证框架的条件检查系统中。

## 功能特性

- ✅ **多尺度模板匹配**：支持不同尺寸的图标检测
- ✅ **相似度阈值控制**：可调节检测精度
- ✅ **批量检测**：支持同时检测多个图标
- ✅ **路径智能解析**：根据应用ID自动查找图标文件
- ✅ **非极大值抑制**：去除重复检测结果
- ✅ **条件系统集成**：与escalate和juxtaposition检查器无缝集成
- ✅ **灵活配置**：支持any/all匹配模式

## 安装和配置

### 依赖项

```bash
pip install opencv-python numpy
```

### 图标资源

图标文件应放置在以下目录结构中：

```
task_configs/icons/
├── weixin/
│   ├── icon_001_通讯录.jpg
│   ├── icon_002_微信.jpg
│   └── icon_000_我.jpg
├── bilibili/
│   └── ...
└── xiecheng/
    └── ...
```

## 在YAML配置中使用

### 基本配置

```yaml
# escalate模式：按优先级依次尝试，任意一个成功即返回True
condition:
  type: escalate
  params:
    icons:
      any: ["icon_001_通讯录", "icon_002_微信"]  # 匹配任意一个图标
    ocr:
      all: ["微信", "通讯录"]
    llm:
      prompt: "当前页面是否为微信主界面？"
      expected_true: true

# juxtaposition模式：要求所有检查器都成功
condition:
  type: juxtaposition 
  params:
    icons:
      all: ["icon_001_回车", "icon_002_发送"]  # 必须匹配所有图标
    ocr:
      any: ["发送"]
```

### 高级配置

```yaml
condition:
  type: escalate
  params:
    icons:
      any: ["icon_001_通讯录", "icon_002_微信"]
      threshold: 0.85  # 自定义相似度阈值（可选）
```

## 匹配模式

### any模式
- 列表中任意一个图标匹配成功即认为条件满足
- 适用于多个可能的界面状态

### all模式  
- 要求列表中所有图标都必须匹配成功
- 适用于确认特定界面元素的完整性

## API使用

### 直接使用图标检测服务

```python
from tools.Icon_detection import get_icon_detection_service, detect_icons_simple

# 获取服务实例
service = get_icon_detection_service()

# 检测单个图标
result = detect_icons_simple(
    image_array,  # numpy数组或文件路径
    ["icon_001_通讯录"], 
    app_id="com.tencent.mm"
)

# 获取详细结果
detailed_result = service.detect_icons(
    image_array,
    ["icon_001_通讯录", "icon_002_微信"],
    app_id="com.tencent.mm",
    match_mode='any'
)
```

### 在条件检查器中使用

```python
from avdag.conditions import get_checker

icons_checker = get_checker("icons_match")

frame = {
    'screenshot': image_array,  # 或文件路径
    'app_id': 'com.tencent.mm'
}

params = {
    "any": ["icon_001_通讯录", "icon_002_微信"]
}

result = icons_checker.check(frame, params, options)
```

## 配置参数

### 全局配置

```python
from tools.Icon_detection import IconDetectionConfig, set_default_config

config = IconDetectionConfig(
    default_threshold=0.8,      # 默认相似度阈值
    scale_range=(0.5, 2.0),     # 缩放范围
    scale_step=0.1,             # 缩放步长
    nms_threshold=0.3           # 非极大值抑制阈值
)

set_default_config(config)
```

### 条件参数

- `any`: 图标名称列表，匹配任意一个
- `all`: 图标名称列表，必须匹配所有
- `threshold`: 相似度阈值（可选），覆盖默认值

## 测试

运行测试脚本验证功能：

```bash
# 基础功能测试
python tools/Icon_detection/test_icon_detection.py

# 集成测试
python tools/Icon_detection/test_integration.py
```

## 工作原理

1. **模板加载**：从配置路径加载图标模板文件
2. **多尺度匹配**：对模板进行多种尺寸缩放，与目标图像匹配
3. **相似度计算**：使用OpenCV的TM_CCOEFF_NORMED方法计算相似度
4. **结果筛选**：根据阈值过滤结果，应用非极大值抑制去重
5. **模式判断**：根据any/all模式决定最终结果

## 扩展性

该工具设计为模块化架构，可以轻松扩展：

- **替换检测算法**：可以替换为SIFT、ORB或深度学习检测器
- **增加图标类型**：支持添加新的应用图标资源
- **自定义路径解析**：可以自定义图标文件查找规则
- **结果后处理**：可以添加自定义的结果过滤和排序逻辑

## 性能优化

- **图标缓存**：已加载的图标模板会被缓存，避免重复读取
- **早期终止**：escalate模式下，一旦匹配成功立即返回
- **尺寸预检查**：避免处理过大的缩放模板
- **并行处理**：支持批量检测多个图标

## 故障排除

### 常见问题

1. **图标检测失败**
   - 检查图标文件是否存在于正确路径
   - 调整相似度阈值（降低threshold值）
   - 确认图像质量和分辨率

2. **路径解析错误**
   - 验证app_id与目录名称的映射
   - 检查图标文件扩展名是否支持（png、jpg、jpeg、bmp）

3. **性能问题**
   - 减少缩放范围或增大缩放步长
   - 使用更高的相似度阈值
   - 清空图标模板缓存

### 调试日志

启用DEBUG级别日志查看详细执行信息：

```python
import logging
logging.getLogger('avdag.condition').setLevel(logging.DEBUG)
```

## 示例应用

参考 `task_rules/weixin/weixin-type1.yaml` 查看完整的配置示例。

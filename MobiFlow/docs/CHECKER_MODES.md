# 检查器模式详细说明

## 概述

验证框架现在支持两种主要的检查器组合模式：

1. **escalate**: 升级模式 - 按顺序尝试，任意一个成功即返回成功
2. **juxtaposition**: 并列模式 - 所有配置的检查器都必须成功

## 1. Escalate 模式（升级模式）

### 工作原理
- 按照 `escalation_order` 的顺序依次尝试配置的检查器
- **任意一个检查器返回 True，立即返回 True**（短路求值）
- 适用于"宽松验证"场景，有多种验证方式但只需要其中一种成功

### 默认升级顺序
```python
escalation_order: ["text", "regex", "action", "icons", "ocr", "llm"]
```

### 配置示例
```yaml
condition:
  type: escalate
  params:
    # 第1优先级：简单文本匹配
    text:
      any: ["搜索", "查找"]
    
    # 第2优先级：正则表达式
    regex:
      pattern: ".*(搜索|查询).*"
      ignore_case: true
    
    # 第2优先级：动作验证
    action:
      type: click
    
    # 第4优先级: icons图标识别
    icons:
      any: ["购物车"]
    # 第5优先级：OCR识别
    ocr:
      any: ["搜索"]
    
    # 第6优先级：LLM验证
    llm:
      prompt: "该步是否执行了搜索？"
```

### 执行流程
1. 检查 `text` 配置，如果匹配成功 → 立即返回 True
2. 如果 `text` 失败，检查 `regex`，如果成功 → 立即返回 True
3. 如果 `regex` 失败，检查 `ui`，如果成功 → 立即返回 True
4. 依此类推...
5. 当只配置某种策略如ocr或llm时，则只使用该方式
6. 如果所有配置的检查器都失败 → 返回 False

### 适用场景
- 搜索操作验证（可能通过文本、UI状态或LLM任一方式确认）
- 页面跳转验证（可能通过多种方式检测）
- 灵活的操作确认

---

## 2. Juxtaposition 模式（并列模式）

### 工作原理
- **所有配置的检查器都必须返回 True**，才认为验证成功
- 任意一个检查器失败，整个验证失败
- 适用于"严格验证"场景，需要多重确认

### 配置示例
```yaml
condition:
  type: juxtaposition
  params:
    # 必须满足：文本包含关键词
    text:
      any: ["确认", "提交"]
    
    # 必须满足：是点击动作
    action:
      type: click
    
    # 必须满足：UI状态正确
    ui:
      key: screen
      equals: confirm_page
    
    # 必须满足：OCR识别成功
    ocr:
      all: ["确认", "提交"]
    
    # 必须满足：LLM确认
    llm:
      prompt: "该步是否点击了确认按钮？"
```

### 执行流程
1. 执行所有配置的检查器
2. 收集所有检查器的结果
3. 只有当**所有结果都是 True** 时，才返回 True
4. 任意一个结果为 False，返回 False

### 适用场景
- 关键操作的严格验证（如支付、确认订单）
- 需要多重确认的敏感操作
- 复合条件验证（必须同时满足多个条件）

---

## 3. 支持的检查器类型

两种模式都支持以下检查器：
优先推荐使用icons、ocr和llm检测，仅基于trace的完整截图就能够判断。

| 检查器 | 参数格式 | 说明 |
|--------|----------|------|
| `text` | `{any: [...], all: [...]}` | 文本包含匹配 |
| `regex` | `{pattern: "...", ignore_case: bool}` | 正则表达式匹配 |
| `ui` | `{key: "...", equals: "...", in: [...]}` | UI状态检查 |
| `action` | `{type: "...", contains: {...}}` | 动作类型验证 |
| `xml` | `{any: [...], all: [...]}` | XML文本匹配 |
| `ocr` | `{any: [...], all: [...], pattern: "..."}` | OCR图像识别 |
| `llm` | `{prompt: "...", expected_true: bool}` | LLM智能验证 |

---


## 4. 自定义升级顺序

可以通过 `VerifierOptions` 自定义 escalate 的执行顺序：

```python
from avdag.types import VerifierOptions

# 自定义顺序：优先UI检查，然后文本，最后LLM
custom_options = VerifierOptions(
    escalation_order=["ui", "ocr", "llm"]
)
```

---

## 6. 实际使用建议

### 使用 Escalate 的场景
- 操作验证有多种可能的确认方式
- 需要从简单到复杂的渐进式验证
- 性能要求较高，希望早期匹配成功

### 使用 Juxtaposition 的场景
- 关键操作需要多重确认
- 必须同时满足多个严格条件
- 需要确保验证的可靠性和准确性

### 组合使用
在复杂任务中，可以在不同节点使用不同模式：

```yaml
nodes:
  - id: search_input
    condition:
      type: escalate  # 搜索输入验证相对宽松
      params:
        text: {any: ["搜索"]}
        action: {type: input}
        
  - id: payment_confirm
    condition:
      type: juxtaposition  # 支付确认必须严格验证
      params:
        text: {all: ["支付", "确认"]}
        action: {type: click}
        ocr: {all: ["支付", "确认"]}
        llm: {prompt: "是否点击了支付确认？"}
```

---

## 7. 调试输出

两种模式都提供详细的调试输出：

### Escalate 模式输出
```
[Escalate] 升级顺序: ['text', 'regex', 'ui', 'action', 'dynamic_match', 'ocr', 'llm']
[Escalate] 配置的检查器: ['text', 'ui', 'llm']
[Escalate] 尝试检查器: text
[Escalate] text 检查结果: False
[Escalate] 尝试检查器: ui
[Escalate] ui 检查结果: True
[Escalate] ui 检查成功，立即返回True
```

### Juxtaposition 模式输出
```
[Juxtaposition] text_match 结果: True
[Juxtaposition] ui_flag 结果: True
[Juxtaposition] action_match 结果: False
[Juxtaposition] 配置的检查器: ['text_match', 'ui_flag', 'action_match']
[Juxtaposition] 各检查器结果: [True, True, False]
[Juxtaposition] 最终结果: False
```

这种详细的输出有助于理解验证过程和调试配置问题。

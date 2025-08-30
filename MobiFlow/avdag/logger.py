"""
统一的日志系统 - 为 avdag 框架和相关工具提供灵活的日志输出配置

支持多种日志级别和输出方式：
- CRITICAL: 关键错误，会导致程序无法继续执行
- ERROR: 普通错误，不影响程序继续运行
- WARNING: 警告信息
- INFO: 一般信息，默认显示
- DEBUG: 调试信息，详细的执行过程
- TRACE: 最详细的跟踪信息

配置方式：
1. 环境变量: AVDAG_LOG_LEVEL=DEBUG
2. 代码配置: set_log_level('DEBUG')
3. 配置文件: 通过 configure_logging() 函数加载

使用方式：
```python
from avdag.logger import get_logger

logger = get_logger(__name__)
logger.info("这是一般信息")
logger.debug("这是调试信息")
logger.error("这是错误信息")
```
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any, Union, List
from enum import Enum
from pathlib import Path

class LogLevel(Enum):
    """日志级别枚举"""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    TRACE = 5
    
    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """从字符串获取日志级别"""
        level_map = {
            'CRITICAL': cls.CRITICAL,
            'FATAL': cls.CRITICAL,
            'ERROR': cls.ERROR,
            'WARNING': cls.WARNING,
            'WARN': cls.WARNING,
            'INFO': cls.INFO,
            'DEBUG': cls.DEBUG,
            'TRACE': cls.TRACE,
        }
        return level_map.get(level_str.upper(), cls.INFO)


class AVDAGLogger:
    """AVDAG 专用日志器"""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(f"avdag.{name}")
        
        # 添加 TRACE 级别支持
        if not hasattr(logging, 'TRACE'):
            logging.addLevelName(LogLevel.TRACE.value, 'TRACE')
            def trace(self, msg, *args, **kwargs):
                if self.isEnabledFor(LogLevel.TRACE.value):
                    self._log(LogLevel.TRACE.value, msg, args, **kwargs)
            logging.Logger.trace = trace
    
    def critical(self, msg: str, *args, **kwargs):
        """记录关键错误信息"""
        self._logger.critical(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """记录错误信息"""
        self._logger.error(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """记录警告信息"""
        self._logger.warning(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """记录一般信息"""
        self._logger.info(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """记录调试信息"""
        self._logger.debug(msg, *args, **kwargs)
    
    def trace(self, msg: str, *args, **kwargs):
        """记录最详细的跟踪信息"""
        if hasattr(self._logger, 'trace'):
            self._logger.trace(msg, *args, **kwargs)
        else:
            self._logger.log(LogLevel.TRACE.value, msg, *args, **kwargs)
    
    def is_enabled_for(self, level: Union[str, LogLevel]) -> bool:
        """检查是否启用了指定级别的日志"""
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        return self._logger.isEnabledFor(level.value)


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI 颜色代码
    COLORS = {
        'CRITICAL': '\033[95m',  # 紫色
        'ERROR': '\033[91m',     # 红色
        'WARNING': '\033[93m',   # 黄色
        'INFO': '\033[92m',      # 绿色
        'DEBUG': '\033[96m',     # 青色
        'TRACE': '\033[90m',     # 灰色
    }
    RESET = '\033[0m'
    
    def _supports_color(self) -> bool:
        """检查是否支持颜色输出（跨平台）"""
        # 检查是否为TTY
        if not sys.stderr.isatty():
            return False
        
        # 检查环境变量
        if os.getenv('NO_COLOR'):
            return False
        
        if os.getenv('FORCE_COLOR'):
            return True
        
        # 平台特定检查
        if sys.platform == 'win32':
            # Windows: 检查是否支持ANSI转义序列
            try:
                # Windows 10及以上版本通常支持ANSI颜色
                import platform
                version = platform.version()
                # Windows 10的版本号通常是10.0.x
                if version.startswith('10.0.') or version.startswith('11.'):
                    return True
            except:
                pass
            
            # 检查TERM环境变量
            term = os.getenv('TERM', '').lower()
            if 'color' in term or 'ansi' in term:
                return True
            
            # 默认Windows支持（现代终端如Windows Terminal、VS Code等）
            return True
        
        else:
            # Unix/Linux: 检查TERM环境变量
            term = os.getenv('TERM', '').lower()
            if term in ['dumb', 'unknown']:
                return False
            
            # 大多数Unix终端支持颜色
            return 'color' in term or 'ansi' in term or 'xterm' in term or term in [
                'screen', 'tmux', 'rxvt', 'konsole', 'gnome-terminal'
            ]
    
    def __init__(self, use_colors: bool = True, show_time: bool = True, show_module: bool = True):
        # 增强的颜色支持检测
        self.use_colors = use_colors and self._supports_color()
        self.show_time = show_time
        self.show_module = show_module
        
        # 构建格式字符串
        fmt_parts = []
        if show_time:
            fmt_parts.append('%(asctime)s')
        fmt_parts.append('[%(levelname)s]')
        if show_module:
            fmt_parts.append('%(name)s')
        fmt_parts.append('%(message)s')
        
        fmt = ' '.join(fmt_parts)
        super().__init__(fmt, datefmt='%H:%M:%S')
    
    def format(self, record):
        # 创建record的副本，避免修改原始record影响其他处理器
        if self.use_colors:
            # 复制record以避免修改原始对象
            import copy
            record_copy = copy.copy(record)
            levelname = record_copy.levelname
            if levelname in self.COLORS:
                record_copy.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            return super().format(record_copy)
        else:
            return super().format(record)


class LoggingConfig:
    """日志配置管理"""
    
    def __init__(self):
        self._configured = False
        self._loggers: Dict[str, AVDAGLogger] = {}
        self._default_level = LogLevel.INFO
        self._handlers: List[logging.Handler] = []
    
    def configure(self, 
                  level: Union[str, LogLevel] = LogLevel.DEBUG,
                  use_colors: bool = True,
                  show_time: bool = True,
                  show_module: bool = True,
                  output_file: Optional[str] = None,
                  config_file: Optional[str] = None) -> None:
        """配置日志系统
        
        Args:
            level: 日志级别
            use_colors: 是否使用颜色输出
            show_time: 是否显示时间
            show_module: 是否显示模块名
            output_file: 输出到文件（可选）
            config_file: 从配置文件加载（可选）
        """
        
        # 从配置文件加载设置
        if config_file and Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            level = level or config.get('level', 'INFO')
            use_colors = config.get('use_colors', use_colors)
            show_time = config.get('show_time', show_time)
            show_module = config.get('show_module', show_module)
            output_file = output_file or config.get('output_file')
        
        # 从环境变量获取级别
        if level is None:
            level = os.getenv('AVDAG_LOG_LEVEL', 'INFO')
        
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        
        self._default_level = level
        
        # 配置根日志器
        root_logger = logging.getLogger('avdag')
        root_logger.setLevel(level.value)
        
        # 移除现有处理器（更彻底的清理）
        for handler in self._handlers:
            root_logger.removeHandler(handler)
        self._handlers.clear()
        
        # 清除根日志器上的所有处理器（防止重复配置导致的问题）
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 总是创建控制台处理器
        console_handler = logging.StreamHandler(sys.stderr)
        console_formatter = ColoredFormatter(use_colors, show_time, show_module)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        self._handlers.append(console_handler)
        
        # 如果指定了输出文件，同时创建文件处理器（不使用颜色）
        if output_file:
            try:
                # 规范化文件路径（跨平台兼容）
                output_path = Path(output_file).resolve()
                # 确保目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 创建文件处理器，强制使用UTF-8编码
                file_handler = logging.FileHandler(
                    str(output_path), 
                    encoding='utf-8',
                    mode='a'  # 追加模式，避免覆盖现有日志
                )
                
                # 构建文件格式字符串（与控制台格式一致，但不包含颜色）
                fmt_parts = []
                if show_time:
                    fmt_parts.append('%(asctime)s')
                fmt_parts.append('[%(levelname)s]')
                if show_module:
                    fmt_parts.append('%(name)s')
                fmt_parts.append('%(message)s')
                fmt = ' '.join(fmt_parts)
                
                file_formatter = logging.Formatter(fmt, datefmt='%H:%M:%S')
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                self._handlers.append(file_handler)
                
            except Exception as e:
                # 如果文件创建失败，打印警告但继续执行（只使用控制台输出）
                print(f"警告: 无法创建日志文件 {output_file}: {e}", file=sys.stderr)
        
        # 防止传播到根日志器（避免重复输出）
        root_logger.propagate = False
        
        self._configured = True
    
    def get_logger(self, name: str) -> AVDAGLogger:
        """获取或创建日志器"""
        if name not in self._loggers:
            self._loggers[name] = AVDAGLogger(name)
        return self._loggers[name]
    
    def set_level(self, level: Union[str, LogLevel]) -> None:
        """设置全局日志级别"""
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        
        self._default_level = level
        root_logger = logging.getLogger('avdag')
        root_logger.setLevel(level.value)
    
    def get_level(self) -> LogLevel:
        """获取当前日志级别"""
        return self._default_level
    
    def is_configured(self) -> bool:
        """检查日志系统是否已配置"""
        return self._configured


# 全局配置实例
_config = LoggingConfig()

def configure_logging(**kwargs) -> None:
    """配置日志系统的便捷函数"""
    _config.configure(**kwargs)

def get_logger(name: str) -> AVDAGLogger:
    """获取日志器的便捷函数"""
    if not _config.is_configured():
        # 自动配置：使用默认设置
        configure_logging()
    return _config.get_logger(name)

def set_log_level(level: Union[str, LogLevel]) -> None:
    """设置日志级别的便捷函数"""
    _config.set_level(level)

def get_log_level() -> LogLevel:
    """获取当前日志级别"""
    return _config.get_level()

def is_debug_enabled() -> bool:
    """检查是否启用调试模式"""
    return _config.get_level().value <= LogLevel.DEBUG.value

def is_trace_enabled() -> bool:
    """检查是否启用跟踪模式"""
    return _config.get_level().value <= LogLevel.TRACE.value


# 兼容性函数：保持与现有代码的兼容性
def debug_print(msg: str, category: str = "DEBUG") -> None:
    """兼容性函数：替代原有的 print 调用"""
    logger = get_logger(category.lower())
    logger.debug(msg)

def info_print(msg: str, category: str = "INFO") -> None:
    """兼容性函数：替代原有的 print 调用"""
    logger = get_logger(category.lower())
    logger.info(msg)

def error_print(msg: str, category: str = "ERROR") -> None:
    """兼容性函数：替代原有的 print 调用"""
    logger = get_logger(category.lower())
    logger.error(msg)

def warning_print(msg: str, category: str = "WARNING") -> None:
    """兼容性函数：替代原有的 print 调用"""
    logger = get_logger(category.lower())
    logger.warning(msg)


# 预定义的常用日志器
def get_verifier_logger() -> AVDAGLogger:
    """获取验证器日志器"""
    return get_logger("verifier")

def get_ocr_logger() -> AVDAGLogger:
    """获取OCR处理日志器"""
    return get_logger("ocr")

def get_llm_logger() -> AVDAGLogger:
    """获取LLM调用日志器"""
    return get_logger("llm")

def get_frame_logger() -> AVDAGLogger:
    """获取帧处理日志器"""
    return get_logger("frame")

def get_condition_logger() -> AVDAGLogger:
    """获取条件检查日志器"""
    return get_logger("condition")


# 模块级别的便捷日志器
logger = get_logger(__name__)

def test_logging_compatibility():
    """测试日志系统的跨平台兼容性"""
    import tempfile
    
    print("=== 日志系统兼容性测试 ===")
    
    # 测试平台信息
    print(f"平台: {sys.platform}")
    print(f"TTY支持: {sys.stderr.isatty()}")
    print(f"编码: {sys.stderr.encoding}")
    
    # 测试颜色支持
    formatter = ColoredFormatter()
    print(f"颜色支持: {formatter.use_colors}")
    
    # 测试环境变量
    print(f"TERM: {os.getenv('TERM', 'N/A')}")
    print(f"NO_COLOR: {os.getenv('NO_COLOR', 'N/A')}")
    print(f"FORCE_COLOR: {os.getenv('FORCE_COLOR', 'N/A')}")
    
    # 测试文件输出
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp:
        temp_log_file = tmp.name
    
    try:
        # 配置日志系统
        configure_logging(
            level='DEBUG',
            use_colors=True,
            show_time=True,
            show_module=True,
            output_file=temp_log_file
        )
        
        # 测试日志输出
        test_logger = get_logger('test')
        test_logger.debug('测试DEBUG')
        test_logger.info('测试INFO')
        test_logger.warning('测试WARNING')
        test_logger.error('测试ERROR')
        
        # 读取文件内容
        with open(temp_log_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        print(f"\n文件日志内容预览:")
        for i, line in enumerate(file_content.splitlines()[:2], 1):
            print(f"  {i}: {line}")
        
        # 检查是否包含颜色代码
        has_color_codes = any(code in file_content for code in ['\033[', '[96m', '[92m'])
        print(f"文件包含颜色代码: {'是' if has_color_codes else '否'}")
        
        print("✅ 兼容性测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_log_file)
        except:
            pass

# 模块级别的便捷日志器
logger = get_logger(__name__)

# 自动配置检查
if not _config.is_configured():
    # 检查是否有环境变量或配置文件
    config_file = os.getenv('AVDAG_LOG_CONFIG')
    if config_file and Path(config_file).exists():
        configure_logging(config_file=config_file)
    else:
        # 使用默认配置
        configure_logging()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标处理和可视化模块
参考UI-TARS/README_coordinates.md的实现
"""

import math
import logging
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# 常量定义（参考README_coordinates.md）
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, 
    width: int, 
    factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, 
    max_pixels: int = MAX_PIXELS
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

class CoordinateProcessor:
    """坐标处理器"""
    
    @staticmethod
    def convert_model_coords_to_actual(
        model_x: int, 
        model_y: int, 
        original_width: int, 
        original_height: int
    ) -> Tuple[int, int]:
        """
        将模型输出的坐标转换为实际设备坐标
        参考README_coordinates.md的实现
        """
        try:
            # 计算调整后的尺寸
            new_height, new_width = smart_resize(original_height, original_width)
            
            # 转换坐标
            actual_x = int(model_x / new_width * original_width)
            actual_y = int(model_y / new_height * original_height)
            
            logger.info(f"坐标转换: 模型({model_x}, {model_y}) -> 实际({actual_x}, {actual_y})")
            logger.info(f"原始尺寸: {original_width}x{original_height}, 调整尺寸: {new_width}x{new_height}")
            
            return actual_x, actual_y
            
        except Exception as e:
            logger.error(f"坐标转换失败: {e}")
            # 如果转换失败，返回原始坐标
            return model_x, model_y
    
    @staticmethod
    def create_visualization_image(
        screenshot_path: str, 
        click_x: int, 
        click_y: int, 
        output_path: str,
        marker_size: int = 20,
        marker_color: str = 'red'
    ) -> bool:
        """
        创建带有点击位置标记的可视化图像
        """
        try:
            # 打开原始截图
            img = Image.open(screenshot_path)
            img_copy = img.copy()
            
            # 创建绘图对象
            draw = ImageDraw.Draw(img_copy)
            
            # 绘制点击位置标记（十字架）
            marker_half = marker_size // 2
            
            # 绘制红色十字架
            draw.line([
                (click_x - marker_half, click_y),
                (click_x + marker_half, click_y)
            ], fill=marker_color, width=3)
            
            draw.line([
                (click_x, click_y - marker_half),
                (click_x, click_y + marker_half)
            ], fill=marker_color, width=3)
            
            # 绘制圆形标记
            draw.ellipse([
                (click_x - marker_half//2, click_y - marker_half//2),
                (click_x + marker_half//2, click_y + marker_half//2)
            ], outline=marker_color, width=2)
            
            # 保存可视化图像
            img_copy.save(output_path)
            logger.info(f"可视化图像已保存: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"创建可视化图像失败: {e}")
            return False
    
    @staticmethod
    def create_matplotlib_visualization(
        screenshot_path: str,
        click_x: int,
        click_y: int,
        output_path: str
    ) -> bool:
        """
        使用matplotlib创建可视化图像
        """
        try:
            # 打开图像
            img = Image.open(screenshot_path)
            
            # 创建matplotlib图像
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.scatter([click_x], [click_y], c='red', s=100, marker='x')  # 用红色X标记点击位置
            plt.title(f'Click Visualization at ({click_x}, {click_y})')
            plt.axis('off')  # 隐藏坐标轴
            
            # 保存图像
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # 关闭图像以释放内存
            
            logger.info(f"matplotlib可视化图像已保存: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建matplotlib可视化图像失败: {e}")
            return False

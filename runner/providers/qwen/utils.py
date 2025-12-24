import os
import sys
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# 导入utils.parse_omni中的extract_all_bounds函数
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from utils.parse_omni import extract_all_bounds
except ImportError:
    print("警告: 无法导入utils.parse_omni，extract_all_bounds将不可用")
    extract_all_bounds = None

font_path = project_root / "msyh.ttf"

def check_text_overlap(text_rect1, text_rect2):
    """检查两个文本矩形是否重叠"""
    x1, y1, x2, y2 = text_rect1
    x3, y3, x4, y4 = text_rect2
    
    if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
        return False
    return True

def draw_bounds_on_screenshot(image_or_path, layer, output_path=None):
    """
    在截图上绘制边界框
    Args:
        image_or_path: PIL Image或图像路径
        layer: (index, bounds, text_rect)的列表
        output_path: 如果提供，将图像保存到此处
    Returns:
        绘制了边界框的PIL Image
    """
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert('RGB')
    else:
        image = image_or_path.copy().convert('RGB')
        
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(str(font_path), 40)
    except Exception:
        font = ImageFont.load_default()
    
    for index, bounds, text_rect in layer:
        left, top, right, bottom = bounds
        draw.rectangle([left, top, right, bottom], outline='red', width=5)

        text = str(index)
        text_x, text_y, _, _ = text_rect

        draw.rectangle(text_rect, fill='red', outline='red', width=1)
        draw.text((text_x, text_y), text, fill='white', font=font)
    
    if output_path:
        image.save(output_path)
        
    return image

def assign_bounds_to_layers(image_path, bounds_list):
    """
    将边界框分配到不同的层以避免文本重叠
    Returns:
        层的列表，每个层是(index, bounds, text_rect)的列表
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype(str(font_path), 40)
    except Exception:
        font = ImageFont.load_default()
    
    layers = []
    
    for index, bounds in enumerate(bounds_list):
        left, top, right, bottom = bounds
        
        text = str(index)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = left
        text_y = top
        text_rect = (text_x, text_y, text_x + text_width + 5, text_y + text_height + 15)

        placed = False
        for layer in layers:
            can_place = True
            for _, existing_bounds, existing_text_rect in layer:
                if (check_text_overlap(bounds, existing_bounds) or 
                    check_text_overlap(text_rect, existing_bounds) or 
                    check_text_overlap(bounds, existing_text_rect) or 
                    check_text_overlap(text_rect, existing_text_rect)):
                    can_place = False
                    break
            
            if can_place:
                layer.append((index, bounds, text_rect))
                placed = True
                break
        
        if not placed:
            layers.append([(index, bounds, text_rect)])
            
    return layers

def process_screenshot(screenshot_path):
    """
    处理截图：提取边界框并生成图层
    Returns:
        (bounds_list, layer_images)
        layer_images是PIL Image的列表
    """
    if extract_all_bounds is None:
        return [], []
        
    bounds_list = extract_all_bounds(screenshot_path)
    layers_data = assign_bounds_to_layers(screenshot_path, bounds_list)
    
    layer_images = []
    for layer in layers_data:
        img = draw_bounds_on_screenshot(screenshot_path, layer)
        layer_images.append(img)
        
    return bounds_list, layer_images

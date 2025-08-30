from PIL import Image, ImageDraw, ImageFont
import os

# from parse_xml import extract_all_bounds
from utils.parse_omni import extract_all_bounds

def check_text_overlap(text_rect1, text_rect2):
    """检查两个文本矩形是否重叠"""
    x1, y1, x2, y2 = text_rect1
    x3, y3, x4, y4 = text_rect2
    
    # 如果一个矩形在另一个的左侧、右侧、上方或下方，则不重叠
    if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
        return False
    return True

def assign_bounds_to_layers(folder_path, screenshot_path, bounds_list):
    """使用贪心算法将bounds分配到不同的图层，避免文本重叠"""
    image = Image.open(screenshot_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 40)
    
    layers = []  # 每个元素是一个包含(index, bounds, text_rect)的列表
    
    for index, bounds in enumerate(bounds_list):
        left, top, right, bottom = bounds
        
        text = str(index)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # text_x = right - text_width
        text_x = left
        text_y = top

        text_rect = (text_x, text_y, text_x + text_width + 5, text_y + text_height + 15)

        # 寻找可以容纳当前bounds的图层
        placed = False
        for layer in layers:
            can_place = True
            for _, existing_bounds, existing_text_rect in layer:
                # if check_text_overlap(text_rect, existing_text_rect):
                if check_text_overlap(bounds, existing_bounds) or check_text_overlap(text_rect, existing_bounds) or check_text_overlap(bounds, existing_text_rect) or check_text_overlap(text_rect, existing_text_rect):
                    can_place = False
                    break
            
            if can_place:
                layer.append((index, bounds, text_rect))
                placed = True
                break
        
        # 如果没有找到合适的图层，创建新图层
        if not placed:
            layers.append([(index, bounds, text_rect)])

    for index, layer in enumerate(layers, 1):
        output_path = os.path.join(folder_path, f"layer_{index}.jpg")
        draw_bounds_on_screenshot(screenshot_path, layer, output_path)
    
    return len(layers)

def draw_bounds_on_screenshot(screenshot_path, layer, output_path):
    """在截图上绘制所有bounds并保存"""
    try:
        image = Image.open(screenshot_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 40)
        
        # 用红色绘制所有bounds并标记索引
        for index, bounds, text_rect in layer:
            left, top, right, bottom = bounds
            draw.rectangle([left, top, right, bottom], outline='red', width=5)

            text = str(index)
            text_x, text_y, _, _ = text_rect

            draw.rectangle(text_rect, fill='red', outline='red', width=1)
            draw.text((text_x, text_y), text, fill='white', font=font)
        
        image.save(output_path)
        # print(f"已保存标注结果到: {output_path}")
        return True
        
    except Exception as e:
        print(f"绘制bounds时出错: {str(e)}")
        return False
 
def process_folder(folder_path, need_clickable=False):
    """处理单个文件夹"""
    hierarchy_path = os.path.join(folder_path, 'hierarchy.xml')
    screenshot_path = os.path.join(folder_path, 'screenshot.jpg')
    
    try:
        # 读取hierarchy.xml
        # with open(hierarchy_path, 'r', encoding='utf-8') as f:
        #     hierarchy_xml = f.read()
        
        # 提取所有bounds
        # bounds_list = extract_all_bounds(hierarchy_xml, need_clickable)
        
        bounds_list = extract_all_bounds(screenshot_path)
        # print(f"在 {folder_path} 中找到 {len(bounds_list)} 个bounds")
        
        return assign_bounds_to_layers(folder_path, screenshot_path, bounds_list), bounds_list
        
    except Exception as e:
        print(f"处理文件夹 {folder_path} 时出错: {str(e)}")
        return 0

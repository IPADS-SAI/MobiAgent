from utils.omni_utils import get_som_labeled_img, check_ocr_box, get_yolo_model
from PIL import Image
import torch
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
current_file = os.path.dirname(os.path.abspath(__file__))
detect_model_path=os.path.join(current_file,"..","weights/icon_detect/model.pt")
caption_model_path=os.path.join(current_file,'..','weights/icon_caption_florence')

som_model = get_yolo_model(detect_model_path)
som_model.to(device)

def extract_all_bounds(screenshot_path):
    """提取截图中的所有边界框信息，优化参数以识别更多可点击元素"""
    image = Image.open(screenshot_path).convert('RGB')
    image_width, image_height = image.size
    
    # OCR检测文本框 - 降低阈值以识别更多文本
    (text, ocr_bbox), _ = check_ocr_box(
        image,
        display_img=False, 
        output_bb_format='xyxy', 
        easyocr_args={
            'text_threshold': 0.7,  # 降低阈值从0.9到0.7
            'low_text': 0.4,        # 添加低置信度文本检测
            'link_threshold': 0.4,  # 降低连接阈值
            'canvas_size': 2560,    # 增加画布大小提高检测精度
            'mag_ratio': 1.5        # 增加放大比例
        }, 
        use_paddleocr=True,
    )

    # YOLO检测UI元素 - 降低阈值以检测更多UI组件
    _, _, parsed_content_list = get_som_labeled_img(
        image, 
        som_model, 
        BOX_TRESHOLD=0.05,  # 大幅降低阈值从0.1到0.05
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        ocr_text=text,
        use_local_semantics=False,
        iou_threshold=0.5,  # 降低IoU阈值以减少过度合并
        scale_img=False
    )

    # 提取边界框并转换为绝对坐标
    bounds_list = []
    
    # 处理YOLO检测到的UI元素
    for item in parsed_content_list:
        bbox = item.get('bbox')
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            # 转换为绝对坐标
            left = int(x1 * image_width)
            top = int(y1 * image_height)
            right = int(x2 * image_width)
            bottom = int(y2 * image_height)
            
            # 验证边界框的有效性
            if right > left and bottom > top and left >= 0 and top >= 0:
                bounds_list.append([left, top, right, bottom])
    
    # 额外添加OCR检测到的文本框（如果还没包含在YOLO结果中）
    if ocr_bbox:
        for bbox in ocr_bbox:
            if len(bbox) >= 4:
                left, top, right, bottom = bbox[:4]
                # 确保坐标为整数
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                
                # 验证边界框的有效性
                if right > left and bottom > top and left >= 0 and top >= 0:
                    # 检查是否已经存在类似的边界框
                    is_duplicate = False
                    for existing_bounds in bounds_list:
                        ex_left, ex_top, ex_right, ex_bottom = existing_bounds
                        # 计算重叠面积
                        overlap_left = max(left, ex_left)
                        overlap_top = max(top, ex_top)
                        overlap_right = min(right, ex_right)
                        overlap_bottom = min(bottom, ex_bottom)
                        
                        if overlap_right > overlap_left and overlap_bottom > overlap_top:
                            overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
                            current_area = (right - left) * (bottom - top)
                            existing_area = (ex_right - ex_left) * (ex_bottom - ex_top)
                            
                            # 如果重叠面积占当前框面积的80%以上，认为是重复的
                            if overlap_area > 0.8 * min(current_area, existing_area):
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        bounds_list.append([left, top, right, bottom])

    print(f"[Extract Bounds] 检测到 {len(bounds_list)} 个边界框")
    return bounds_list

def find_clicked_element(bounds_list, click_x, click_y, expand_ratio=0.1, nearby_threshold=20):
    """
    找到包含点击位置的最合适边界框，增加容错和扩展机制
    
    Args:
        bounds_list: 边界框列表
        click_x, click_y: 点击坐标
        expand_ratio: 边界框扩展比例（默认5%）
        nearby_threshold: 附近点击的距离阈值（像素）
    """
    if not bounds_list:
        print(f"[Find Element] 警告：没有可用的边界框")
        return None
    
    smallest_bounds = None
    smallest_area = float('inf')
    exact_matches = []  # 精确匹配的边界框
    nearby_matches = []  # 附近的边界框
    
    print(f"[Find Element] 查找点击位置 ({click_x}, {click_y}) 对应的元素，共有 {len(bounds_list)} 个候选边界框")
    
    # 第一轮：查找精确包含点击位置的边界框
    for i, bounds in enumerate(bounds_list):
        left, top, right, bottom = bounds
        
        # 扩展边界框以提高命中率
        width = right - left
        height = bottom - top
        expand_w = int(width * expand_ratio)
        expand_h = int(height * expand_ratio)
        
        expanded_left = max(0, left - expand_w)
        expanded_top = max(0, top - expand_h)
        expanded_right = right + expand_w
        expanded_bottom = bottom + expand_h
        
        # 检查点击位置是否在扩展后的边界框内
        if expanded_left <= click_x <= expanded_right and expanded_top <= click_y <= expanded_bottom:
            area = width * height
            exact_matches.append((bounds, area, i))
            
            if area < smallest_area:
                smallest_area = area
                smallest_bounds = bounds
            
            print(f"[Find Element] 找到匹配边界框 {i}: {bounds} (面积: {area})")
    
    # 如果找到精确匹配，返回面积最小的
    if exact_matches:
        print(f"[Find Element] 找到 {len(exact_matches)} 个精确匹配，选择面积最小的")
        return smallest_bounds
    
    # 第二轮：查找附近的边界框
    print(f"[Find Element] 未找到精确匹配，查找距离 {nearby_threshold} 像素内的边界框")
    
    for i, bounds in enumerate(bounds_list):
        left, top, right, bottom = bounds
        
        # 计算点击位置到边界框的最短距离
        distance = 0
        
        # 点击位置在边界框左侧
        if click_x < left:
            if click_y < top:
                # 左上角
                distance = ((click_x - left) ** 2 + (click_y - top) ** 2) ** 0.5
            elif click_y > bottom:
                # 左下角
                distance = ((click_x - left) ** 2 + (click_y - bottom) ** 2) ** 0.5
            else:
                # 左侧边
                distance = left - click_x
        # 点击位置在边界框右侧
        elif click_x > right:
            if click_y < top:
                # 右上角
                distance = ((click_x - right) ** 2 + (click_y - top) ** 2) ** 0.5
            elif click_y > bottom:
                # 右下角
                distance = ((click_x - right) ** 2 + (click_y - bottom) ** 2) ** 0.5
            else:
                # 右侧边
                distance = click_x - right
        # 点击位置在边界框的水平范围内
        else:
            if click_y < top:
                # 上方
                distance = top - click_y
            elif click_y > bottom:
                # 下方
                distance = click_y - bottom
            else:
                # 内部（不应该到这里，因为第一轮已经处理了）
                distance = 0
        
        if distance <= nearby_threshold:
            area = (right - left) * (bottom - top)
            nearby_matches.append((bounds, area, distance, i))
            print(f"[Find Element] 找到附近边界框 {i}: {bounds} (距离: {distance:.1f}, 面积: {area})")
    
    # 如果找到附近的边界框，选择距离最近的，如果距离相同则选择面积最小的
    if nearby_matches:
        # 按距离排序，距离相同按面积排序
        nearby_matches.sort(key=lambda x: (x[2], x[1]))
        best_match = nearby_matches[0]
        print(f"[Find Element] 选择最近的边界框: {best_match[0]} (距离: {best_match[2]:.1f})")
        return best_match[0]
    
    # 最后的兜底策略：选择面积最小且距离在合理范围内的边界框
    print(f"[Find Element] 使用兜底策略，查找距离 {nearby_threshold * 2} 像素内面积最小的边界框")
    
    fallback_matches = []
    for i, bounds in enumerate(bounds_list):
        left, top, right, bottom = bounds
        
        # 计算边界框中心点到点击位置的距离
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        center_distance = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
        
        if center_distance <= nearby_threshold * 2:
            area = (right - left) * (bottom - top)
            fallback_matches.append((bounds, area, center_distance, i))
    
    if fallback_matches:
        # 按面积排序，选择面积最小的
        fallback_matches.sort(key=lambda x: x[1])
        best_match = fallback_matches[0]
        print(f"[Find Element] 兜底策略选择: {best_match[0]} (中心距离: {best_match[2]:.1f}, 面积: {best_match[1]})")
        return best_match[0]
    
    print(f"[Find Element] 警告：未找到合适的边界框")
    return None
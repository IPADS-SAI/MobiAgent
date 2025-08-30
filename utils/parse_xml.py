import xml.etree.ElementTree as ET
import re

def parse_bounds(bounds_str):
    """解析bounds字符串，返回(left, top, right, bottom)"""
    if not bounds_str:
        return None
    
    # 使用正则表达式提取坐标
    match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
    if match:
        left, top, right, bottom = map(int, match.groups())
        return [left, top, right, bottom]
    return None

def is_point_in_bounds(x, y, bounds):
    """检查点(x,y)是否在bounds范围内"""
    if not bounds:
        return False
    
    [left, top, right, bottom] = bounds
    return left <= x <= right and top <= y <= bottom

def extract_all_bounds(hierarchy_xml, need_clickable=False):
    """从hierarchy.xml中提取所有bounds"""
    try:
        root = ET.fromstring(hierarchy_xml)
        bounds_set = set()
        
        # 递归遍历所有节点
        def traverse_node(node):
            clickable = node.get('clickable', 'false')
            bounds_str = node.get('bounds', '')
            if bounds_str and (need_clickable is False or clickable == 'true'):
                bounds = parse_bounds(bounds_str)
                if bounds:
                    # 将列表转换为元组添加到集合中，避免重复
                    bounds_set.add(tuple(bounds))
            
            # 递归处理子节点
            for child in node:
                traverse_node(child)
        
        traverse_node(root)
        # 将集合转换回列表形式返回
        bounds_list = [list(bounds) for bounds in bounds_set]
        return bounds_list
    
    except Exception as e:
        print(f"解析层次结构时出错: {str(e)}")
        return []

def find_clicked_element(hierarchy_xml, click_x, click_y):
    bounds_list = extract_all_bounds(hierarchy_xml, need_clickable=True)

    smallest_bounds = None
    smallest_area = float('inf')

    for bounds in bounds_list:
        if is_point_in_bounds(click_x, click_y, bounds):
            left, top, right, bottom = bounds
            area = (right - left) * (bottom - top)
            if area < smallest_area:
                smallest_area = area
                smallest_bounds = bounds

    return smallest_bounds
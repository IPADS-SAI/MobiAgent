from utils.omni_utils import get_som_labeled_img, check_ocr_box, get_yolo_model
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
detect_model_path='./weights/icon_detect/model.pt'
caption_model_path='./weights/icon_caption_florence'

som_model = get_yolo_model(detect_model_path)
som_model.to(device)

def extract_all_bounds(screenshot_path):
    """提取截图中的所有边界框信息"""
    image = Image.open(screenshot_path).convert('RGB')
    
    # OCR检测文本框
    (text, ocr_bbox), _ = check_ocr_box(
        image,
        display_img=False, 
        output_bb_format='xyxy', 
        easyocr_args={'text_threshold': 0.9}, 
        use_paddleocr=True,
    )

    # YOLO检测UI元素
    _, _, parsed_content_list = get_som_labeled_img(
        image, 
        som_model, 
        BOX_TRESHOLD=0.1, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        ocr_text=text,
        use_local_semantics=False,
        iou_threshold=0.7, 
        scale_img=False
    )

    # 提取边界框并转换为绝对坐标
    image_width, image_height = image.size
    bounds_list = []

    for item in parsed_content_list:
        bbox = item.get('bbox')
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            # 转换为绝对坐标
            left = int(x1 * image_width)
            top = int(y1 * image_height)
            right = int(x2 * image_width)
            bottom = int(y2 * image_height)
            bounds_list.append([left, top, right, bottom])

    return bounds_list

def find_clicked_element(bounds_list, click_x, click_y):
    """找到包含点击位置的最小边界框"""
    smallest_bounds = None
    smallest_area = float('inf')

    for bounds in bounds_list:
        left, top, right, bottom = bounds
        # 检查点击位置是否在边界框内
        if left <= click_x <= right and top <= click_y <= bottom:
            area = (right - left) * (bottom - top)
            if area < smallest_area:
                smallest_area = area
                smallest_bounds = bounds

    return smallest_bounds
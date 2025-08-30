from .utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
from typing import Dict
class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(model_name=config['caption_model_name'], model_name_or_path=config['caption_model_path'], device=device)
        print('Omniparser initialized')

    def parse(self, image):
        text, ocr_bbox = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold': 0.9,'paragraph': False}, use_paddleocr=True)
        parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD=self.config['box_threshold'], ocr_bbox=ocr_bbox, caption_model_processor=self.caption_model_processor, ocr_text=text, use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)

        return parsed_content_list

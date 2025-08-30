from typing import Dict, List
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np

class UIElement:
    def __init__(self, bbox: List[int], content: str = None, sub_img: Image.Image = None):
        # bbox: [x1, y1, x2, y2], round(relative * 1000) format
        self.bbox = bbox
        # content: icon caption or ocr result
        self.content = content
        # sub_img: cropped image of the UI element
        self.sub_img = sub_img

    def __eq__(self, other):
        if self.sub_img is not None and other.sub_img is not None:
            img1 = np.array(self.sub_img)
            img2 = np.array(other.sub_img)

            if img1.shape != img2.shape:
                img2_pil_resized = other.sub_img.resize(self.sub_img.size, Image.Resampling.LANCZOS)
                img2 = np.array(img2_pil_resized)
    
            similarity = ssim(img1, img2, channel_axis=2, data_range=255)
            return similarity > 0.9
        if self.content is not None and other.content is not None:
            return self.content == other.content
        return True

class Action:
    def __init__(self, name: str, param: Dict[str, str], extra: Dict[str, str] = None):
        self.name = name
        self.param = param
        self.extra = extra
        self.target_elem = None

    def extract_target_elem(self, screen, parser):
        pass

    def __eq__(self, other):
        return self.name == other.name and self.param == other.param

    def __str__(self):
        return f"{self.name}({', '.join([f'{k}={v}' for k, v in self.param.items()])})"

class GeneralAgentAction(Action):
    def __init__(self, name: str, param: Dict[str, str], extra: Dict[str, str] = None):
        super().__init__(name, param, extra)

    def extract_target_elem(self, screen, parser):
        if self.target_elem is not None:
            return
        if screen is None:
            return
        if self.name not in ["click", "longclick"]:
            return
        bbox = self.param['bbox']
        target_element = self.param['target_element']
        sub_img = screen.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        self.target_elem = UIElement(bbox, target_element, sub_img)

    def __eq__(self, other):
        if not isinstance(other, GeneralAgentAction):
            return False
        if self.name != other.name:
            return False
        if self.name in ["click", "longclick"]:
            box1 = self.param["bbox"]
            box2 = other.param["bbox"]
            center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
            center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
            center1_in_box2 = (box2[0] <= center1[0] <= box2[2] and box2[1] <= center1[1] <= box2[3])
            center2_in_box1 = (box1[0] <= center2[0] <= box1[2] and box1[1] <= center2[1] <= box1[3])
            return center1_in_box2 and center2_in_box1
        return self.param == other.param
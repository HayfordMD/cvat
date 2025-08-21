import cv2
import numpy as np
from PIL import Image
import easyocr
from paddleocr import PaddleOCR
import re
import torch

class ModelHandler:
    def __init__(self, labels):
        self.labels = labels
        
        # Initialize both OCR engines for better coverage
        # EasyOCR is good for general text detection
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # PaddleOCR is excellent for detecting numbers
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=torch.cuda.is_available(),
            show_log=False
        )
        
        # Pattern for matching jersey numbers (1-99, possibly with leading zero)
        self.jersey_pattern = re.compile(r'^[0-9]{1,2}$')
    
    def is_jersey_number(self, text):
        """Check if detected text is likely a jersey number"""
        # Clean the text
        text = text.strip()
        
        # Check if it matches jersey number pattern
        if self.jersey_pattern.match(text):
            num = int(text)
            # Jersey numbers are typically 0-99
            return 0 <= num <= 99
        return False
    
    def process_with_easyocr(self, image_np):
        """Process image with EasyOCR"""
        results = []
        
        try:
            # Run OCR
            ocr_results = self.reader.readtext(image_np)
            
            for (bbox, text, confidence) in ocr_results:
                # Check if detected text is a jersey number
                if self.is_jersey_number(text):
                    # Convert bbox format to CVAT format
                    points = np.array(bbox).flatten().tolist()
                    
                    results.append({
                        'text': text,
                        'confidence': confidence,
                        'points': points,
                        'bbox': bbox
                    })
        except Exception as e:
            print(f"EasyOCR error: {e}")
        
        return results
    
    def process_with_paddle(self, image_np):
        """Process image with PaddleOCR"""
        results = []
        
        try:
            # Run OCR
            ocr_results = self.paddle_ocr.ocr(image_np, cls=True)
            
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    bbox, (text, confidence) = line
                    
                    # Check if detected text is a jersey number
                    if self.is_jersey_number(text):
                        # Convert bbox to flat list
                        points = np.array(bbox).flatten().tolist()
                        
                        results.append({
                            'text': text,
                            'confidence': confidence,
                            'points': points,
                            'bbox': bbox
                        })
        except Exception as e:
            print(f"PaddleOCR error: {e}")
        
        return results
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced, image_np
    
    def merge_detections(self, detections, iou_threshold=0.5):
        """Merge overlapping detections from different OCR engines"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Start with the current detection
            best_det = det1
            used.add(i)
            
            # Check for overlapping detections
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate IoU
                iou = self.calculate_iou(det1['bbox'], det2['bbox'])
                
                if iou > iou_threshold:
                    # If same number detected, boost confidence
                    if det1['text'] == det2['text']:
                        best_det['confidence'] = min(1.0, best_det['confidence'] * 1.1)
                    used.add(j)
            
            merged.append(best_det)
        
        return merged
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        # Convert to x1, y1, x2, y2 format
        box1_flat = np.array(box1).reshape(-1, 2)
        box2_flat = np.array(box2).reshape(-1, 2)
        
        x1_min, y1_min = box1_flat.min(axis=0)
        x1_max, y1_max = box1_flat.max(axis=0)
        
        x2_min, y2_min = box2_flat.min(axis=0)
        x2_max, y2_max = box2_flat.max(axis=0)
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def infer(self, image, confidence_threshold=0.5):
        """Run inference on the input image"""
        # Preprocess image
        enhanced_gray, original_np = self.preprocess_image(image)
        
        # Convert enhanced grayscale back to RGB for OCR
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        
        all_detections = []
        
        # Process with both OCR engines
        easyocr_results = self.process_with_easyocr(enhanced_rgb)
        paddle_results = self.process_with_paddle(original_np)
        
        # Also try on original image for better color detection
        easyocr_orig = self.process_with_easyocr(original_np)
        
        # Combine all detections
        all_detections.extend(easyocr_results)
        all_detections.extend(paddle_results)
        all_detections.extend(easyocr_orig)
        
        # Merge overlapping detections
        merged_detections = self.merge_detections(all_detections)
        
        # Format results for CVAT
        results = []
        for det in merged_detections:
            if det['confidence'] >= confidence_threshold:
                results.append({
                    "confidence": float(det['confidence']),
                    "label": self.labels.get(1, "jersey_number"),
                    "points": det['points'],
                    "type": "polygon",
                    "attributes": {
                        "number": det['text']
                    }
                })
        
        return results
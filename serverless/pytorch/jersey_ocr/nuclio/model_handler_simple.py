import cv2
import numpy as np
from PIL import Image
import easyocr
import re

class ModelHandler:
    def __init__(self, labels):
        self.labels = labels
        
        # Initialize EasyOCR
        try:
            import torch
            gpu = torch.cuda.is_available()
        except:
            gpu = False
            
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        
        # Pattern for matching jersey numbers
        self.jersey_pattern = re.compile(r'^[0-9]{1,2}$')
    
    def is_jersey_number(self, text):
        """Check if detected text is likely a jersey number"""
        text = text.strip()
        
        # Check if it matches jersey number pattern
        if self.jersey_pattern.match(text):
            num = int(text)
            return 0 <= num <= 99
        
        # Also check for partial matches (sometimes OCR gets O instead of 0)
        text_cleaned = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
        if text_cleaned != text and self.jersey_pattern.match(text_cleaned):
            num = int(text_cleaned)
            return 0 <= num <= 99
            
        return False
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Create multiple preprocessing versions
        versions = []
        
        # Original
        versions.append(image_np)
        
        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()
        
        # Version 1: Simple threshold
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        versions.append(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB))
        
        # Version 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        versions.append(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2RGB))
        
        # Version 3: CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        versions.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
        
        # Version 4: Inverted
        inverted = cv2.bitwise_not(gray)
        versions.append(cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB))
        
        return versions
    
    def process_image(self, image_np):
        """Process image with EasyOCR"""
        results = []
        
        try:
            # Run OCR with different settings
            ocr_results = self.reader.readtext(
                image_np,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7
            )
            
            for (bbox, text, confidence) in ocr_results:
                # Clean the text
                text_cleaned = text.strip()
                
                # Check if it's a jersey number
                if self.is_jersey_number(text_cleaned):
                    # Convert bbox format to CVAT format
                    points = np.array(bbox).flatten().tolist()
                    
                    # Extract the actual number
                    number = text_cleaned.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
                    if self.jersey_pattern.match(number):
                        results.append({
                            'text': number,
                            'confidence': confidence,
                            'points': points,
                            'bbox': bbox
                        })
                        
                # Also check for individual digits (sometimes jerseys are detected as separate digits)
                elif len(text_cleaned) == 1 and text_cleaned.isdigit():
                    points = np.array(bbox).flatten().tolist()
                    results.append({
                        'text': text_cleaned,
                        'confidence': confidence * 0.8,  # Lower confidence for single digits
                        'points': points,
                        'bbox': bbox,
                        'single_digit': True
                    })
                    
        except Exception as e:
            print(f"OCR error: {e}")
        
        return results
    
    def merge_nearby_digits(self, detections, distance_threshold=50):
        """Merge nearby single digits into jersey numbers"""
        single_digits = [d for d in detections if d.get('single_digit', False)]
        multi_digits = [d for d in detections if not d.get('single_digit', False)]
        
        if len(single_digits) < 2:
            return multi_digits + single_digits
        
        # Sort single digits by x-coordinate
        single_digits.sort(key=lambda d: min(d['bbox'][0][0], d['bbox'][1][0], d['bbox'][2][0], d['bbox'][3][0]))
        
        merged = []
        i = 0
        while i < len(single_digits):
            current = single_digits[i]
            
            # Check if next digit is close enough
            if i + 1 < len(single_digits):
                next_digit = single_digits[i + 1]
                
                # Calculate distance between bounding boxes
                curr_right = max(current['bbox'][0][0], current['bbox'][1][0], current['bbox'][2][0], current['bbox'][3][0])
                next_left = min(next_digit['bbox'][0][0], next_digit['bbox'][1][0], next_digit['bbox'][2][0], next_digit['bbox'][3][0])
                
                distance = next_left - curr_right
                
                if distance < distance_threshold:
                    # Merge the two digits
                    merged_text = current['text'] + next_digit['text']
                    if self.is_jersey_number(merged_text):
                        # Create merged bounding box
                        all_points = current['bbox'] + next_digit['bbox']
                        xs = [p[0] for p in all_points]
                        ys = [p[1] for p in all_points]
                        
                        merged_bbox = [
                            [min(xs), min(ys)],
                            [max(xs), min(ys)],
                            [max(xs), max(ys)],
                            [min(xs), max(ys)]
                        ]
                        
                        merged.append({
                            'text': merged_text,
                            'confidence': (current['confidence'] + next_digit['confidence']) / 2,
                            'points': np.array(merged_bbox).flatten().tolist(),
                            'bbox': merged_bbox
                        })
                        i += 2
                        continue
            
            # If not merged, add as single digit
            merged.append(current)
            i += 1
        
        return multi_digits + merged
    
    def infer(self, image, confidence_threshold=0.5):
        """Run inference on the input image"""
        # Get multiple preprocessed versions
        image_versions = self.preprocess_image(image)
        
        all_detections = []
        
        # Process each version
        for version in image_versions:
            detections = self.process_image(version)
            all_detections.extend(detections)
        
        # Remove duplicates based on similar bounding boxes
        unique_detections = []
        used = set()
        
        for i, det1 in enumerate(all_detections):
            if i in used:
                continue
                
            # Check if this detection overlaps with any already added
            is_duplicate = False
            for det2 in unique_detections:
                if det1['text'] == det2['text']:
                    # Calculate center distance
                    c1 = np.array(det1['bbox']).mean(axis=0)
                    c2 = np.array(det2['bbox']).mean(axis=0)
                    dist = np.linalg.norm(c1 - c2)
                    
                    if dist < 30:  # If centers are close, it's a duplicate
                        is_duplicate = True
                        # Keep the one with higher confidence
                        if det1['confidence'] > det2['confidence']:
                            unique_detections.remove(det2)
                            unique_detections.append(det1)
                        break
            
            if not is_duplicate:
                unique_detections.append(det1)
            used.add(i)
        
        # Merge nearby single digits
        final_detections = self.merge_nearby_digits(unique_detections)
        
        # Format results for CVAT
        results = []
        for det in final_detections:
            if det['confidence'] >= confidence_threshold and not det.get('single_digit', False):
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
#!/usr/bin/env python3
"""
Advanced Computer Vision Pipelines

Comprehensive computer vision toolkit covering object detection, segmentation,
face recognition, medical imaging, and advanced neural network architectures.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import detection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
from collections import defaultdict

# Scientific computing
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage
from scipy import spatial

# Deep learning frameworks
try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Specialized CV libraries
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Medical imaging
try:
    import pydicom
    import SimpleITK as sitk
    MEDICAL_IMAGING_AVAILABLE = True
except ImportError:
    MEDICAL_IMAGING_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Container for object detection results"""
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    class_names: List[str]
    image_size: Tuple[int, int]
    inference_time: float

@dataclass
class SegmentationResult:
    """Container for segmentation results"""
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    class_names: List[str]
    inference_time: float

class AdvancedImagePreprocessor:
    """Advanced image preprocessing with augmentation strategies"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.setup_transforms()
    
    def setup_transforms(self):
        """Setup various transformation pipelines"""
        
        # Basic preprocessing
        self.basic_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Training augmentations
        self.train_transform = A.Compose([
            A.Resize(*self.target_size),
            A.RandomRotate90(),
            A.Flip(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, 
                             rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Medical imaging specific transforms
        self.medical_transform = A.Compose([
            A.Resize(*self.target_size),
            A.RandomRotate90(),
            A.Flip(),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ], p=0.8),
            A.CLAHE(p=0.8),
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def preprocess_image(self, image: Union[np.ndarray, PIL.Image.Image], 
                        transform_type: str = "basic") -> torch.Tensor:
        """Preprocess single image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if transform_type == "basic":
            return self.basic_transform(image)
        elif transform_type == "train":
            return self.train_transform(image=np.array(image))["image"]
        elif transform_type == "medical":
            return self.medical_transform(image=np.array(image))["image"]
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced image enhancement techniques"""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            (img_float * 255).astype(np.uint8), None, 10, 10, 7, 21
        )
        
        # CLAHE for contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)

class ObjectDetectionPipeline:
    """Advanced object detection pipeline with multiple models"""
    
    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn", 
                 confidence_threshold: float = 0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.load_class_names()
    
    def load_model(self):
        """Load pretrained detection model"""
        if self.model_name == "fasterrcnn_resnet50_fpn":
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif self.model_name == "maskrcnn_resnet50_fpn":
            self.model = detection.maskrcnn_resnet50_fpn(pretrained=True)
        elif self.model_name == "retinanet_resnet50_fpn":
            self.model = detection.retinanet_resnet50_fpn(pretrained=True)
        elif self.model_name == "ssd300_vgg16":
            self.model = detection.ssd300_vgg16(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded {self.model_name} on {self.device}")
    
    def load_class_names(self):
        """Load COCO class names"""
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Perform object detection on image"""
        start_time = time.time()
        
        # Preprocess image
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Extract results
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter by confidence
        keep_indices = scores >= self.confidence_threshold
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        inference_time = time.time() - start_time
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=self.class_names,
            image_size=image.shape[:2],
            inference_time=inference_time
        )
    
    def visualize_detections(self, image: np.ndarray, 
                           detection_result: DetectionResult) -> np.ndarray:
        """Visualize detection results on image"""
        vis_image = image.copy()
        
        for i, (box, score, label) in enumerate(zip(
            detection_result.boxes, 
            detection_result.scores, 
            detection_result.labels
        )):
            x1, y1, x2, y2 = box.astype(int)
            class_name = detection_result.class_names[label]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(vis_image, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return vis_image

class InstanceSegmentationPipeline:
    """Instance segmentation using Mask R-CNN"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load pretrained Mask R-CNN model"""
        self.model = detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO class names
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Perform instance segmentation"""
        start_time = time.time()
        
        # Preprocess
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Extract results
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        
        # Filter by confidence
        keep_indices = scores >= self.confidence_threshold
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        masks = masks[keep_indices]
        
        inference_time = time.time() - start_time
        
        return SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=self.class_names,
            inference_time=inference_time
        )
    
    def visualize_segmentation(self, image: np.ndarray, 
                             seg_result: SegmentationResult) -> np.ndarray:
        """Visualize segmentation results"""
        vis_image = image.copy()
        
        # Generate colors for different instances
        colors = plt.cm.tab20(np.linspace(0, 1, len(seg_result.masks)))[:, :3] * 255
        
        for i, (mask, box, score, label) in enumerate(zip(
            seg_result.masks, seg_result.boxes, 
            seg_result.scores, seg_result.labels
        )):
            # Apply mask
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            color_mask = np.zeros_like(vis_image)
            color_mask[mask_binary == 1] = colors[i % len(colors)]
            
            # Blend with original image
            vis_image = cv2.addWeighted(vis_image, 0.8, color_mask.astype(np.uint8), 0.2, 0)
            
            # Draw bounding box and label
            x1, y1, x2, y2 = box.astype(int)
            class_name = seg_result.class_names[label]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[i % len(colors)], 2)
            
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(vis_image, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)
        
        return vis_image

class FaceAnalysisPipeline:
    """Comprehensive face analysis pipeline"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
    
    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces.tolist()
    
    def detect_faces_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        detections = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                confidence = detection.score[0]
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence
                })
        
        return detections
    
    def extract_face_landmarks(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract facial landmarks using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        landmarks_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    landmarks.append([x, y])
                landmarks_list.append(np.array(landmarks))
        
        return landmarks_list
    
    def recognize_faces(self, image: np.ndarray, 
                       known_encodings: List[np.ndarray], 
                       known_names: List[str]) -> List[Dict]:
        """Recognize faces using face_recognition library"""
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError("face_recognition library not available")
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        recognized_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    confidence = 1 - distances[best_match_index]
            
            recognized_faces.append({
                'bbox': (left, top, right - left, bottom - top),
                'name': name,
                'confidence': confidence
            })
        
        return recognized_faces
    
    def estimate_age_gender(self, face_image: np.ndarray) -> Dict:
        """Estimate age and gender (placeholder - would use specialized models)"""
        # This is a placeholder - in practice, you'd use models like:
        # - Age estimation: DEX, IMDB-WIKI
        # - Gender classification: VGG-Face, FaceNet
        
        # Simulate age/gender prediction
        estimated_age = np.random.randint(18, 80)
        estimated_gender = np.random.choice(['Male', 'Female'])
        confidence = np.random.uniform(0.7, 0.95)
        
        return {
            'age': estimated_age,
            'gender': estimated_gender,
            'confidence': confidence
        }
    
    def analyze_face_emotions(self, face_image: np.ndarray) -> Dict:
        """Analyze facial emotions (placeholder)"""
        # Placeholder - would use models like FER2013, AffectNet
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        scores = np.random.dirichlet(np.ones(len(emotions)))
        
        emotion_scores = dict(zip(emotions, scores))
        predicted_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'emotion_scores': emotion_scores,
            'predicted_emotion': predicted_emotion[0],
            'confidence': predicted_emotion[1]
        }

class MedicalImageAnalysis:
    """Medical image analysis pipeline"""
    
    def __init__(self):
        self.setup_models()
    
    def setup_models(self):
        """Setup medical image analysis models"""
        # Placeholder for medical imaging models
        # In practice, you'd load specialized models for:
        # - Chest X-ray analysis
        # - MRI/CT scan analysis
        # - Pathology image analysis
        # - Retinal image analysis
        pass
    
    def load_dicom_image(self, dicom_path: str) -> Tuple[np.ndarray, Dict]:
        """Load and preprocess DICOM image"""
        if not MEDICAL_IMAGING_AVAILABLE:
            raise ImportError("pydicom and SimpleITK not available")
        
        # Load DICOM file
        dicom_data = pydicom.dcmread(dicom_path)
        
        # Extract pixel data
        pixel_array = dicom_data.pixel_array
        
        # Normalize pixel values
        if dicom_data.get('RescaleIntercept') and dicom_data.get('RescaleSlope'):
            pixel_array = pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
        
        # Convert to 8-bit for visualization
        pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Extract metadata
        metadata = {
            'patient_id': dicom_data.get('PatientID', 'Unknown'),
            'study_date': dicom_data.get('StudyDate', 'Unknown'),
            'modality': dicom_data.get('Modality', 'Unknown'),
            'body_part': dicom_data.get('BodyPartExamined', 'Unknown'),
            'rows': dicom_data.get('Rows', 0),
            'columns': dicom_data.get('Columns', 0),
            'pixel_spacing': dicom_data.get('PixelSpacing', [1, 1])
        }
        
        return pixel_array, metadata
    
    def analyze_chest_xray(self, image: np.ndarray) -> Dict:
        """Analyze chest X-ray for pathologies"""
        # Placeholder for chest X-ray analysis
        # Would use models like CheXNet, DenseNet-121
        
        pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
        # Simulate predictions
        scores = np.random.uniform(0, 1, len(pathologies))
        predictions = dict(zip(pathologies, scores))
        
        # Find most likely pathology
        max_pathology = max(predictions.items(), key=lambda x: x[1])
        
        return {
            'pathology_scores': predictions,
            'most_likely_pathology': max_pathology[0],
            'confidence': max_pathology[1],
            'normal_probability': 1 - max(scores)
        }
    
    def segment_organs(self, image: np.ndarray, organ_type: str = 'lung') -> np.ndarray:
        """Segment organs in medical images"""
        # Placeholder for organ segmentation
        # Would use U-Net, V-Net, or specialized architectures
        
        if organ_type == 'lung':
            # Simple lung segmentation using thresholding (placeholder)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Threshold to create binary mask
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return cleaned
        
        else:
            raise ValueError(f"Organ type '{organ_type}' not supported")
    
    def measure_anatomical_structures(self, segmentation_mask: np.ndarray, 
                                    pixel_spacing: List[float] = [1.0, 1.0]) -> Dict:
        """Measure anatomical structures from segmentation"""
        # Find contours
        contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        measurements = {}
        for i, contour in enumerate(contours):
            # Calculate area (in pixels and physical units)
            area_pixels = cv2.contourArea(contour)
            area_mm2 = area_pixels * pixel_spacing[0] * pixel_spacing[1]
            
            # Calculate perimeter
            perimeter_pixels = cv2.arcLength(contour, True)
            perimeter_mm = perimeter_pixels * np.mean(pixel_spacing)
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            width_mm = w * pixel_spacing[0]
            height_mm = h * pixel_spacing[1]
            
            # Calculate shape descriptors
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area_pixels / hull_area if hull_area > 0 else 0
            
            # Circularity
            circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2) if perimeter_pixels > 0 else 0
            
            measurements[f'structure_{i}'] = {
                'area_pixels': area_pixels,
                'area_mm2': area_mm2,
                'perimeter_pixels': perimeter_pixels,
                'perimeter_mm': perimeter_mm,
                'width_mm': width_mm,
                'height_mm': height_mm,
                'solidity': solidity,
                'circularity': circularity
            }
        
        return measurements

class AdvancedCVPipeline:
    """Main pipeline orchestrating all CV components"""
    
    def __init__(self):
        self.preprocessor = AdvancedImagePreprocessor()
        self.object_detector = ObjectDetectionPipeline()
        self.segmentation = InstanceSegmentationPipeline()
        self.face_analyzer = FaceAnalysisPipeline()
        self.medical_analyzer = MedicalImageAnalysis()
        
        self.results_cache = {}
    
    def process_image(self, image_path: str, analysis_types: List[str]) -> Dict:
        """Process image with specified analysis types"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        results = {
            'image_path': image_path,
            'image_shape': image.shape,
            'timestamp': time.time()
        }
        
        # Object detection
        if 'object_detection' in analysis_types:
            detection_result = self.object_detector.detect(image)
            results['object_detection'] = {
                'detections': len(detection_result.boxes),
                'inference_time': detection_result.inference_time,
                'objects_found': [
                    self.object_detector.class_names[label] 
                    for label in detection_result.labels
                ]
            }
            
            # Visualize and save
            vis_image = self.object_detector.visualize_detections(image, detection_result)
            output_path = image_path.replace('.', '_object_detection.')
            cv2.imwrite(output_path, vis_image)
            results['object_detection']['visualization'] = output_path
        
        # Instance segmentation
        if 'segmentation' in analysis_types:
            seg_result = self.segmentation.segment(image)
            results['segmentation'] = {
                'instances': len(seg_result.masks),
                'inference_time': seg_result.inference_time,
                'objects_segmented': [
                    self.segmentation.class_names[label] 
                    for label in seg_result.labels
                ]
            }
            
            # Visualize and save
            vis_image = self.segmentation.visualize_segmentation(image, seg_result)
            output_path = image_path.replace('.', '_segmentation.')
            cv2.imwrite(output_path, vis_image)
            results['segmentation']['visualization'] = output_path
        
        # Face analysis
        if 'face_analysis' in analysis_types:
            faces_opencv = self.face_analyzer.detect_faces_opencv(image)
            results['face_analysis'] = {
                'faces_detected': len(faces_opencv),
                'face_locations': faces_opencv
            }
            
            # Extract face landmarks if MediaPipe is available
            if MEDIAPIPE_AVAILABLE:
                landmarks = self.face_analyzer.extract_face_landmarks(image)
                results['face_analysis']['landmarks_extracted'] = len(landmarks)
        
        # Medical image analysis (if DICOM)
        if 'medical_analysis' in analysis_types and image_path.lower().endswith('.dcm'):
            try:
                pixel_array, metadata = self.medical_analyzer.load_dicom_image(image_path)
                chest_analysis = self.medical_analyzer.analyze_chest_xray(pixel_array)
                
                results['medical_analysis'] = {
                    'metadata': metadata,
                    'pathology_analysis': chest_analysis
                }
            except Exception as e:
                results['medical_analysis'] = {'error': str(e)}
        
        return results
    
    def batch_process(self, image_directory: str, 
                     analysis_types: List[str],
                     output_file: str = 'batch_results.json') -> Dict:
        """Process multiple images in batch"""
        image_dir = Path(image_directory)
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
        
        image_files = [
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in supported_formats
        ]
        
        batch_results = {
            'total_images': len(image_files),
            'analysis_types': analysis_types,
            'start_time': time.time(),
            'results': {}
        }
        
        for i, image_file in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.process_image(str(image_file), analysis_types)
                batch_results['results'][image_file.name] = result
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
                batch_results['results'][image_file.name] = {'error': str(e)}
        
        batch_results['end_time'] = time.time()
        batch_results['total_time'] = batch_results['end_time'] - batch_results['start_time']
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        return batch_results
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=== Computer Vision Analysis Report ===\n")
        
        if 'total_images' in results:
            # Batch processing report
            report.append(f"Batch Processing Summary:")
            report.append(f"Total Images: {results['total_images']}")
            report.append(f"Processing Time: {results.get('total_time', 0):.2f} seconds")
            report.append(f"Analysis Types: {', '.join(results['analysis_types'])}")
            report.append("")
            
            # Summary statistics
            successful = sum(1 for r in results['results'].values() if 'error' not in r)
            failed = results['total_images'] - successful
            
            report.append(f"Success Rate: {successful}/{results['total_images']} ({successful/results['total_images']*100:.1f}%)")
            report.append(f"Failed: {failed}")
            report.append("")
            
            # Object detection summary
            if 'object_detection' in results['analysis_types']:
                total_objects = sum(
                    r.get('object_detection', {}).get('detections', 0)
                    for r in results['results'].values()
                    if 'error' not in r
                )
                report.append(f"Total Objects Detected: {total_objects}")
                
                # Most common objects
                all_objects = []
                for r in results['results'].values():
                    if 'error' not in r and 'object_detection' in r:
                        all_objects.extend(r['object_detection'].get('objects_found', []))
                
                if all_objects:
                    from collections import Counter
                    object_counts = Counter(all_objects)
                    report.append("Most Common Objects:")
                    for obj, count in object_counts.most_common(5):
                        report.append(f"  {obj}: {count}")
                report.append("")
            
            # Face analysis summary
            if 'face_analysis' in results['analysis_types']:
                total_faces = sum(
                    r.get('face_analysis', {}).get('faces_detected', 0)
                    for r in results['results'].values()
                    if 'error' not in r
                )
                images_with_faces = sum(
                    1 for r in results['results'].values()
                    if 'error' not in r and r.get('face_analysis', {}).get('faces_detected', 0) > 0
                )
                report.append(f"Total Faces Detected: {total_faces}")
                report.append(f"Images with Faces: {images_with_faces}")
                report.append("")
        
        else:
            # Single image report
            report.append(f"Image: {results.get('image_path', 'Unknown')}")
            report.append(f"Shape: {results.get('image_shape', 'Unknown')}")
            report.append("")
            
            for analysis_type, result in results.items():
                if analysis_type in ['image_path', 'image_shape', 'timestamp']:
                    continue
                
                report.append(f"{analysis_type.replace('_', ' ').title()}:")
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key != 'visualization':
                            report.append(f"  {key}: {value}")
                report.append("")
        
        return "\n".join(report)

# Example usage and testing
def main():
    """Example usage of the computer vision pipeline"""
    
    # Initialize pipeline
    cv_pipeline = AdvancedCVPipeline()
    
    # Example 1: Single image analysis
    print("=== Single Image Analysis ===")
    # Note: Replace with actual image path
    # image_path = "example_image.jpg"
    # analysis_types = ['object_detection', 'segmentation', 'face_analysis']
    # results = cv_pipeline.process_image(image_path, analysis_types)
    # print(cv_pipeline.generate_report(results))
    
    # Example 2: Batch processing
    print("\n=== Batch Processing Example ===")
    # Note: Replace with actual directory path
    # image_directory = "images/"
    # analysis_types = ['object_detection', 'face_analysis']
    # batch_results = cv_pipeline.batch_process(image_directory, analysis_types)
    # print(cv_pipeline.generate_report(batch_results))
    
    # Example 3: Medical image analysis
    print("\n=== Medical Image Analysis Example ===")
    medical_analyzer = MedicalImageAnalysis()
    
    # Create synthetic medical image for demonstration
    synthetic_xray = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    
    # Analyze chest X-ray
    chest_analysis = medical_analyzer.analyze_chest_xray(synthetic_xray)
    print("Chest X-ray Analysis:")
    print(f"Most likely pathology: {chest_analysis['most_likely_pathology']}")
    print(f"Confidence: {chest_analysis['confidence']:.3f}")
    print(f"Normal probability: {chest_analysis['normal_probability']:.3f}")
    
    # Segment lungs
    lung_mask = medical_analyzer.segment_organs(synthetic_xray, 'lung')
    measurements = medical_analyzer.measure_anatomical_structures(lung_mask)
    print(f"\nLung measurements: {len(measurements)} structures found")
    
    # Example 4: Face analysis
    print("\n=== Face Analysis Example ===")
    face_analyzer = FaceAnalysisPipeline()
    
    # Create synthetic face image
    synthetic_face = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Detect faces using OpenCV
    faces = face_analyzer.detect_faces_opencv(synthetic_face)
    print(f"OpenCV detected {len(faces)} faces")
    
    # Analyze emotions (placeholder)
    emotion_analysis = face_analyzer.analyze_face_emotions(synthetic_face)
    print(f"Predicted emotion: {emotion_analysis['predicted_emotion']} "
          f"(confidence: {emotion_analysis['confidence']:.3f})")
    
    print("\nPipeline demonstration complete!")

if __name__ == "__main__":
    main() 
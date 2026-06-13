import subprocess
import sys
import os

# Fix for Streamlit Cloud: ultralytics installs opencv-python which needs
# libGL.so.1 (unavailable on the container). Replace with headless version.
try:
    import cv2
except ImportError:
    # Try replacing opencv-python with headless using uv (Streamlit Cloud's installer)
    subprocess.run(["uv", "pip", "install", "--reinstall", "opencv-python-headless"])
    # Fallback to pip if uv is not available
    subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "opencv-python-headless"])
    # Clear all partially loaded cv2 modules from cache
    for mod in list(sys.modules.keys()):
        if mod == "cv2" or mod.startswith("cv2."):
            del sys.modules[mod]
    # Set library path in case libGL is in a non-standard location
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
    import cv2

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import io
import pandas as pd
import zipfile
import tempfile
from datetime import datetime
from fpdf import FPDF
import numpy as np
import base64
import math
from skimage.metrics import structural_similarity as ssim

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="SmartDetect - AI Image Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ========== SHARED MODEL LOADER ==========

@st.cache_resource
def load_yolo_model():
    """Load and cache the YOLOv8 model once for the whole app."""
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


# ========== IMPROVED EARTH PRO ANALYSIS FUNCTIONS ==========

def merge_overlapping_boxes(changes, iou_threshold=0.3):
    """Merge overlapping detection boxes using Non-Maximum Suppression"""
    if not changes:
        return []

    # Convert to format for NMS
    boxes = []
    for change in changes:
        x = change['x']
        y = change['y']
        w = change['width']
        h = change['height']
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append([x1, y1, x2, y2, change['confidence']])

    boxes = np.array(boxes)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes[:, :4].tolist(),
        boxes[:, 4].tolist(),
        score_threshold=0.1,
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return []

    # Flatten indices if needed
    if isinstance(indices, tuple):
        indices = indices[0] if len(indices) > 0 else []
    indices = indices.flatten() if hasattr(indices, 'flatten') else indices

    merged_changes = []
    for idx in indices:
        idx = int(idx)
        change = changes[idx]
        merged_changes.append(change)

    return merged_changes


def detect_changes_comprehensive(img_old, img_new, min_area=50):
    """
    COMPREHENSIVE BUILDING DETECTION - Detects all buildings, large and small
    Uses 6 different detection methods and combines them intelligently
    """
    old_arr = np.array(img_old)
    new_arr = np.array(img_new)

    if old_arr.shape != new_arr.shape:
        new_arr = cv2.resize(new_arr, (old_arr.shape[1], old_arr.shape[0]))

    # Convert to grayscale
    gray_old = cv2.cvtColor(old_arr, cv2.COLOR_RGB2GRAY)
    gray_new = cv2.cvtColor(new_arr, cv2.COLOR_RGB2GRAY)

    # Advanced lighting normalization using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_old_norm = clahe.apply(gray_old)
    gray_new_norm = clahe.apply(gray_new)

    all_changes = []
    img_area = gray_new.shape[0] * gray_new.shape[1]
    max_area = img_area * 0.35  # Maximum 35% of image

    # ===== METHOD 1: Multi-Scale Intensity Difference =====
    diff_intensity = cv2.absdiff(gray_old_norm, gray_new_norm)

    # Use multiple thresholds to catch different change magnitudes
    thresholds = [3, 6, 10, 15, 25]
    for thresh_val in thresholds:
        _, binary = cv2.threshold(diff_intensity, thresh_val, 255, cv2.THRESH_BINARY)

        # Minimal morphology to preserve details
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                mask = np.zeros(gray_new.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_diff = cv2.mean(diff_intensity, mask=mask)[0]
                confidence = min(mean_diff / 60.0, 1.0)

                all_changes.append({
                    "x": int(x + w / 2), "y": int(y + h / 2),
                    "width": int(w), "height": int(h),
                    "area": int(area),
                    "confidence": float(max(0.20, confidence)),
                    "method": f"intensity_t{thresh_val}",
                })

    # ===== METHOD 2: Edge Structure Detection =====
    edges_old = cv2.Canny(gray_old_norm, 15, 80)  # Very sensitive
    edges_new = cv2.Canny(gray_new_norm, 15, 80)
    diff_edges = cv2.absdiff(edges_old, edges_new)

    _, binary_edges = cv2.threshold(diff_edges, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(diff_edges, mask=mask)[0]
            confidence = min(mean_diff / 80.0, 1.0)

            all_changes.append({
                "x": int(x + w / 2), "y": int(y + h / 2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.35, confidence)),
                "method": "edges",
            })

    # ===== METHOD 3: RGB Color Change Detection =====
    diff_color = cv2.absdiff(old_arr, new_arr)
    diff_color_gray = cv2.cvtColor(diff_color, cv2.COLOR_RGB2GRAY)

    _, binary_color = cv2.threshold(diff_color_gray, 8, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    binary_color = cv2.morphologyEx(binary_color, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(diff_color_gray, mask=mask)[0]
            confidence = min(mean_diff / 70.0, 1.0)

            all_changes.append({
                "x": int(x + w / 2), "y": int(y + h / 2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.25, confidence)),
                "method": "color",
            })

    # ===== METHOD 4: Gradient/Texture Detection =====
    sobelx_old = cv2.Sobel(gray_old_norm, cv2.CV_64F, 1, 0, ksize=3)
    sobely_old = cv2.Sobel(gray_old_norm, cv2.CV_64F, 0, 1, ksize=3)
    sobelx_new = cv2.Sobel(gray_new_norm, cv2.CV_64F, 1, 0, ksize=3)
    sobely_new = cv2.Sobel(gray_new_norm, cv2.CV_64F, 0, 1, ksize=3)

    gradient_old = np.sqrt(sobelx_old ** 2 + sobely_old ** 2)
    gradient_new = np.sqrt(sobelx_new ** 2 + sobely_new ** 2)
    gradient_diff = np.abs(gradient_new - gradient_old).astype(np.uint8)

    _, binary_grad = cv2.threshold(gradient_diff, 8, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary_grad = cv2.morphologyEx(binary_grad, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(gradient_diff, mask=mask)[0]
            confidence = min(mean_diff / 80.0, 1.0)

            all_changes.append({
                "x": int(x + w / 2), "y": int(y + h / 2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.30, confidence)),
                "method": "gradient",
            })

    # ===== METHOD 5: Adaptive Thresholding =====
    adaptive = cv2.adaptiveThreshold(
        diff_intensity, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 13, -2,
    )
    kernel = np.ones((2, 2), np.uint8)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(diff_intensity, mask=mask)[0]
            confidence = min(mean_diff / 70.0, 1.0)

            all_changes.append({
                "x": int(x + w / 2), "y": int(y + h / 2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.25, confidence)),
                "method": "adaptive",
            })

    # ===== METHOD 6: Laplacian (Detail Detection) =====
    lap_old = cv2.Laplacian(gray_old_norm, cv2.CV_64F)
    lap_new = cv2.Laplacian(gray_new_norm, cv2.CV_64F)
    lap_diff = np.abs(lap_new - lap_old).astype(np.uint8)

    _, binary_lap = cv2.threshold(lap_diff, 5, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    binary_lap = cv2.morphologyEx(binary_lap, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary_lap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            mask = np.zeros(gray_new.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_diff = cv2.mean(lap_diff, mask=mask)[0]
            confidence = min(mean_diff / 50.0, 1.0)

            all_changes.append({
                "x": int(x + w / 2), "y": int(y + h / 2),
                "width": int(w), "height": int(h),
                "area": int(area),
                "confidence": float(max(0.25, confidence)),
                "method": "laplacian",
            })

    # Merge overlapping detections
    merged_changes = merge_overlapping_boxes(all_changes, iou_threshold=0.4)

    return merged_changes


def detect_changes_yolo(img_old, img_new, model, min_confidence=0.15):
    """Enhanced YOLO detection with lower confidence threshold for buildings"""
    results_old = model(img_old, conf=min_confidence)[0]
    results_new = model(img_new, conf=min_confidence)[0]

    boxes_old = []
    if results_old.boxes is not None:
        for box in results_old.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()
            conf = float(box.conf[0])
            if conf >= min_confidence:
                boxes_old.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'class': cls_name,
                    'conf': conf,
                })

    boxes_new = []
    if results_new.boxes is not None:
        for box in results_new.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id].lower()
            conf = float(box.conf[0])
            if conf >= min_confidence:
                boxes_new.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'class': cls_name,
                    'conf': conf,
                })

    new_objects = []
    for box_new in boxes_new:
        is_new = True
        x_new_center = (box_new['xyxy'][0] + box_new['xyxy'][2]) / 2
        y_new_center = (box_new['xyxy'][1] + box_new['xyxy'][3]) / 2

        for box_old in boxes_old:
            x_old_center = (box_old['xyxy'][0] + box_old['xyxy'][2]) / 2
            y_old_center = (box_old['xyxy'][1] + box_old['xyxy'][3]) / 2

            # Check if same object (within tolerance)
            distance = np.sqrt((x_new_center - x_old_center) ** 2 + (y_new_center - y_old_center) ** 2)
            if distance < 30:  # Reduced tolerance
                is_new = False
                break

        if is_new:
            x0, y0, x1, y1 = box_new['xyxy']
            new_objects.append({
                "x": int((x0 + x1) / 2),
                "y": int((y0 + y1) / 2),
                "width": int(x1 - x0),
                "height": int(y1 - y0),
                "area": int((x1 - x0) * (y1 - y0)),
                "confidence": box_new['conf'],
                "class": box_new['class'],
                "type": f"New {box_new['class'].capitalize()}",
            })

    return new_objects


def classify_change_type(change, year_old, year_new):
    """Classify the type of change with improved categorization"""
    if 'type' in change and change['type']:
        return change['type']

    if 'class' in change:
        obj_type = change['class'].lower()
        if obj_type in ['building', 'house']:
            return "New Building"
        elif obj_type in ['road', 'street']:
            return "New Road"
        elif obj_type in ['tree', 'plant', 'vegetation', 'potted plant']:
            return "Vegetation Growth"
        elif obj_type in ['car', 'truck', 'bus', 'vehicle']:
            return "New Vehicle/Structure"
        else:
            return f"New {obj_type.capitalize()}"

    # Classify by size and shape
    area = change.get('area', change.get('width', 0) * change.get('height', 0))
    width = change.get('width', 0)
    height = change.get('height', 0)
    aspect_ratio = width / height if height > 0 else 1

    if area > 8000:
        if aspect_ratio > 2.5:
            return "New Road/Highway"
        else:
            return "New Large Building"
    elif area > 3000:
        if aspect_ratio > 3:
            return "New Road Segment"
        elif 0.6 < aspect_ratio < 1.5:
            return "New Medium Building"
        else:
            return "New Structure"
    elif area > 800:
        if aspect_ratio > 3.5:
            return "New Path/Lane"
        else:
            return "New Small Building"
    elif area > 200:
        if aspect_ratio > 4:
            return "New Narrow Path"
        else:
            return "New Small Shop/Structure"
    else:
        return "Minor Construction"


# ========== ORIGINAL SMARTDETECT FUNCTIONS ==========

def get_base64_image(image_path):
    """Convert local image to base64 for CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None


def get_math_animation_value():
    """Mathematical formulas for dynamic visual effects"""
    now = datetime.now()
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    sine_pulse = (math.sin(seconds * math.pi / 30) + 1) / 2
    golden_ratio = 1.618033988749
    golden_value = (seconds * golden_ratio) % 1
    fib_sequence = [0.1, 0.15, 0.2, 0.25, 0.35, 0.45]
    fib_index = int((seconds / 10) % len(fib_sequence))
    fib_opacity = fib_sequence[fib_index]

    return {
        'sine': sine_pulse,
        'golden': golden_value,
        'fib_opacity': fib_opacity,
        'rotation': (seconds * 0.5) % 360,
    }


def detect_cracks_opencv(image):
    """Detects cracks using computer vision (OpenCV) techniques"""
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    anomalies = []
    min_area = 100

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            anomalies.append({
                "x": x + w / 2,
                "y": y + h / 2,
                "width": w,
                "height": h,
                "confidence": 100.0,
                "class": "crack/defect",
            })
    return anomalies


def detect_stains_opencv(image):
    """Detects stains/discoloration using color statistics"""
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    median = cv2.medianBlur(blurred, 21)
    diff = cv2.absdiff(blurred, median)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anomalies = []
    min_area = 200

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            anomalies.append({
                "x": float(x + w / 2),
                "y": float(y + h / 2),
                "width": float(w),
                "height": float(h),
                "confidence": 100.0,
                "class": "stain/discoloration",
            })
    return anomalies


def compare_images_ssim(img1, img2):
    """Compare two images using SSIM and return score and difference map"""
    gray1 = cv2.cvtColor(np.array(img1.convert('RGB')), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2.convert('RGB')), cv2.COLOR_RGB2GRAY)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    (score, diff) = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return score, diff, thresh


# Get mathematical values for animations
math_values = get_math_animation_value()

# ============================================================
#  CLEAN LIGHT AESTHETIC — Soft Aurora / Glass Cards
# ============================================================

# ---- Soft Light Background Scene ----
st.markdown("""
<div class="lt-scene">
  <div class="lt-blob lt-b1"></div>
  <div class="lt-blob lt-b2"></div>
  <div class="lt-blob lt-b3"></div>
  <div class="lt-blob lt-b4"></div>
  <div class="lt-dots"></div>
</div>
""", unsafe_allow_html=True)

# ---- Logo ----
st.markdown("""
<div class="lt-logo-wrap" style="text-align:center; padding-top:18px; padding-bottom:0;">
<svg width="92" height="92" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" class="lt-logo">
<defs>
<linearGradient id="ltGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#6366f1"/>
<stop offset="55%" style="stop-color:#8b5cf6"/>
<stop offset="100%" style="stop-color:#14b8a6"/>
</linearGradient>
</defs>
<polygon points="60,8 104,33 104,87 60,112 16,87 16,33" fill="rgba(99,102,241,0.06)" stroke="url(#ltGrad)" stroke-width="3"/>
<circle cx="60" cy="60" r="17" fill="none" stroke="url(#ltGrad)" stroke-width="3"/>
<circle cx="60" cy="60" r="6" fill="url(#ltGrad)"/>
<line x1="60" y1="29" x2="60" y2="43" stroke="#8b5cf6" stroke-width="2.5" stroke-linecap="round"/>
<line x1="60" y1="77" x2="60" y2="91" stroke="#8b5cf6" stroke-width="2.5" stroke-linecap="round"/>
<line x1="29" y1="60" x2="43" y2="60" stroke="#14b8a6" stroke-width="2.5" stroke-linecap="round"/>
<line x1="77" y1="60" x2="91" y2="60" stroke="#14b8a6" stroke-width="2.5" stroke-linecap="round"/>
<circle cx="60" cy="60" r="27" fill="none" stroke="#6366f1" stroke-width="1" stroke-dasharray="4 6" opacity="0.5"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="16s" repeatCount="indefinite"/></circle>
</svg>
</div>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown("""
<div class="lt-title-wrap" style="text-align:center;">
<div class="lt-title">SmartDetect</div>
<div class="lt-subtitle">AI Anomaly Detection&nbsp;&nbsp;<span class="lt-status">Online</span>&nbsp;&nbsp;v2.0</div>
</div>
""", unsafe_allow_html=True)

# Default theme and model
theme = "Light"
model_choice = "Roboflow Default"

# ---- CLEAN LIGHT THEME STYLES ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

:root {
  --ink:#1f2937; --muted:#5b6472; --soft:#8b93a3;
  --pri:#6366f1; --pri2:#8b5cf6; --teal:#14b8a6;
  --glass:rgba(255,255,255,0.55);
  --glass-strong:rgba(255,255,255,0.72);
  --glass-border:rgba(255,255,255,0.65);
  --line:rgba(148,163,184,0.35);
}

/* ===== KEYFRAMES (fully animated) ===== */
@keyframes lt-kenburns { 0%{transform:scale(1.06) translate(0,0)} 50%{transform:scale(1.14) translate(-1.5%,-1%)} 100%{transform:scale(1.06) translate(0,0)} }
@keyframes lt-veil { 0%,100%{opacity:0.92} 50%{opacity:0.84} }
@keyframes lt-float1 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(36px,-26px) scale(1.08)} }
@keyframes lt-float2 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-42px,28px) scale(1.1)} }
@keyframes lt-float3 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(26px,34px) scale(1.05)} }
@keyframes lt-fadeup { 0%{opacity:0;transform:translateY(16px)} 100%{opacity:1;transform:translateY(0)} }
@keyframes lt-pulse { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(20,184,166,0.5)} 50%{opacity:0.55;box-shadow:0 0 0 6px rgba(20,184,166,0)} }
@keyframes lt-titlein { 0%{opacity:0;letter-spacing:10px} 100%{opacity:1;letter-spacing:2px} }
@keyframes lt-shine { 0%{background-position:0% 50%} 100%{background-position:200% 50%} }

/* ===== HIDE STREAMLIT CHROME ===== */
header[data-testid="stHeader"], div[data-testid="stToolbar"], section[data-testid="stSidebar"], .stDeployButton, #MainMenu, footer { display:none !important; }

/* ===== FONT: JetBrains Mono everywhere (text only, icons preserved) ===== */
html, body, .stApp { font-family:'JetBrains Mono','SFMono-Regular',ui-monospace,monospace !important; color:var(--ink) !important; }
.stApp p, .stApp label, .stApp li, .stMarkdown, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
h1,h2,h3,h4,h5,h6, .stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4,
.stButton > button, .stDownloadButton > button, .stTabs [data-baseweb="tab"],
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div,
.stRadio label, .stCheckbox label, .stSlider label, .stSlider span,
[data-testid="stMetric"], [data-testid="stMetric"] label, [data-testid="stMetricValue"],
.stAlert, .stAlert p, [data-testid="stAlert"] p, [data-testid="stFileUploader"] button {
  font-family:'JetBrains Mono','SFMono-Regular',ui-monospace,monospace !important;
}
/* preserve Streamlit Material icon fonts (prevents icon-ligature text glitches like "uploadUpload") */
.stApp [data-testid="stIconMaterial"], .stApp .material-icons, .stApp [class*="material-symbols"], .stApp [class*="material-icons"], .stApp i {
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons','Material Icons Outlined' !important;
}

/* ===== ROBOT PHOTO BACKGROUND + LIGHT VEIL (animated) ===== */
.stApp { background:transparent !important; }
[data-testid="stAppViewContainer"]::before {
  content:""; position:fixed; inset:0; z-index:-3;
  background:url("data:image/jpeg;base64,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") center center / cover no-repeat;
  animation:lt-kenburns 42s ease-in-out infinite;
}
[data-testid="stAppViewContainer"]::after {
  content:""; position:fixed; inset:0; z-index:-2; pointer-events:none;
  background:linear-gradient(180deg, rgba(248,250,253,0.74) 0%, rgba(247,249,252,0.86) 45%, rgba(246,248,252,0.92) 100%);
  animation:lt-veil 14s ease-in-out infinite;
}

/* soft animated color glows layered over the veil */
.lt-scene { position:fixed; inset:0; z-index:-1; overflow:hidden; pointer-events:none; }
.lt-blob { position:absolute; border-radius:50%; filter:blur(72px); opacity:0.32; mix-blend-mode:screen; }
.lt-b1 { width:480px;height:480px; top:-120px;left:-90px; background:radial-gradient(circle at 30% 30%, #c7d2fe, transparent 70%); animation:lt-float1 24s ease-in-out infinite; }
.lt-b2 { width:440px;height:440px; top:8%;right:-110px; background:radial-gradient(circle at 30% 30%, #ddd6fe, transparent 70%); animation:lt-float2 28s ease-in-out infinite; }
.lt-b3 { width:420px;height:420px; bottom:-130px;left:16%; background:radial-gradient(circle at 30% 30%, #99f6e4, transparent 70%); animation:lt-float3 26s ease-in-out infinite; }
.lt-b4 { width:340px;height:340px; bottom:5%;right:10%; background:radial-gradient(circle at 30% 30%, #fbcfe8, transparent 70%); animation:lt-float1 32s ease-in-out infinite; }
.lt-dots { display:none !important; }

.block-container { position:relative; z-index:1; padding-bottom:3rem; animation:lt-fadeup 0.6s ease both; }

/* ===== TEXT COLORS ===== */
.stApp p, .stApp span, .stApp label, .stApp li, .stMarkdown, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] { color:var(--ink) !important; }
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4 { color:#111827 !important; letter-spacing:0; font-weight:700; }

/* ===== LOGO / TITLE ===== */
.lt-logo { display:inline-block; animation:lt-float3 6s ease-in-out infinite; filter:drop-shadow(0 8px 18px rgba(99,102,241,0.3)); }
.lt-title-wrap { margin:6px 0 16px; }
.lt-title { font-weight:800; font-size:3rem; line-height:1.05; margin-top:6px;
  background:linear-gradient(100deg,#4f46e5,#8b5cf6 40%,#14b8a6 70%,#4f46e5); background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; color:#4f46e5;
  animation:lt-titlein 0.9s ease both, lt-shine 8s linear infinite; }
.lt-subtitle { margin-top:6px; color:#475569; font-size:0.9rem; font-weight:500; letter-spacing:0.4px; }
.lt-status { color:#0d9488; font-weight:700; }
.lt-status::before { content:""; display:inline-block; width:8px;height:8px; border-radius:50%; background:#14b8a6; margin-right:6px; vertical-align:middle; animation:lt-pulse 2s ease-in-out infinite; }

/* ===== TABS (glass / vibrancy) ===== */
.stTabs [data-baseweb="tab-list"] { gap:6px; padding:8px; border-radius:16px; background:var(--glass); backdrop-filter:blur(22px) saturate(180%); -webkit-backdrop-filter:blur(22px) saturate(180%); border:1px solid var(--glass-border); box-shadow:0 8px 30px rgba(15,23,42,0.10); }
.stTabs [data-baseweb="tab"] { height:42px; white-space:nowrap; border-radius:11px; padding:0 16px; background:transparent; color:#475569 !important; border:1px solid transparent; font-size:0.82rem; font-weight:600; transition:all .18s ease; }
.stTabs [data-baseweb="tab"]:hover { color:#4f46e5 !important; background:rgba(238,242,255,0.7); transform:translateY(-1px); }
.stTabs [aria-selected="true"] { color:#fff !important; border:none !important; background:linear-gradient(135deg,#6366f1,#8b5cf6) !important; box-shadow:0 8px 20px rgba(99,102,241,0.4); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { background:transparent !important; }

/* ===== INPUTS (glass) ===== */
.stTextInput > div > div > input, .stSelectbox > div > div, .stTextArea textarea, .stNumberInput input { background:var(--glass-strong) !important; backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); border:1px solid var(--line) !important; border-radius:11px !important; color:var(--ink) !important; }
.stTextInput > div > div > input:focus, .stTextArea textarea:focus { border-color:#6366f1 !important; box-shadow:0 0 0 3px rgba(99,102,241,0.18) !important; }

/* ===== BUTTONS (frosted accent) ===== */
.stButton > button { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important; color:#fff !important; border:none !important; border-radius:11px !important; font-weight:700; letter-spacing:.2px; font-size:0.86rem; white-space:nowrap !important; width:auto !important; min-width:fit-content !important; padding:0.5rem 1.3rem !important; box-shadow:0 6px 18px rgba(99,102,241,0.32); transition:all .18s ease; }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 26px rgba(99,102,241,0.42) !important; filter:brightness(1.05); }
.stButton > button:active { transform:translateY(0); }
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] { background:linear-gradient(135deg,#4f46e5,#7c3aed) !important; }
.stDownloadButton > button { background:var(--glass-strong) !important; backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); color:#4f46e5 !important; border:1px solid rgba(199,210,254,0.9) !important; border-radius:11px !important; font-weight:700; white-space:nowrap !important; width:auto !important; min-width:fit-content !important; padding:0.5rem 1.3rem !important; box-shadow:0 4px 14px rgba(99,102,241,0.16); }
.stDownloadButton > button:hover { background:rgba(238,242,255,0.92) !important; border-color:#6366f1 !important; transform:translateY(-2px); }

/* ===== FILE UPLOADER (glass, NO text/icon glitch) ===== */
[data-testid="stFileUploader"] section { display:flex !important; flex-direction:row !important; flex-wrap:wrap !important; align-items:center !important; gap:14px !important; padding:18px 20px !important; background:var(--glass) !important; backdrop-filter:blur(20px) saturate(180%); -webkit-backdrop-filter:blur(20px) saturate(180%); border:1.5px dashed rgba(99,102,241,0.45) !important; border-radius:16px !important; transition:all .2s ease; }
[data-testid="stFileUploader"] section:hover { border-color:#6366f1 !important; box-shadow:0 8px 24px rgba(99,102,241,0.12); }
[data-testid="stFileUploaderDropzoneInstructions"] { flex:1 1 220px !important; min-width:180px !important; display:flex !important; flex-direction:row !important; align-items:center !important; gap:10px !important; position:static !important; white-space:normal !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div { position:static !important; display:flex !important; flex-direction:column !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small, [data-testid="stFileUploaderDropzoneInstructions"] div { position:static !important; white-space:normal !important; overflow-wrap:anywhere !important; word-break:normal !important; color:var(--muted) !important; letter-spacing:0 !important; text-transform:none !important; }
[data-testid="stFileUploader"] button { flex:0 0 auto !important; white-space:nowrap !important; text-transform:none !important; letter-spacing:.2px !important; width:auto !important; min-width:max-content !important; overflow:visible !important; text-indent:0 !important; padding:0.5rem 1.3rem !important; background:linear-gradient(135deg,#6366f1,#8b5cf6) !important; color:#fff !important; border:none !important; border-radius:10px !important; font-weight:700; box-shadow:0 6px 18px rgba(99,102,241,0.3); }
[data-testid="stFileUploader"] button:hover { filter:brightness(1.05); transform:translateY(-1px); }

/* ===== ALERTS (glass) ===== */
.stAlert, [data-testid="stAlert"] { background:var(--glass-strong) !important; backdrop-filter:blur(16px) saturate(160%); -webkit-backdrop-filter:blur(16px) saturate(160%); border:1px solid var(--line) !important; border-left:4px solid #6366f1 !important; border-radius:13px !important; box-shadow:0 6px 18px rgba(15,23,42,0.06); }
.stAlert p, [data-testid="stAlert"] p, .stAlert span, [data-testid="stAlert"] span, .stAlert div, [data-testid="stAlert"] div { color:var(--ink) !important; }
div[data-testid="stAlert"][data-type="success"] { border-left-color:#10b981 !important; }
div[data-testid="stAlert"][data-type="error"] { border-left-color:#ef4444 !important; }
div[data-testid="stAlert"][data-type="warning"] { border-left-color:#f59e0b !important; }

/* ===== METRICS (glass) ===== */
[data-testid="stMetric"] { background:var(--glass) !important; backdrop-filter:blur(22px) saturate(180%); -webkit-backdrop-filter:blur(22px) saturate(180%); border:1px solid var(--glass-border); border-radius:16px; padding:16px; box-shadow:0 8px 24px rgba(15,23,42,0.08); }
[data-testid="stMetric"] label { color:#5b6472 !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#4f46e5 !important; font-size:1.9rem !important; font-weight:800; }

/* ===== SLIDERS / RADIO / CHECKBOX ===== */
.stSlider label, .stSlider span { color:var(--ink) !important; }
.stRadio label, .stCheckbox label { color:var(--ink) !important; }

/* ===== DATAFRAME (glass) ===== */
[data-testid="stDataFrame"], .stDataFrame { background:var(--glass-strong) !important; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border:1px solid var(--line) !important; border-radius:13px !important; overflow:hidden; box-shadow:0 6px 18px rgba(15,23,42,0.06); }
[data-testid="stDataFrame"] th { background:rgba(238,242,255,0.9) !important; color:#4338ca !important; }

/* ===== PROGRESS / SPINNER ===== */
.stProgress > div > div > div { background:linear-gradient(90deg,#6366f1,#8b5cf6) !important; }
.stProgress > div > div { background:rgba(226,232,240,0.7) !important; }
.stSpinner > div { border-top-color:#6366f1 !important; }

/* ===== IMAGES / EXPANDER / VIDEO ===== */
[data-testid="stImage"] img { border:1px solid var(--glass-border); border-radius:13px; box-shadow:0 8px 22px rgba(15,23,42,0.12); }
.streamlit-expanderHeader, [data-testid="stExpander"] summary { color:var(--ink) !important; background:var(--glass) !important; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border:1px solid var(--glass-border) !important; border-radius:12px !important; }
[data-testid="stExpander"] { border:none !important; background:transparent !important; }
video, [data-testid="stCameraInput"] > div { border:1px solid var(--glass-border) !important; border-radius:12px !important; }

/* ===== LINKS / DIVIDER ===== */
a, a:visited { color:#4f46e5 !important; text-decoration:none; border-bottom:1px dotted #818cf8; }
a:hover { color:#7c3aed !important; }
hr, .stMarkdown hr { border:none; height:1px; background:linear-gradient(90deg,transparent,#c7d2fe,#a78bfa,transparent); }

/* ===== TOOLTIP FIX (kills dark/black tooltip glitch) ===== */
[data-baseweb="tooltip"], [role="tooltip"], [data-testid="stTooltipContent"], div[data-baseweb="popover"] [data-testid="stTooltipContent"] {
  background:rgba(255,255,255,0.94) !important; backdrop-filter:blur(14px) saturate(180%); -webkit-backdrop-filter:blur(14px) saturate(180%);
  color:#1f2937 !important; border:1px solid rgba(148,163,184,0.4) !important; border-radius:10px !important; box-shadow:0 8px 24px rgba(15,23,42,0.18) !important;
}
[data-baseweb="tooltip"] *, [role="tooltip"] *, [data-testid="stTooltipContent"] * { color:#1f2937 !important; background:transparent !important; }
[data-testid="stTooltipIcon"] svg { fill:#6366f1 !important; }

/* ===== SCROLLBAR / SELECTION / COLUMNS ===== */
::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-track { background:rgba(226,232,240,0.4); }
::-webkit-scrollbar-thumb { background:linear-gradient(#a5b4fc,#c4b5fd); border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:linear-gradient(#818cf8,#a78bfa); }
::selection { background:rgba(139,92,246,0.25); color:#1f2937; }
[data-testid="column"] { padding:0 8px; }

</style>
""", unsafe_allow_html=True)

# ---- Main Application Logic ----
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📤 Upload & Preview", "🔍 Detection & AI Correction",
    "📹 Snapshot Video Detection", "🌍 Earth Pro Analysis",
    "📊 Feedback & Report", "🧭 Tutorial", "ℹ️ About/Docs",
])

# ---------- Tab 1: Upload & Preview ----------
with tab1:
    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.markdown("### 🖼️ Image Gallery")
        num_images = len(uploaded_files)
        cols_per_row = min(num_images, 4)
        cols = st.columns(cols_per_row)
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % cols_per_row]:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption=uploaded_file.name, width=200)
                uploaded_file.seek(0)

# ---------- Tab 2: Detection & Correction ----------
with tab2:
    uploaded_files = st.session_state.uploaded_files
    if not uploaded_files:
        st.warning("Upload images in the first tab.")
    else:
        st.markdown("### ⚙️ Detection Settings")

        detection_mode = st.radio(
            "Detection Mode",
            ["🛡️ Object Detection (YOLOv8)", "🛣️ Road/Surface Cracks (OpenCV)", "🎨 Stain/Discoloration (OpenCV)"],
            help="Choose your detection target: Objects, Cracks, or Stains.",
        )

        threshold = st.slider("Minimum Confidence (%)", 0, 100, 50)

        st.markdown("### 🤖 AI Correction Settings")
        use_ai_correction = st.checkbox("Enable AI-Powered Correction", value=True, help="Use AI to intelligently remove anomalies")

        if use_ai_correction:
            st.info("🎨 AI will intelligently remove detected anomalies and generate clean, natural-looking corrections.")
        else:
            st.warning("⚠️ AI correction disabled. Only detection will be performed.")

        st.write(f"**Detection:** Anomalies with confidence ≥ {threshold}% will be shown.")

        color_picker_high = "#00ff00"
        color_picker_mid = "#ff0000"
        color_picker_low = "#ffff00"

        if "session_results" not in st.session_state:
            st.session_state.session_results = []

        def get_color(conf):
            if conf >= 0.9:
                return color_picker_high
            elif conf >= 0.7:
                return color_picker_mid
            else:
                return color_picker_low

        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        session_results = []

        with zipfile.ZipFile(temp_zip.name, "w") as zip_all:
            for idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"---\n#### Image {idx + 1}: {uploaded_file.name}")
                image_bytes = uploaded_file.getvalue()
                orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                preds = []

                with st.spinner(f"Detecting anomalies ({detection_mode}) for {uploaded_file.name}..."):
                    try:
                        if "Road/Surface Cracks" in detection_mode:
                            preds = detect_cracks_opencv(orig_img)
                            st.success(f"✅ Crack Detection complete! Found {len(preds)} defects.")

                        elif "Stain/Discoloration" in detection_mode:
                            preds = detect_stains_opencv(orig_img)
                            st.success(f"✅ Stain Detection complete! Found {len(preds)} defects.")

                        else:
                            model = load_yolo_model()
                            results = model(orig_img)

                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x, y, w, h = box.xywh[0].tolist()
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = model.names[cls]

                                    if conf * 100 >= threshold:
                                        preds.append({
                                            "x": x,
                                            "y": y,
                                            "width": w,
                                            "height": h,
                                            "confidence": conf,
                                            "class": label,
                                        })

                            st.success(f"✅ AI Detection complete! Found {len(preds)} objects/anomalies.")

                    except Exception as e:
                        st.error(f"❌ Detection Error: {str(e)}")
                        continue

                if preds:
                    df = pd.DataFrame(preds)[["x", "y", "width", "height", "confidence"]]
                    df["confidence (%)"] = (df["confidence"] * 100).round(2)
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download anomaly results as CSV",
                        data=csv,
                        file_name=f"anomaly_results_{idx+1}.csv",
                        mime="text/csv",
                        key=f"csv_download_{idx}",
                    )

                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine="openpyxl")
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Download anomaly results as Excel",
                        data=excel_buffer,
                        file_name=f"anomaly_results_{idx+1}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_download_{idx}",
                    )

                    zip_all.writestr(f"anomaly_results_{idx+1}.csv", csv)
                    zip_all.writestr(f"anomaly_results_{idx+1}.xlsx", excel_buffer.getvalue())

                im_anno = orig_img.copy()
                draw = ImageDraw.Draw(im_anno)
                for pred in preds:
                    x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                    y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                    x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                    y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                    color = get_color(pred["confidence"])
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

                im_corr = orig_img.copy()

                if use_ai_correction and preds:
                    with st.spinner("🤖 AI is generating corrected image..."):
                        try:
                            for pred in preds:
                                x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                box = (x0, y0, x1, y1)
                                region = im_corr.crop(box).filter(ImageFilter.GaussianBlur(20))
                                im_corr.paste(region, box)

                            st.success("✅ AI correction completed!")
                        except Exception as e:
                            st.error(f"AI correction failed: {str(e)}")
                            im_corr = orig_img.copy()

                st.markdown("""
<div style="display: flex; justify-content: center; gap: 25px; margin: 15px 0; padding: 12px 20px; background: #eef2ff; border-radius: 4px;">
<div style="display: flex; align-items: center; gap: 8px;">
<div style="width: 20px; height: 20px; background-color: #00ff00; border-radius: 50%; box-shadow: 0 2px 6px rgba(0,255,0,0.4);"></div>
<span><b>High Confidence</b> ≥90%</span>
</div>
<div style="display: flex; align-items: center; gap: 8px;">
<div style="width: 20px; height: 20px; background-color: #ffff00; border-radius: 50%; box-shadow: 0 2px 6px rgba(255,255,0,0.4);"></div>
<span><b>Medium Confidence</b> 70-89%</span>
</div>
<div style="display: flex; align-items: center; gap: 8px;">
<div style="width: 20px; height: 20px; background-color: #ff0000; border-radius: 50%; box-shadow: 0 2px 6px rgba(255,0,0,0.4);"></div>
<span><b>Low Confidence</b> &lt;70%</span>
</div>
</div>
""", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(orig_img, caption="Original", use_container_width=True, output_format="PNG")
                with col2:
                    st.image(im_anno, caption="Detected", use_container_width=True, output_format="PNG")
                with col3:
                    st.image(im_corr, caption="Corrected", use_container_width=True, output_format="PNG")

                img_anno_b = io.BytesIO()
                im_anno.save(img_anno_b, format="PNG")
                img_anno_b.seek(0)
                st.download_button(
                    label="Download Annotated",
                    data=img_anno_b,
                    file_name=f"annotated_{idx+1}.png",
                    mime="image/png",
                    key=f"anno_download_{idx}",
                )

                img_corr_b = io.BytesIO()
                im_corr.save(img_corr_b, format="PNG")
                img_corr_b.seek(0)
                st.download_button(
                    label="Download Corrected",
                    data=img_corr_b,
                    file_name=f"corrected_{idx+1}.png",
                    mime="image/png",
                    key=f"corr_download_{idx}",
                )

                zip_all.writestr(f"annotated_{idx+1}.png", img_anno_b.getvalue())
                zip_all.writestr(f"corrected_{idx+1}.png", img_corr_b.getvalue())

                session_results.append({
                    "filename": uploaded_file.name,
                    "num_anomalies": len(preds),
                    "ai_corrected": use_ai_correction,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "correction_file": f"corrected_{idx+1}.png",
                })

        st.session_state.session_results = session_results

        with open(temp_zip.name, "rb") as zf:
            all_zip_bytes = zf.read()

        try:
            os.unlink(temp_zip.name)
        except Exception:
            pass

        st.download_button(
            label="Download All Results/Images as ZIP",
            data=all_zip_bytes,
            file_name="SmartDetect_results.zip",
            mime="application/zip",
            key="zip_download_all",
        )

        st.write("---")
        st.markdown("## Session Results")
        if session_results:
            st.dataframe(pd.DataFrame(session_results))

# ---------- Tab 3: SnapShot Video Detection ----------
with tab3:
    st.markdown("### 📹 Snapshot Video Anomaly Detection")
    st.info("🎥 Detect anomalies in real-time from your webcam or video feed.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 🎬 Video Source")
        video_source = st.radio(
            "Select video source:",
            ["Webcam", "Upload Video File"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("#### ⚙️ Detection Settings")
        video_threshold = st.slider("Confidence Threshold (%)", 0, 100, 60, key="video_threshold")
        show_boxes = st.checkbox("Show Detection Boxes", value=True, key="show_boxes")

    st.markdown("---")

    if video_source == "Webcam":
        st.markdown("#### 📷 Webcam Feed")
        st.warning("⚠️ **Note:** Webcam access requires browser permissions.")

        st.markdown("""
**How to use:**
1. Click "Enable Webcam" below
2. Point camera at surfaces to check
3. Click "Capture & Detect"
4. View detected anomalies
""")

        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False

        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button("🎥 Enable Webcam", key="start_webcam", use_container_width=True):
                st.session_state.webcam_running = True

        with col_stop:
            if st.button("⏹️ Disable Webcam", key="stop_webcam", use_container_width=True):
                st.session_state.webcam_running = False

        video_detection_mode = st.radio(
            "Video Detection Mode",
            ["🛡️ Object Detection (YOLOv8)", "🛣️ Road/Surface Cracks (OpenCV)", "🎨 Stain/Discoloration (OpenCV)"],
            horizontal=True,
            key="video_mode_select",
        )

        st.markdown("---")

        if st.session_state.webcam_running:
            st.markdown("**Step 1 — Capture:** Click **Take Photo** in the camera below.")
            camera_image = st.camera_input("Live Camera Feed", key="camera_feed")

            st.markdown("**Step 2 — Detect:** Then click the button below to analyze the captured frame.")
            capture_btn = st.button(
                "📸 Capture & Detect",
                key="snapshot",
                use_container_width=True,
                type="primary",
                disabled=camera_image is None,
            )

            if camera_image is None:
                st.info("📷 Take a photo first using the camera above — then 'Capture & Detect' will light up.")
            else:
                img_bytes = camera_image.getvalue()
                # Run detection automatically when a NEW frame is captured, or when
                # "Capture & Detect" is pressed. Results are cached so they persist
                # across reruns (e.g. clicking the download button).
                need_detect = (st.session_state.get("webcam_last_frame") != img_bytes) or capture_btn

                if need_detect:
                    st.session_state.webcam_last_frame = img_bytes
                    with st.spinner("🔍 Detecting anomalies..."):
                        try:
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                            preds = []

                            if "Road/Surface Cracks" in video_detection_mode:
                                preds = detect_cracks_opencv(img)
                                msg = f"✅ Found {len(preds)} defects"
                            elif "Stain/Discoloration" in video_detection_mode:
                                preds = detect_stains_opencv(img)
                                msg = f"✅ Found {len(preds)} stains"
                            else:
                                model = load_yolo_model()
                                results = model(img)

                                for result in results:
                                    boxes = result.boxes
                                    for box in boxes:
                                        x, y, w, h = box.xywh[0].tolist()
                                        conf = float(box.conf[0])
                                        cls = int(box.cls[0])
                                        label = model.names[cls]

                                        if conf * 100 >= video_threshold:
                                            preds.append({
                                                "x": x,
                                                "y": y,
                                                "width": w,
                                                "height": h,
                                                "confidence": conf,
                                                "class": label,
                                            })

                                msg = f"✅ Found {len(preds)} anomalies"

                            img_annotated = img.copy()
                            if show_boxes:
                                draw = ImageDraw.Draw(img_annotated)
                                for pred in preds:
                                    x0 = int(float(pred["x"]) - float(pred["width"]) / 2)
                                    y0 = int(float(pred["y"]) - float(pred["height"]) / 2)
                                    x1 = int(float(pred["x"]) + float(pred["width"]) / 2)
                                    y1 = int(float(pred["y"]) + float(pred["height"]) / 2)
                                    draw.rectangle([x0, y0, x1, y1], outline="#FF0000", width=3)

                            buf = io.BytesIO()
                            img_annotated.save(buf, format="PNG")

                            st.session_state.webcam_result = {
                                "orig": img_bytes,
                                "annotated": buf.getvalue(),
                                "msg": msg,
                            }
                        except Exception as e:
                            st.session_state.webcam_result = None
                            st.error(f"❌ Error: {str(e)}")

                result = st.session_state.get("webcam_result")
                if result:
                    st.success(result["msg"])
                    col_orig, col_detect = st.columns(2)
                    with col_orig:
                        st.markdown("**Original Frame**")
                        st.image(result["orig"], use_container_width=True)
                    with col_detect:
                        st.markdown("**Detected Anomalies**")
                        st.image(result["annotated"], use_container_width=True)

                    st.download_button(
                        label="📥 Download Annotated Frame",
                        data=result["annotated"],
                        file_name="webcam_detection.png",
                        mime="image/png",
                        key="webcam_dl",
                    )
        else:
            st.session_state.pop("webcam_result", None)
            st.session_state.pop("webcam_last_frame", None)
            st.info("👆 Click 'Enable Webcam' to start")
    else:
        st.markdown("#### 📁 Upload Video File")
        uploaded_video = st.file_uploader("Upload video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"], key="video_upload")
        if uploaded_video:
            st.video(uploaded_video)
            st.info("Video processing feature - Install opencv-python and ffmpeg to enable")

# ---------- Tab 4: IMPROVED EARTH PRO ANALYSIS ----------
with tab4:
    st.markdown("### 🌍 Google Earth Pro Image Comparison - ENHANCED")

    st.info("""
📸 **How to use this feature:**

1. Open **Google Earth Pro** on your computer
2. Navigate to the area you want to analyze
3. Take a **screenshot** of the area from an earlier year (use the time slider)
4. Take another **screenshot** of the **same exact area** from a recent year
5. Upload both screenshots below

**🎯 Enhanced Detection:** This improved version uses 6 different detection algorithms to catch ALL building changes - from tiny shops to large commercial buildings!
""")

    st.markdown("---")

    col_upload1, col_upload2 = st.columns(2)

    with col_upload1:
        st.markdown("### 📤 Upload Earlier Year Image")
        st.caption("Upload a Google Earth Pro screenshot from an earlier year")
        earlier_image = st.file_uploader(
            "Choose earlier year image",
            type=["jpg", "jpeg", "png"],
            key="earth_earlier",
            help="Screenshot from Google Earth Pro - earlier year",
        )
        if earlier_image:
            img_earlier = Image.open(earlier_image).convert("RGB")
            st.image(img_earlier, caption="Earlier Year (Baseline)", use_container_width=True)

    with col_upload2:
        st.markdown("### 📤 Upload Recent Year Image")
        st.caption("Upload a Google Earth Pro screenshot from a recent year")
        recent_image = st.file_uploader(
            "Choose recent year image",
            type=["jpg", "jpeg", "png"],
            key="earth_recent",
            help="Screenshot from Google Earth Pro - recent year",
        )
        if recent_image:
            img_recent = Image.open(recent_image).convert("RGB")
            st.image(img_recent, caption="Recent Year (Current)", use_container_width=True)

    if earlier_image and recent_image:
        st.markdown("---")
        st.markdown("#### ⚙️ Enhanced Analysis Settings")

        col_settings1, col_settings2, col_settings3 = st.columns(3)

        with col_settings1:
            min_change_area = st.slider(
                "Minimum Building Size (pixels²)",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="Lower = Detects smaller buildings. Recommended: 30-80",
            )

        with col_settings2:
            yolo_confidence = st.slider(
                "AI Confidence (%)",
                min_value=10,
                max_value=70,
                value=15,
                help="Lower = More detections. Recommended: 15-25%",
            )

        with col_settings3:
            use_yolo = st.checkbox(
                "Enable YOLO AI",
                value=True,
                help="Use deep learning for building detection",
            )

        st.markdown("---")

        if st.button("🔍 Analyze Changes (Enhanced)", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("📸 Loading images...")
                img_old = Image.open(earlier_image).convert("RGB")
                img_new = Image.open(recent_image).convert("RGB")
                progress_bar.progress(10)

                if img_old.size != img_new.size:
                    status_text.text("📐 Resizing images to match...")
                    img_new = img_new.resize(img_old.size)
                progress_bar.progress(15)

                status_text.text("📊 Calculating image similarity...")
                ssim_score, diff_map, thresh_map = compare_images_ssim(img_old, img_new)
                progress_bar.progress(25)

                status_text.text("🔬 Running comprehensive computer vision analysis (6 methods)...")
                opencv_changes = detect_changes_comprehensive(img_old, img_new, min_area=min_change_area)
                progress_bar.progress(60)

                st.info(f"✅ Computer Vision detected {len(opencv_changes)} changes")

                yolo_changes = []
                if use_yolo:
                    status_text.text("🤖 Running deep learning (YOLO) analysis...")
                    model = load_yolo_model()
                    yolo_changes = detect_changes_yolo(img_old, img_new, model, min_confidence=yolo_confidence / 100)
                    st.info(f"✅ YOLO AI detected {len(yolo_changes)} new objects")

                progress_bar.progress(80)

                all_changes = opencv_changes + yolo_changes

                status_text.text("🏷️ Classifying detected changes...")
                for change in all_changes:
                    if 'type' not in change or not change.get('type'):
                        change['type'] = classify_change_type(change, "Earlier", "Recent")
                progress_bar.progress(90)

                status_text.text("🎨 Creating annotated visualization...")
                img_annotated = img_new.copy()
                draw = ImageDraw.Draw(img_annotated)

                try:
                    font = ImageFont.truetype("arial.ttf", 13)
                    font_small = ImageFont.truetype("arial.ttf", 11)
                except Exception:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()

                def get_box_color(confidence):
                    if confidence >= 0.8:
                        return "#00ff00"  # Green
                    elif confidence >= 0.5:
                        return "#ffff00"  # Yellow
                    else:
                        return "#ff6600"  # Orange

                for idx, change in enumerate(all_changes):
                    x = int(change['x'])
                    y = int(change['y'])
                    w = int(change['width'])
                    h = int(change['height'])
                    conf = change.get('confidence', 0.5)
                    change_type = change.get('type', 'Change')

                    x0 = int(x - w / 2)
                    y0 = int(y - h / 2)
                    x1 = int(x + w / 2)
                    y1 = int(y + h / 2)

                    color = get_box_color(conf)

                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

                    label_text = f"{change_type}"
                    conf_text = f"{conf*100:.0f}%"

                    try:
                        bbox = draw.textbbox((x0, y0 - 28), label_text, font=font)
                        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 0, 0))
                        draw.text((x0, y0 - 28), label_text, fill=color, font=font)
                        draw.text((x0, y0 - 14), conf_text, fill="white", font=font_small)
                    except Exception:
                        draw.text((x0, y0 - 20), f"{label_text} {conf_text}", fill=color)

                progress_bar.progress(95)

                st.session_state.earth_results = {
                    'img_old': img_old,
                    'img_new': img_new,
                    'img_annotated': img_annotated,
                    'all_changes': all_changes,
                    'ssim_score': ssim_score,
                    'opencv_count': len(opencv_changes),
                    'yolo_count': len(yolo_changes),
                }

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")

                st.balloons()
                st.success(f"🎉 Analysis complete! Found **{len(all_changes)}** total changes (OpenCV: {len(opencv_changes)}, YOLO: {len(yolo_changes)})")

                progress_bar.empty()
                status_text.empty()

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                progress_bar.empty()
                status_text.empty()

    if "earth_results" in st.session_state:
        results = st.session_state.earth_results

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric("🔍 Total Changes", len(results['all_changes']))
        with col_m2:
            st.metric("🖥️ OpenCV Detections", results['opencv_count'])
        with col_m3:
            st.metric("🤖 YOLO Detections", results['yolo_count'])
        with col_m4:
            st.metric(
                "📏 Similarity",
                f"{results['ssim_score']:.1%}",
                delta=f"{(1-results['ssim_score'])*100:.1f}% changed",
                delta_color="inverse",
            )

        st.markdown("---")

        st.markdown("""
<div style="display: flex; justify-content: center; gap: 25px; margin: 15px 0; padding: 12px 20px; background: #eef2ff; border-radius: 4px;">
<div style="display: flex; align-items: center; gap: 8px;">
<div style="width: 20px; height: 20px; background-color: #00ff00; border-radius: 50%;"></div>
<span><b>High</b> ≥80%</span>
</div>
<div style="display: flex; align-items: center; gap: 8px;">
<div style="width: 20px; height: 20px; background-color: #ffff00; border-radius: 50%;"></div>
<span><b>Medium</b> 50-79%</span>
</div>
<div style="display: flex; align-items: center; gap: 8px;">
<div style="width: 20px; height: 20px; background-color: #ff6600; border-radius: 50%;"></div>
<span><b>Low</b> &lt;50%</span>
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("### 🖼️ Image Comparison")

        col_img1, col_img2, col_img3 = st.columns(3)

        with col_img1:
            st.markdown("**Earlier Year**")
            st.image(results['img_old'], use_container_width=True)

        with col_img2:
            st.markdown("**Recent Year**")
            st.image(results['img_new'], use_container_width=True)

        with col_img3:
            st.markdown("**🎯 All Detected Changes**")
            st.image(results['img_annotated'], use_container_width=True)

        buf_annotated = io.BytesIO()
        results['img_annotated'].save(buf_annotated, format='PNG')
        buf_annotated.seek(0)
        st.download_button(
            "📥 Download Annotated Image",
            buf_annotated,
            "earth_pro_enhanced_analysis.png",
            "image/png",
            use_container_width=True,
        )

        st.markdown("---")

        if results['all_changes']:
            st.markdown("### 📋 Detailed Change Analysis")

            changes_data = []
            for idx, change in enumerate(results['all_changes'], 1):
                changes_data.append({
                    "ID": idx,
                    "Type": change.get('type', 'Change Detected'),
                    "Confidence": f"{change.get('confidence', 0.5)*100:.1f}%",
                    "Method": change.get('method', 'N/A'),
                    "X": int(change['x']),
                    "Y": int(change['y']),
                    "Width": int(change['width']),
                    "Height": int(change['height']),
                    "Area (px²)": int(change.get('area', change['width'] * change['height'])),
                })

            df_changes = pd.DataFrame(changes_data)
            st.dataframe(df_changes, use_container_width=True)

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                csv_data = df_changes.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download as CSV",
                    csv_data,
                    "earth_pro_enhanced_changes.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with col_dl2:
                excel_buffer = io.BytesIO()
                df_changes.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button(
                    "📥 Download as Excel",
                    excel_buffer,
                    "earth_pro_enhanced_changes.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        else:
            st.info("No significant changes detected between the two images.")

# ---------- Tab 5: Feedback & Report ----------
with tab5:
    st.markdown("### 📝 Leave Feedback & Generate PDF Report")
    if "feedback_list" not in st.session_state:
        st.session_state.feedback_list = []

    feedback = st.text_area("Type feedback or bug report:", key="feedback_input")
    if st.button("Submit Feedback", key="submit_feedback_btn"):
        if feedback:
            st.session_state.feedback_list.append(feedback)
            st.success("Thank you for your feedback!")

    st.write("---")
    st.markdown("#### 💬 All Feedback")
    if st.session_state.feedback_list:
        for i, fb in enumerate(st.session_state.feedback_list, 1):
            st.markdown(f"**{i}:** {fb}")
    else:
        st.info("No feedback yet.")

    st.write("---")
    st.markdown("#### 📄 Generate PDF Summary Report")

    if st.button("🔄 Generate & Auto-Download PDF Report", key="generate_pdf_btn"):
        pdf_file = "SmartDetect_Session_Report.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="SmartDetect AI Image Anomaly Detection Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(10)

        session_results = st.session_state.get("session_results", [])

        if session_results:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="Detection Results:", ln=True)
            pdf.set_font("Arial", size=11)
            for sess in session_results:
                pdf.multi_cell(0, 8, txt=f"  Image: {sess['filename']}\n  Anomalies: {sess['num_anomalies']}\n  AI Correction: {'Enabled' if sess.get('ai_corrected', False) else 'Disabled'}\n  Date: {sess['date']}\n")
                pdf.ln(5)
        else:
            pdf.cell(200, 10, txt="No detection results yet.", ln=True)

        pdf.ln(10)
        if st.session_state.feedback_list:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="User Feedback:", ln=True)
            pdf.set_font("Arial", size=11)
            for fb in st.session_state.feedback_list:
                pdf.multi_cell(0, 8, txt=f"  - {fb}")
                pdf.ln(3)

        pdf.output(pdf_file)

        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode()

        auto_download_js = f'<script>var link = document.createElement("a");link.href = "data:application/pdf;base64,{b64_pdf}";link.download = "{pdf_file}";link.click();</script>'
        components.html(auto_download_js, height=0)
        st.success(f"✅ PDF Report generated: {pdf_file}")

        st.download_button("📥 Download PDF (Manual)", pdf_bytes, pdf_file, "application/pdf", key="pdf_manual_download")

# ---------- Tab 6: Tutorial ----------
with tab6:
    st.markdown("""
## How to Use This App (Tutorial)

**Step 1: Upload & Preview**
Upload your images (JPG, PNG) to begin the analysis.

**Step 2: Detection & AI Correction**
Choose your detection mode and let AI find and correct anomalies.

**Step 3: Snapshot Video Detection**
Use your webcam for real-time anomaly detection.

**Step 4: 🌍 Earth Pro Analysis (ENHANCED!)**
- Open Google Earth Pro and navigate to your area of interest
- Use the time slider to select an earlier year
- Take a screenshot
- Select a recent year and take another screenshot
- Upload both images here
- **Enhanced detection uses 6 different computer vision methods**
- Detects buildings of ALL sizes - from small shops to large complexes
- Optionally enable YOLO AI for even better results

**Step 5: Generate Reports**
Create PDF reports and provide feedback on your experience.

### 🎯 Tips for Best Results:
- Use high-resolution Google Earth Pro screenshots
- Ensure both images are from the exact same viewpoint
- Lower the "Minimum Building Size" slider to detect smaller buildings
- Enable YOLO AI for comprehensive building detection
- Experiment with different confidence thresholds
""")

# ---------- Tab 7: About/Docs ----------
with tab7:
    st.markdown("""
<div style="text-align: center; max-width: 800px; margin: 0 auto; font-family: 'JetBrains Mono', monospace;">

<h2 style="font-weight: 800; color: #4f46e5; text-shadow: none; font-family: 'JetBrains Mono', monospace; font-size: 2rem; letter-spacing: 2px;">
&gt; ABOUT SmartDetect_</h2>

<p style="font-size: 1rem; line-height: 1.8; color: #475569; border-left: 2px solid #c7d2fe; padding-left: 15px; text-align: left;">
SmartDetect is a cutting-edge AI solution for quality control, infrastructure maintenance, and urban development monitoring.
Powered by YOLOv8 deep learning and 6 OpenCV computer vision methods.
</p>

<div style="background: #f8fafc; padding: 20px; border: 1px solid #e0e7ff; margin: 20px 0; text-align: left;">
<h3 style="color: #4f46e5; text-shadow: none; font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;">
[SYS] Enhanced Earth Pro Analysis</h3>
<p style="color: #475569; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; line-height: 2;">
<span style="color:#4f46e5;">[1]</span> Multi-Scale Intensity Analysis<br>
<span style="color:#4f46e5;">[2]</span> Edge Structure Detection<br>
<span style="color:#4f46e5;">[3]</span> RGB Color Change Detection<br>
<span style="color:#4f46e5;">[4]</span> Gradient/Texture Analysis<br>
<span style="color:#4f46e5;">[5]</span> Adaptive Thresholding<br>
<span style="color:#4f46e5;">[6]</span> Laplacian Detail Detection<br><br>
<span style="color:#4f46e5;">+</span> Optional YOLO deep learning for building detection
</p>
</div>

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 16px; margin: 30px 0;">
<div style="background: #f8fafc; padding: 20px; border: 1px solid #e0e7ff; width: 200px;">
<div style="font-size: 1.6rem; color: #4f46e5; text-shadow: none;">[ AI ]</div>
<h4 style="color: #4f46e5; font-family: 'JetBrains Mono', monospace;">AI Correction</h4>
<p style="font-size: 0.85rem; color: #475569;">Intelligent anomaly removal</p>
</div>
<div style="background: #f8fafc; padding: 20px; border: 1px solid #e0e7ff; width: 200px;">
<div style="font-size: 1.6rem; color: #4f46e5; text-shadow: none;">[ LIVE ]</div>
<h4 style="color: #4f46e5; font-family: 'JetBrains Mono', monospace;">Live Detection</h4>
<p style="font-size: 0.85rem; color: #475569;">Real-time analysis</p>
</div>
<div style="background: #f8fafc; padding: 20px; border: 1px solid #e0e7ff; width: 200px;">
<div style="font-size: 1.6rem; color: #4f46e5; text-shadow: none;">[ SAT ]</div>
<h4 style="color: #4f46e5; font-family: 'JetBrains Mono', monospace;">Earth Pro</h4>
<p style="font-size: 0.85rem; color: #475569;">6-method satellite analysis</p>
</div>
</div>

<div style="border-top: 1px solid #e0e7ff; padding-top: 20px; margin-top: 10px;">
<h3 style="color: #4f46e5; font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;">[ CREDITS ]</h3>
<p style="color: #475569; font-size: 0.9rem; line-height: 2;">
<span style="color:#4f46e5;">dev://</span> Sugnik Tarafder<br>
<span style="color:#4f46e5;">dev://</span> Arifur Rahaman<br>
<span style="color:#4f46e5;">dev://</span> Sk Shonju Ali<br>
<span style="color:#4f46e5;">dev://</span> Trishan Nayek
</p>
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin-top: 20px; font-size: 0.85rem; color: #6366f1; font-family: JetBrains Mono, monospace; letter-spacing: 2px;'>SmartDetect v2.0 • AI Anomaly Detection • 6-Method Detection Engine</div>", unsafe_allow_html=True)

# scripts/ocr_utils.py

import cv2
import pytesseract
from PIL import Image
import numpy as np
import re

# Ensure tesseract is installed in your system and path is set
# Example (Windows): pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image for OCR (grayscale, noise removal, thresholding)."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.medianBlur(gray, 3)
    
    # Thresholding for better OCR
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text(image_path: str) -> str:
    """Extract raw text from image using Tesseract OCR."""
    processed_img = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_img)
    return text

def parse_medical_values(text: str) -> dict:
    """Parse medical values like blood pressure, cholesterol from OCR text."""
    results = {}
    
    bp_pattern = re.search(r"(BP|Blood Pressure)[:\- ]?(\d{2,3}/\d{2,3})", text, re.IGNORECASE)
    chol_pattern = re.search(r"(Cholesterol|Chol)[:\- ]?(\d{2,3})", text, re.IGNORECASE)
    
    if bp_pattern:
        results["blood_pressure"] = bp_pattern.group(2)
    if chol_pattern:
        results["cholesterol"] = int(chol_pattern.group(2))
    
    return results

def ocr_pipeline(image_path: str) -> dict:
    """Complete OCR pipeline with preprocessing, extraction, and parsing."""
    text = extract_text(image_path)
    values = parse_medical_values(text)
    return {"raw_text": text, "parsed_values": values}

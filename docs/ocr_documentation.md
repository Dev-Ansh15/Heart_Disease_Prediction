# OCR System Documentation

## Overview
Day 4 focused on developing an OCR pipeline to extract medical data (blood pressure, cholesterol, etc.) from scanned medical documents.

## Components
- **Preprocessing**: Grayscale conversion, noise reduction, thresholding
- **OCR Engine**: Tesseract OCR (`pytesseract`)
- **Parsing**: Regex-based extraction of medical values
- **Utilities**: Error handling, modular functions in `scripts/ocr_utils.py`

## Pipeline
1. Load image (JPG/PNG/PDF → image)
2. Preprocess (noise reduction + binarization)
3. Extract text (Tesseract)
4. Parse structured values (Regex)
5. Return structured dictionary

## Sample Output
```json
{
  "raw_text": "Patient: John Doe\nBlood Pressure: 120/80\nCholesterol: 200",
  "parsed_values": {"blood_pressure": "120/80", "cholesterol": 200}
}

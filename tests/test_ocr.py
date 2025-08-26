# tests/test_ocr.py

import sys 
import os 
sys.path.append(os.path.dirname(os.getcwd()))
from scripts import ocr_utils

def test_preprocess_image():
    sample_img = "data/raw/other_sources/sample_medical_report.png"
    processed = ocr_utils.preprocess_image(sample_img)
    assert processed is not None

def test_ocr_pipeline():
    sample_img = "data/raw/other_sources/sample_medical_report.png"
    results = ocr_utils.ocr_pipeline(sample_img)
    assert "raw_text" in results
    assert isinstance(results["parsed_values"], dict)

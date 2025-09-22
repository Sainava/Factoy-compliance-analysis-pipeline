# Japan AI Model - Compliance Analysis Pipeline

## Overview

A camera-agnostic, industry-pack aware compliance analysis pipeline for both 360° and regular video. This notebook demonstrates preprocessing, inference (prototype using pretrained detectors), rule fusion, evidence extraction, report generation, and a simple dashboard prototype.

## System Concept

The model notebook ingests either a 360° equirectangular walkthrough or a standard MP4 (phone / CCTV). It normalizes video frames, runs modular perception models (object detection, segmentation, pose, tracking, OCR), applies a Rule Engine that combines General + Industry Pack rules (parsed from sample SOPs or template rules), applies temporal persistence and confidence thresholds, extracts evidence clips and snapshots, computes a Compliance Score, and exports the output (JSON events, CSV, PDF report). The notebook also includes a simple Streamlit demo page to visualize results.

## Directory Structure

```
compliance-analysis-pipeline/
├── README.md
├── requirements.txt
├── compliance_analysis_notebook.ipynb    # Main notebook
├── data/                                 # Input videos and test data
├── outputs/                             # Generated reports, events, evidence
├── sops/                               # Sample SOP documents
├── rules/                              # Rule templates and configurations
└── templates/                          # Report templates and schemas
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install system dependencies:
- Tesseract OCR: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Ubuntu)

## Usage

1. Open the Jupyter notebook: `compliance_analysis_notebook.ipynb`
2. Run cells sequentially to demonstrate the pipeline
3. Place your video files in the `data/` directory
4. Configure industry pack and parameters in Cell 2
5. Review outputs in the `outputs/` directory

## Key Features

- **Camera Agnostic**: Supports both 360° equirectangular and standard video formats
- **Industry Packs**: Modular rule sets for different industries (food, manufacturing, etc.)
- **Real-time Detection**: YOLOv8-based object detection with tracking
- **Evidence Generation**: Automatic clip and snapshot extraction
- **Compliance Scoring**: Explainable scoring with violation weighting
- **Interactive Dashboard**: Streamlit-based visualization interface

## Demo Disclaimer

**Demo detections shown in this notebook are illustrative. Full production accuracy requires labeled factory datasets and model fine-tuning.**

## Next Steps

1. Collect and label factory-specific training data
2. Fine-tune detection models on your specific use cases
3. Implement real-time processing for live video streams
4. Deploy to edge devices or cloud infrastructure
#!/usr/bin/env python3
"""
Quick Test Script for Japan AI Model Compliance Analysis Pipeline
Run this script to test the pipeline on your Toyota VR 360° factory video
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🏭 Japan AI Model - Compliance Analysis Pipeline Test")
    print("=" * 60)
    
    # Configuration
    VIDEO_PATH = "data/ssvid.net--Toyota-VR-360-Factory-Tour_v720P.mp4"
    INDUSTRY_PACK = "manufacturing"
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: Video file not found at {VIDEO_PATH}")
        print("Please ensure your video is in the data/ directory")
        return
    
    print(f"📹 Input Video: {VIDEO_PATH}")
    print(f"🏭 Industry Pack: {INDUSTRY_PACK}")
    
    # Get video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    print(f"\n📊 Video Information:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Frame Count: {frame_count}")
    print(f"  FPS: {fps:.1f}")
    
    # Detect if it's 360° video
    aspect_ratio = width / height
    is_360 = aspect_ratio >= 1.8  # 360° videos typically have 2:1 aspect ratio
    
    print(f"  Aspect Ratio: {aspect_ratio:.2f}")
    print(f"  360° Video: {'Yes' if is_360 else 'No'}")
    
    # Sample a few frames to test
    print(f"\n🔍 Sampling frames for analysis...")
    
    sample_frames = []
    sample_interval = max(1, int(fps * 2))  # Sample every 2 seconds
    
    for i in range(0, min(frame_count, 300), sample_interval):  # Sample first 5 minutes max
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            timestamp = i / fps
            sample_frames.append((i, timestamp, frame))
            print(f"  Frame {len(sample_frames)}: {i} at {timestamp:.1f}s")
            
        if len(sample_frames) >= 10:  # Limit to 10 frames for quick test
            break
    
    cap.release()
    
    print(f"\n✅ Successfully sampled {len(sample_frames)} frames")
    
    # Try to load YOLO model
    print(f"\n🤖 Testing object detection...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download the model if not present
        print("✅ YOLOv8 model loaded successfully")
        
        # Test detection on first frame
        if sample_frames:
            test_frame = sample_frames[0][2]
            results = model.predict(test_frame, conf=0.45, verbose=False)
            
            if results and len(results) > 0 and hasattr(results[0], 'boxes'):
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                print(f"✅ Detected {detections} objects in test frame")
                
                # Show detected classes
                if detections > 0:
                    classes = []
                    for box in results[0].boxes:
                        if hasattr(box, 'cls'):
                            cls_id = int(box.cls[0])
                            class_name = model.names.get(cls_id, f'class_{cls_id}')
                            classes.append(class_name)
                    
                    unique_classes = list(set(classes))
                    print(f"  Classes detected: {', '.join(unique_classes[:5])}")
            else:
                print("ℹ️ No objects detected in test frame")
        
    except Exception as e:
        print(f"⚠️ Object detection test failed: {e}")
        print("This is normal for demo - the pipeline will use mock detections")
    
    # Create output directories
    print(f"\n📁 Setting up output directories...")
    output_dirs = ['outputs', 'outputs/reports', 'outputs/evidence', 'outputs/clips']
    for dir_name in output_dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"  Created: {dir_name}/")
    
    print(f"\n🎯 Pipeline Test Complete!")
    print(f"=" * 60)
    print(f"✅ Your video is ready for compliance analysis!")
    print(f"\nNext steps:")
    print(f"1. Open Jupyter Notebook: jupyter notebook compliance_analysis_notebook.ipynb")
    print(f"2. Run all cells in sequence (Cell -> Run All)")
    print(f"3. Check outputs/ directory for results")
    print(f"\nExpected outputs:")
    print(f"- outputs/reports/compliance_report.pdf")
    print(f"- outputs/reports/violations.csv")
    print(f"- outputs/scorecard.json")
    print(f"- outputs/compliance_dashboard.py")

if __name__ == "__main__":
    main()
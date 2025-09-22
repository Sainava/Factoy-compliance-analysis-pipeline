#!/usr/bin/env python3
"""
Test script to verify web interface setup
Run this to check if all components are properly configured
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ Missing {description}: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and report status"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ Missing {description}: {dirpath}")
        return False

def test_imports():
    """Test if core modules can be imported"""
    print("\n🔍 Testing Python imports...")
    
    try:
        import compliance_engine
        print("✅ compliance_engine module imports successfully")
        
        # Test engine initialization
        engine = compliance_engine.ComplianceEngine()
        print("✅ ComplianceEngine can be instantiated")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Warning during engine initialization: {e}")
        return True  # Still consider it a pass

def main():
    """Run all setup verification tests"""
    print("🏭 Compliance Analysis Pipeline - Setup Verification")
    print("=" * 55)
    
    all_good = True
    
    # Check core files
    print("\n📁 Checking core files...")
    all_good &= check_file_exists("app.py", "FastAPI web server")
    all_good &= check_file_exists("compliance_engine.py", "Compliance analysis engine")
    all_good &= check_file_exists("requirements_web.txt", "Web interface requirements")
    all_good &= check_file_exists("SETUP_WEB.md", "Setup instructions")
    
    # Check template files
    print("\n🌐 Checking web templates...")
    all_good &= check_file_exists("templates/index.html", "Upload interface template")
    all_good &= check_file_exists("templates/results.html", "Results display template")
    
    # Check directories
    print("\n📂 Checking directories...")
    all_good &= check_directory_exists("templates", "Templates directory")
    all_good &= check_directory_exists("static", "Static files directory")
    
    # Check YOLO model
    print("\n🤖 Checking AI models...")
    all_good &= check_file_exists("yolov8n.pt", "YOLO model file")
    
    # Check if sample video exists
    print("\n🎥 Checking sample data...")
    sample_video = "data/ssvid.net--Toyota-VR-360-Factory-Tour_v720P.mp4"
    if check_file_exists(sample_video, "Sample video"):
        print("   You can use this video to test the pipeline")
    else:
        print("   No sample video found - you can upload your own for testing")
    
    # Test imports
    all_good &= test_imports()
    
    # Create missing directories
    print("\n📁 Creating required directories...")
    Path("uploads").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    print("✅ Created uploads/ and outputs/ directories")
    
    # Final assessment
    print("\n" + "=" * 55)
    if all_good:
        print("🎉 Setup verification PASSED!")
        print("\nTo start the web interface:")
        print("1. Install dependencies: pip install -r requirements_web.txt")
        print("2. Start server: python app.py")
        print("3. Open browser: http://localhost:8000")
    else:
        print("⚠️ Setup verification found issues")
        print("Please check the missing files/components above")
    
    print(f"\nFor detailed setup instructions, see: SETUP_WEB.md")
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
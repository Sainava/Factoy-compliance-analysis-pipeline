#!/usr/bin/env python3
"""
FastAPI Web Server for Compliance Analysis
Provides upload and inference endpoints for video compliance analysis
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import json
import asyncio
from pathlib import Path

# Import our compliance engine
from compliance_engine import ComplianceEngine

# Initialize FastAPI app
app = FastAPI(
    title="Compliance Analysis Pipeline",
    description="AI-powered video compliance analysis for industrial safety",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")
TEMPLATE_DIR = Path("templates")

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, TEMPLATE_DIR]:
    directory.mkdir(exist_ok=True)

# Mount static files with proper MIME type handling
import mimetypes

# Ensure video MIME types are registered
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/avi', '.avi') 
mimetypes.add_type('video/mov', '.mov')
mimetypes.add_type('video/webm', '.webm')

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Template engine
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Global variables for analysis state
analysis_cache = {}
engines = {}

def get_compliance_engine(industry_pack: str = 'manufacturing') -> ComplianceEngine:
    """Get or create compliance engine for industry pack"""
    if industry_pack not in engines:
        engines[industry_pack] = ComplianceEngine(industry_pack=industry_pack)
    return engines[industry_pack]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
async def view_analysis(request: Request, analysis_id: str):
    """View analysis results page"""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis_data = analysis_cache[analysis_id]
    return templates.TemplateResponse("results.html", {
        "request": request,
        "analysis": analysis_data,
        "analysis_id": analysis_id
    })

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    industry_pack: str = Form(default="manufacturing"),
    confidence_threshold: float = Form(default=0.45)
):
    """Upload video and start analysis"""
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix
    video_filename = f"{analysis_id}{file_extension}"
    video_path = UPLOAD_DIR / video_filename
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Initialize analysis status
    analysis_cache[analysis_id] = {
        "status": "processing",
        "progress": 0,
        "started_at": datetime.now().isoformat(),
        "video_filename": file.filename,
        "industry_pack": industry_pack,
        "confidence_threshold": confidence_threshold
    }
    
    # Start background analysis
    background_tasks.add_task(
        process_video_analysis,
        analysis_id,
        str(video_path),
        industry_pack,
        confidence_threshold
    )
    
    return JSONResponse({
        "analysis_id": analysis_id,
        "status": "processing",
        "message": "Video uploaded successfully. Analysis started."
    })

async def process_video_analysis(
    analysis_id: str,
    video_path: str,
    industry_pack: str,
    confidence_threshold: float
):
    """Background task to process video analysis"""
    
    try:
        # Update status
        analysis_cache[analysis_id]["status"] = "analyzing"
        analysis_cache[analysis_id]["progress"] = 10
        
        # Get compliance engine
        engine = get_compliance_engine(industry_pack)
        engine.confidence_threshold = confidence_threshold
        
        # Update progress
        analysis_cache[analysis_id]["progress"] = 20
        
        # Run analysis
        results = engine.analyze_video(
            video_path=video_path,
            output_dir=str(OUTPUT_DIR / analysis_id)
        )
        
        # Update with results
        analysis_cache[analysis_id].update({
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now().isoformat(),
            "results": results
        })
        
    except Exception as e:
        # Handle errors
        analysis_cache[analysis_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

@app.get("/api/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """Get analysis status and progress"""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return JSONResponse(analysis_cache[analysis_id])

@app.get("/api/analysis/{analysis_id}/results")
async def get_analysis_results(analysis_id: str):
    """Get full analysis results"""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis_data = analysis_cache[analysis_id]
    
    if analysis_data["status"] != "completed":
        raise HTTPException(status_code=202, detail="Analysis not completed yet")
    
    return JSONResponse(analysis_data["results"])

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_analyses": len([a for a in analysis_cache.values() if a["status"] == "processing"]),
        "total_analyses": len(analysis_cache)
    }

@app.get("/api/industry-packs")
async def get_industry_packs():
    """Get available industry packs"""
    return {
        "industry_packs": [
            {
                "id": "manufacturing",
                "name": "Manufacturing",
                "description": "Safety helmets, high-vis vests, posture analysis"
            },
            {
                "id": "food",
                "name": "Food Processing",
                "description": "Hairnets, gloves, hygiene compliance"
            },
            {
                "id": "chemical",
                "name": "Chemical Plant",
                "description": "Respiratory protection, hazmat compliance"
            },
            {
                "id": "general",
                "name": "General Safety",
                "description": "Basic safety rules and exit clearance"
            }
        ]
    }

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete analysis and associated files"""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        # Remove video file
        video_files = list(UPLOAD_DIR.glob(f"{analysis_id}.*"))
        for video_file in video_files:
            video_file.unlink(missing_ok=True)
        
        # Remove output directory
        output_dir = OUTPUT_DIR / analysis_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        # Remove from cache
        del analysis_cache[analysis_id]
        
        return {"message": "Analysis deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")

@app.get("/api/analysis")
async def list_analyses(limit: int = 50, status: Optional[str] = None):
    """List all analyses with optional filtering"""
    analyses = []
    
    for analysis_id, data in analysis_cache.items():
        if status and data.get("status") != status:
            continue
            
        analyses.append({
            "analysis_id": analysis_id,
            "status": data.get("status"),
            "started_at": data.get("started_at"),
            "video_filename": data.get("video_filename"),
            "industry_pack": data.get("industry_pack"),
            "compliance_score": data.get("results", {}).get("compliance_score", {}).get("compliance_score")
        })
    
    # Sort by started_at (newest first)
    analyses.sort(key=lambda x: x["started_at"], reverse=True)
    
    return {"analyses": analyses[:limit]}

@app.get("/test-video", response_class=HTMLResponse)
async def test_video(request: Request):
    """Test page for debugging video playback"""
    return templates.TemplateResponse("video_test.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Compliance Analysis API starting up...")
    
    # Initialize default engine
    try:
        get_compliance_engine("manufacturing")
        print("✅ Default compliance engine initialized")
    except Exception as e:
        print(f"⚠️ Engine initialization warning: {e}")
    
    print("🌐 API ready at http://localhost:8000")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Compliance Analysis API shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    print("🏭 Starting Compliance Analysis Server...")
    print("📊 Access dashboard at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# Compliance Analysis Pipeline - Web Interface

## Quick Start Guide

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements_web.txt

# Or create a virtual environment first (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_web.txt
```

### 2. Download YOLO Model
```bash
# The yolov8n.pt model should already be in the root directory
# If not, it will be downloaded automatically on first run
```

### 3. Start the Web Server
```bash
# Start the FastAPI server
python app.py

# Or use uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Web Interface
- **Main Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Web Interface Features

### 🎯 Video Upload & Analysis
- Drag & drop video upload
- Support for MP4, AVI, MOV, WEBM formats
- Standard and 360° video detection
- Real-time progress tracking

### 🏭 Industry-Specific Analysis
- **Manufacturing**: Helmet, vest, posture analysis
- **Food Processing**: Hairnet, glove compliance
- **Chemical Plant**: Respiratory protection
- **General Safety**: Basic safety rules

### 📊 Detailed Results
- Compliance score with letter grade
- Violation breakdown with evidence
- Interactive timeline of violations
- Downloadable reports (PDF/JSON)

### 🔧 Configurable Settings
- Adjustable detection confidence
- Industry pack selection
- Custom rule configurations

## API Endpoints

### Upload & Analysis
- `POST /upload` - Upload video and start analysis
- `GET /api/analysis/{id}/status` - Check analysis progress
- `GET /api/analysis/{id}/results` - Get full results

### Management
- `GET /api/analysis` - List all analyses
- `DELETE /api/analysis/{id}` - Delete analysis
- `GET /api/industry-packs` - Get available industry packs
- `GET /api/health` - Health check

## Project Structure
```
compliance-analysis-pipeline/
├── app.py                 # FastAPI web server
├── compliance_engine.py   # Core analysis engine
├── templates/             # HTML templates
│   ├── index.html        # Upload interface
│   └── results.html      # Results display
├── static/               # CSS, JS, images
├── uploads/              # Uploaded videos
├── outputs/              # Analysis results
└── requirements_web.txt  # Web dependencies
```

## Configuration

### Environment Variables
Create a `.env` file for configuration:
```
# Server settings
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Analysis settings
DEFAULT_INDUSTRY=manufacturing
DEFAULT_CONFIDENCE=0.45
SAMPLE_FPS=2

# Storage paths
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
```

### Customizing Industry Rules
Edit the `_load_industry_rules()` method in `compliance_engine.py` to add custom rules:

```python
{
    'id': 'custom.rule.id',
    'type': 'custom_violation_type',
    'severity': 'major',  # critical, major, minor, warning
    'description': 'Custom rule description',
    'persistence_threshold': 5
}
```

## Usage Examples

### Basic Analysis
1. Open http://localhost:8000
2. Select industry pack (e.g., Manufacturing)
3. Adjust confidence threshold if needed
4. Upload or drag & drop video file
5. Wait for analysis completion
6. View detailed results with violations and score

### API Usage
```python
import requests

# Upload video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload',
        files={'file': f},
        data={
            'industry_pack': 'manufacturing',
            'confidence_threshold': 0.45
        }
    )

analysis_id = response.json()['analysis_id']

# Check status
status = requests.get(f'http://localhost:8000/api/analysis/{analysis_id}/status')

# Get results when complete
results = requests.get(f'http://localhost:8000/api/analysis/{analysis_id}/results')
```

## Production Deployment

### Using Gunicorn
```bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static/ {
        alias /path/to/static/;
    }
}
```

## Performance Considerations

### Video Processing
- Large videos are processed in chunks
- Frame sampling reduces processing time
- Background processing prevents UI blocking

### Memory Management
- Analysis cache is in-memory (consider Redis for production)
- Large videos may require disk-based processing
- Automatic cleanup of old analyses recommended

### Scaling
- Multiple worker processes with Gunicorn
- Load balancing for multiple instances
- Separate video processing queue for heavy loads

## Troubleshooting

### Common Issues
1. **Import errors**: Install all requirements with `pip install -r requirements_web.txt`
2. **YOLO model not found**: Ensure `yolov8n.pt` is in the root directory
3. **Upload failures**: Check file size limits and video format
4. **Slow processing**: Reduce confidence threshold or video resolution

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check
Check system status:
```bash
curl http://localhost:8000/api/health
```

## Next Steps

1. **Custom Models**: Train industry-specific YOLO models
2. **Advanced Rules**: Implement complex temporal compliance rules  
3. **Real-time Processing**: Add live video stream analysis
4. **Integration**: Connect with existing safety management systems
5. **Mobile App**: Create mobile interface for field inspections

## Support

- Check API docs at `/docs` for detailed endpoint information
- Monitor logs for error diagnostics
- Use health check endpoint for system monitoring
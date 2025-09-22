#!/usr/bin/env python3
"""
Compliance Analysis Engine
Extracted core functions from the Jupyter notebook for web interface deployment
"""

import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

class ComplianceEngine:
    """Main compliance analysis engine"""
    
    def __init__(self, industry_pack='manufacturing', confidence_threshold=0.45):
        self.industry_pack = industry_pack
        self.confidence_threshold = confidence_threshold
        self.sample_fps = 2
        self.img_size = 640
        
        # Initialize models
        self.yolo_model = None
        self.pose_estimator = None
        self.tracker = None
        
        # Industry-specific configurations
        self.industry_rules = self._load_industry_rules()
        self.severity_weights = {
            'critical': 30,
            'major': 20,
            'minor': 10,
            'warning': 3
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models"""
        print("🤖 Initializing AI models...")
        
        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("✅ YOLOv8 model loaded")
            except Exception as e:
                print(f"⚠️ YOLO initialization failed: {e}")
        
        # Initialize MediaPipe Pose
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_pose = mp.solutions.pose
                self.pose_estimator = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5
                )
                print("✅ MediaPipe Pose initialized")
            except Exception as e:
                print(f"⚠️ MediaPipe initialization failed: {e}")
        
        # Initialize Tracker
        if DEEPSORT_AVAILABLE:
            try:
                self.tracker = DeepSort(max_age=30, n_init=3)
                print("✅ DeepSort tracker initialized")
            except Exception as e:
                print(f"⚠️ Tracker initialization failed: {e}")
                self.tracker = MockTracker()
        else:
            self.tracker = MockTracker()
            print("📝 Using mock tracker")
    
    def _load_industry_rules(self):
        """Load industry-specific compliance rules"""
        rules = {
            'manufacturing': [
                {
                    'id': 'PPE.helmet.required',
                    'type': 'missing_helmet',
                    'severity': 'major',
                    'description': 'Workers must wear safety helmets',
                    'persistence_threshold': 5
                },
                {
                    'id': 'PPE.vest.required', 
                    'type': 'missing_vest',
                    'severity': 'major',
                    'description': 'Workers must wear high-visibility vests',
                    'persistence_threshold': 5
                },
                {
                    'id': 'ergonomics.posture',
                    'type': 'poor_posture',
                    'severity': 'minor',
                    'description': 'Workers should maintain proper posture',
                    'persistence_threshold': 10
                }
            ],
            'food': [
                {
                    'id': 'hygiene.hairnet.required',
                    'type': 'missing_hairnet',
                    'severity': 'major',
                    'description': 'Food handlers must wear hairnets',
                    'persistence_threshold': 3
                },
                {
                    'id': 'hygiene.gloves.required',
                    'type': 'missing_gloves', 
                    'severity': 'major',
                    'description': 'Food handlers must wear gloves',
                    'persistence_threshold': 3
                }
            ],
            'chemical': [
                {
                    'id': 'PPE.respirator.required',
                    'type': 'missing_respirator',
                    'severity': 'critical',
                    'description': 'Workers must wear respiratory protection',
                    'persistence_threshold': 2
                }
            ],
            'general': [
                {
                    'id': 'safety.exit.clear',
                    'type': 'blocked_exit',
                    'severity': 'critical',
                    'description': 'Emergency exits must remain clear',
                    'persistence_threshold': 1
                }
            ]
        }
        return rules.get(self.industry_pack, rules['general'])
    
    def detect_video_type(self, video_path: str) -> Dict:
        """Detect if video is standard or 360°"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        # Detect 360° video by aspect ratio
        aspect_ratio = width / height
        is_360 = aspect_ratio >= 1.8  # 360° videos typically have 2:1 ratio
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': frame_count / fps if fps > 0 else 0,
            'aspect_ratio': aspect_ratio,
            'is_360': is_360,
            'video_type': '360°' if is_360 else 'Standard'
        }
    
    def preprocess_video(self, video_path: str) -> List[Tuple]:
        """Extract and preprocess frames from video"""
        print("🔧 Preprocessing video...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate sampling interval
        sample_interval = max(1, int(fps / self.sample_fps))
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps
                
                # Resize frame for processing
                processed_frame = cv2.resize(frame, (self.img_size, self.img_size))
                
                frames.append((frame_idx, timestamp, processed_frame, frame))  # processed + original
            
            frame_idx += 1
        
        cap.release()
        print(f"✅ Extracted {len(frames)} frames for analysis")
        return frames
    
    def run_object_detection(self, frames: List[Tuple]) -> List[Dict]:
        """Run YOLO object detection on frames"""
        print("🎯 Running object detection...")
        
        if not self.yolo_model:
            return self._generate_mock_detections(frames)
        
        all_detections = []
        
        for frame_idx, timestamp, processed_frame, original_frame in frames:
            try:
                results = self.yolo_model.predict(
                    processed_frame, 
                    conf=self.confidence_threshold,
                    verbose=False
                )
                
                frame_detections = []
                if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    for box in results[0].boxes:
                        detection = {
                            'frame_idx': frame_idx,
                            'timestamp': timestamp,
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'confidence': float(box.conf[0]),
                            'class_id': int(box.cls[0]),
                            'class_name': self.yolo_model.names[int(box.cls[0])]
                        }
                        frame_detections.append(detection)
                
                all_detections.extend(frame_detections)
                
            except Exception as e:
                print(f"⚠️ Detection failed for frame {frame_idx}: {e}")
                continue
        
        print(f"✅ Detected {len(all_detections)} objects")
        return all_detections
    
    def _generate_mock_detections(self, frames: List[Tuple]) -> List[Dict]:
        """Generate mock detections for demo"""
        print("📝 Generating mock detections...")
        
        mock_detections = []
        np.random.seed(42)  # Reproducible results
        
        for frame_idx, timestamp, processed_frame, original_frame in frames:
            # Generate 1-3 random detections per frame
            num_detections = np.random.randint(1, 4)
            
            for i in range(num_detections):
                # Random person detection
                x1 = np.random.randint(50, 400)
                y1 = np.random.randint(50, 400) 
                x2 = x1 + np.random.randint(80, 150)
                y2 = y1 + np.random.randint(100, 200)
                
                detection = {
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': np.random.uniform(0.5, 0.9),
                    'class_id': 0,  # person
                    'class_name': 'person'
                }
                mock_detections.append(detection)
                
                # Sometimes add helmet/vest
                if np.random.random() < 0.3:  # 30% chance missing PPE
                    ppe_detection = detection.copy()
                    ppe_detection['class_name'] = np.random.choice(['helmet', 'vest'])
                    ppe_detection['class_id'] = 1 if ppe_detection['class_name'] == 'helmet' else 2
                    mock_detections.append(ppe_detection)
        
        return mock_detections
    
    def run_tracking(self, detections: List[Dict]) -> List[Dict]:
        """Apply tracking to detections"""
        print("🏃 Running object tracking...")
        
        # Group detections by frame
        frame_groups = {}
        for det in detections:
            frame_idx = det['frame_idx']
            if frame_idx not in frame_groups:
                frame_groups[frame_idx] = []
            frame_groups[frame_idx].append(det)
        
        tracked_detections = []
        
        # Process frames in order
        for frame_idx in sorted(frame_groups.keys()):
            frame_detections = frame_groups[frame_idx]
            
            if hasattr(self.tracker, 'update_tracks'):
                # Real DeepSort tracking
                detection_list = []
                for det in frame_detections:
                    x1, y1, x2, y2 = det['bbox']
                    detection_list.append([x1, y1, x2, y2, det['confidence']])
                
                if detection_list:
                    # Check if this is the real DeepSort tracker or mock
                    if hasattr(self.tracker, '__class__') and 'DeepSort' in str(self.tracker.__class__):
                        tracks = self.tracker.update_tracks(detection_list, frame=None)
                        
                        for i, track in enumerate(tracks):
                            if track.is_confirmed() and i < len(frame_detections):
                                tracked_det = frame_detections[i].copy()
                                tracked_det['track_id'] = track.track_id
                                tracked_detections.append(tracked_det)
                    else:
                        # Mock tracker - call without frame parameter
                        tracked_objects = self.tracker.update_tracks(frame_detections)
                        tracked_detections.extend(tracked_objects)
        
        print(f"✅ Generated {len(tracked_detections)} tracked detections")
        return tracked_detections
    
    def analyze_pose(self, frames: List[Tuple], detections: List[Dict]) -> List[Dict]:
        """Analyze worker posture"""
        print("🤸 Analyzing posture...")
        
        pose_results = []
        
        if not self.pose_estimator:
            # Generate mock pose data
            for det in detections:
                if det['class_name'] == 'person':
                    pose_result = {
                        'timestamp': det['timestamp'],
                        'track_id': det.get('track_id', 0),
                        'poor_posture': np.random.random() < 0.2,  # 20% chance
                        'risk_score': np.random.uniform(0.1, 0.8),
                        'posture_type': np.random.choice(['normal', 'bending', 'reaching'])
                    }
                    pose_results.append(pose_result)
            return pose_results
        
        # Real pose analysis would go here
        # For now, return mock data
        return pose_results
    
    def evaluate_compliance_rules(self, detections: List[Dict], pose_data: List[Dict]) -> List[Dict]:
        """Evaluate compliance violations based on rules"""
        print("⚖️ Evaluating compliance rules...")
        
        violations = []
        
        # Group detections by timestamp for rule evaluation
        timestamp_groups = {}
        for det in detections:
            ts = det['timestamp']
            if ts not in timestamp_groups:
                timestamp_groups[ts] = []
            timestamp_groups[ts].append(det)
        
        # Evaluate each rule
        for rule in self.industry_rules:
            violations.extend(
                self._evaluate_single_rule(rule, timestamp_groups, pose_data)
            )
        
        print(f"✅ Found {len(violations)} compliance violations")
        return violations
    
    def _evaluate_single_rule(self, rule: Dict, timestamp_groups: Dict, pose_data: List[Dict]) -> List[Dict]:
        """Evaluate a single compliance rule"""
        violations = []
        
        for timestamp, detections in timestamp_groups.items():
            violation_detected = False
            violating_detection = None
            
            if 'helmet' in rule['type']:
                # Check for missing helmet
                persons = [d for d in detections if d['class_name'] == 'person']
                helmets = [d for d in detections if d['class_name'] == 'helmet']
                
                if persons and len(helmets) < len(persons):
                    violation_detected = True
                    violating_detection = persons[0]
            
            elif 'vest' in rule['type']:
                # Check for missing vest
                persons = [d for d in detections if d['class_name'] == 'person']
                vests = [d for d in detections if d['class_name'] == 'vest']
                
                if persons and len(vests) < len(persons):
                    violation_detected = True
                    violating_detection = persons[0]
            
            elif 'posture' in rule['type']:
                # Check pose data
                poor_postures = [p for p in pose_data if p['timestamp'] == timestamp and p['poor_posture']]
                if poor_postures:
                    violation_detected = True
                    violating_detection = {
                        'timestamp': timestamp,
                        'bbox': [320, 240, 420, 340],  # Center of frame
                        'track_id': poor_postures[0].get('track_id', 0)
                    }
            
            if violation_detected and violating_detection:
                violation = {
                    'id': f"V-{len(violations)+1:03d}",
                    'rule_id': rule['id'],
                    'type': rule['type'],
                    'severity': rule['severity'],
                    'description': rule['description'],
                    'timestamp': timestamp,
                    'bbox': violating_detection['bbox'],
                    'track_id': violating_detection.get('track_id', 0),
                    'confidence': violating_detection.get('confidence', 0.8),
                    'duration': rule['persistence_threshold']  # Simplified for demo
                }
                violations.append(violation)
        
        return violations
    
    def calculate_compliance_score(self, violations: List[Dict], video_duration: float) -> Dict:
        """Calculate overall compliance score"""
        print("📊 Calculating compliance score...")
        
        base_score = 100
        total_penalty = 0
        
        violation_counts = {'critical': 0, 'major': 0, 'minor': 0, 'warning': 0}
        
        for violation in violations:
            severity = violation['severity']
            violation_counts[severity] += 1
            
            # Calculate penalty
            weight = self.severity_weights[severity]
            duration_factor = violation.get('duration', 1) / 10
            confidence_factor = violation['confidence']
            
            penalty = weight * duration_factor * confidence_factor
            total_penalty += penalty
        
        final_score = max(0, base_score - total_penalty)
        
        # Assign letter grade
        if final_score >= 90:
            grade = 'A'
        elif final_score >= 80:
            grade = 'B'
        elif final_score >= 70:
            grade = 'C'
        elif final_score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'compliance_score': round(final_score, 1),
            'grade': grade,
            'total_violations': len(violations),
            'violation_counts': violation_counts,
            'penalty_breakdown': {
                'total_penalty': round(total_penalty, 1),
                'base_score': base_score
            }
        }
    
    def generate_evidence(self, violations: List[Dict], video_path: str, output_dir: str) -> List[Dict]:
        """Generate evidence clips and thumbnails"""
        print("📎 Generating evidence...")
        
        os.makedirs(output_dir, exist_ok=True)
        evidence_dir = os.path.join(output_dir, 'evidence')
        os.makedirs(evidence_dir, exist_ok=True)
        
        enhanced_violations = []
        
        # Open video for evidence extraction
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("⚠️ Could not open video for evidence generation")
            return violations
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for violation in violations:
            timestamp = violation['timestamp']
            violation_id = violation['id']
            
            try:
                # Calculate frame number from timestamp
                frame_number = int(timestamp * fps)
                frame_number = min(frame_number, total_frames - 1)  # Ensure within bounds
                
                # Set video position to the violation timestamp
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Generate thumbnail
                    thumbnail_filename = f"thumb_{violation_id}.jpg"
                    thumbnail_path = os.path.join(evidence_dir, thumbnail_filename)
                    
                    # Create a copy of the frame for processing
                    display_frame = frame.copy()
                    
                    # Draw bounding box on thumbnail if available
                    if 'bbox' in violation:
                        try:
                            x1, y1, x2, y2 = [int(coord) for coord in violation['bbox']]
                            # Scale bbox to original frame size if needed
                            h, w = frame.shape[:2]
                            if w != self.img_size or h != self.img_size:
                                # Assume bbox was from resized frame, scale back
                                scale_x = w / self.img_size
                                scale_y = h / self.img_size
                                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                            
                            # Ensure coordinates are within frame bounds
                            x1 = max(0, min(x1, w-1))
                            y1 = max(0, min(y1, h-1))
                            x2 = max(x1+1, min(x2, w))
                            y2 = max(y1+1, min(y2, h))
                            
                            # Draw red rectangle around violation
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            
                            # Add violation text
                            violation_text = violation.get('type', 'violation').replace('_', ' ').title()
                            cv2.putText(display_frame, violation_text, (x1, max(y1-10, 20)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        except Exception as bbox_error:
                            print(f"⚠️ Error drawing bbox for {violation_id}: {bbox_error}")
                    
                    # Save thumbnail
                    success = cv2.imwrite(thumbnail_path, display_frame)
                    if not success:
                        print(f"⚠️ Failed to save thumbnail for {violation_id}")
                        thumbnail_url = None
                    else:
                        # Get the relative path from the analysis output directory
                        analysis_id = os.path.basename(output_dir)
                        thumbnail_url = f"/outputs/{analysis_id}/evidence/{thumbnail_filename}"
                    
                    # Generate video clip (5 seconds around violation)
                    clip_duration = 5.0  # seconds
                    start_time_clip = max(0, timestamp - clip_duration/2)
                    end_time_clip = timestamp + clip_duration/2
                    
                    start_frame = int(start_time_clip * fps)
                    end_frame = min(int(end_time_clip * fps), total_frames - 1)
                    
                    clip_filename = f"clip_{violation_id}.mp4"
                    clip_path = os.path.join(evidence_dir, clip_filename)
                    
                    # Create video clip
                    clip_success = self._create_video_clip(video_path, clip_path, start_frame, end_frame, fps)
                    
                    if clip_success and os.path.exists(clip_path):
                        clip_url = f"/outputs/{analysis_id}/evidence/{clip_filename}"
                    else:
                        print(f"⚠️ Failed to create clip for {violation_id}")
                        clip_url = None
                    
                    evidence_paths = {
                        'thumbnail': thumbnail_url,
                        'clip': clip_url
                    }
                    
                    print(f"✅ Generated evidence for {violation_id}: thumbnail={thumbnail_url is not None}, clip={clip_url is not None}")
                    
                else:
                    print(f"⚠️ Could not extract frame at timestamp {timestamp}s for {violation_id}")
                    evidence_paths = {
                        'thumbnail': None,
                        'clip': None
                    }
            
            except Exception as e:
                print(f"⚠️ Evidence generation failed for {violation_id}: {e}")
                evidence_paths = {
                    'thumbnail': None,
                    'clip': None
                }
            
            enhanced_violation = violation.copy()
            enhanced_violation['evidence'] = evidence_paths
            enhanced_violations.append(enhanced_violation)
        
        cap.release()
        print(f"✅ Evidence generation complete for {len(enhanced_violations)} violations")
        return enhanced_violations
    
    def _create_video_clip(self, input_path: str, output_path: str, start_frame: int, end_frame: int, fps: float):
        """Create a web-compatible video clip from start_frame to end_frame"""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Use H.264 codec for web compatibility
            # Try different codec options in order of preference
            codec_options = [
                cv2.VideoWriter_fourcc(*'H264'),  # H.264 codec
                cv2.VideoWriter_fourcc(*'avc1'),  # Another H.264 variant
                cv2.VideoWriter_fourcc(*'mp4v'),  # Fallback
                cv2.VideoWriter_fourcc(*'XVID')   # Last resort
            ]
            
            out = None
            for fourcc in codec_options:
                try:
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    # Test if the writer is working
                    if out.isOpened():
                        break
                    else:
                        out.release()
                        out = None
                except:
                    if out:
                        out.release()
                    out = None
                    continue
            
            if out is None:
                print(f"❌ Could not initialize video writer for {output_path}")
                cap.release()
                return False
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = start_frame
            frames_written = 0
            while frame_count <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_written += 1
                frame_count += 1
            
            cap.release()
            out.release()
            
            # Verify the file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ Created video clip: {output_path} ({frames_written} frames)")
                return True
            else:
                print(f"❌ Video clip creation failed: {output_path}")
                return False
            
        except Exception as e:
            print(f"❌ Error creating video clip: {e}")
            return False
            
        except Exception as e:
            print(f"⚠️ Clip creation failed: {e}")
            return False
    
    def analyze_video(self, video_path: str, output_dir: str = 'outputs') -> Dict:
        """Main analysis pipeline"""
        print(f"🏭 Starting compliance analysis for {self.industry_pack} industry...")
        
        start_time = datetime.now()
        
        try:
            # 1. Detect video type
            video_metadata = self.detect_video_type(video_path)
            print(f"📹 Video Type: {video_metadata['video_type']} ({video_metadata['width']}x{video_metadata['height']})")
            
            # 2. Preprocess video
            frames = self.preprocess_video(video_path)
            
            # 3. Object detection
            detections = self.run_object_detection(frames)
            
            # 4. Tracking
            tracked_detections = self.run_tracking(detections)
            
            # 5. Pose analysis
            pose_data = self.analyze_pose(frames, tracked_detections)
            
            # 6. Rule evaluation
            violations = self.evaluate_compliance_rules(tracked_detections, pose_data)
            
            # 7. Generate evidence
            violations_with_evidence = self.generate_evidence(violations, video_path, output_dir)
            
            # 8. Calculate score
            score_data = self.calculate_compliance_score(violations_with_evidence, video_metadata['duration'])
            
            # 9. Compile results
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            results = {
                'analysis_id': f"analysis_{int(start_time.timestamp())}",
                'timestamp': start_time.isoformat(),
                'processing_time': round(processing_time, 2),
                'video_metadata': video_metadata,
                'industry_pack': self.industry_pack,
                'compliance_score': score_data,
                'violations': violations_with_evidence,
                'total_detections': len(tracked_detections),
                'timeline': self._generate_timeline(violations_with_evidence),
                'model_info': {
                    'yolo_available': YOLO_AVAILABLE,
                    'mediapipe_available': MEDIAPIPE_AVAILABLE,
                    'deepsort_available': DEEPSORT_AVAILABLE
                }
            }
            
            print(f"✅ Analysis complete! Score: {score_data['compliance_score']}/100 ({score_data['grade']})")
            return results
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise
    
    def _generate_timeline(self, violations: List[Dict]) -> List[Dict]:
        """Generate timeline markers for violations"""
        timeline = []
        for violation in violations:
            timeline.append({
                'timestamp': violation['timestamp'],
                'type': violation['type'],
                'severity': violation['severity'],
                'description': violation['description']
            })
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        return timeline


class MockTracker:
    """Mock tracker for when DeepSort is not available"""
    
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
    
    def update_tracks(self, detections):
        """Mock tracking implementation"""
        tracked_objects = []
        
        for det in detections:
            # Simple assignment of track IDs
            center_x = (det['bbox'][0] + det['bbox'][2]) / 2
            center_y = (det['bbox'][1] + det['bbox'][3]) / 2
            center = (center_x, center_y)
            
            # Find closest existing track or create new one
            closest_id = None
            min_distance = float('inf')
            
            for track_id, track_info in self.tracks.items():
                last_center = track_info['last_center']
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                if distance < min_distance and distance < 100:  # Threshold
                    min_distance = distance
                    closest_id = track_id
            
            if closest_id is None:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    'created_at': det['timestamp'],
                    'last_center': center,
                    'last_seen': det['timestamp']
                }
            else:
                track_id = closest_id
                self.tracks[track_id]['last_center'] = center
                self.tracks[track_id]['last_seen'] = det['timestamp']
            
            # Add tracking info
            tracked_det = det.copy()
            tracked_det['track_id'] = track_id
            tracked_objects.append(tracked_det)
        
        return tracked_objects


if __name__ == "__main__":
    # Test the engine
    engine = ComplianceEngine(industry_pack='manufacturing')
    
    # Test video type detection
    test_video = "data/ssvid.net--Toyota-VR-360-Factory-Tour_v720P.mp4"
    if os.path.exists(test_video):
        metadata = engine.detect_video_type(test_video)
        print(f"Test video metadata: {metadata}")
    else:
        print("Test video not found - engine ready for deployment")
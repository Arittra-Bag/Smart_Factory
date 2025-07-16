
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

class LiveEnsembleDetector:
    def __init__(self):
        self.base_path = os.getcwd()
        self.runs_path = os.path.join(self.base_path, 'runs', 'detect')
        self.models = {}
        self.model_colors = {}
        self.ensemble_results = {}
        self.enabled_models = {}  # Track which models are currently enabled
        self.stable_detections = []  # Store stable bounding boxes
        self.previous_frame = None  # Store previous frame for motion detection
        
    def scan_and_load_models(self):
        """Scan all detection runs and load the best models"""
        if not os.path.exists(self.runs_path):
            print(f"Error: Detection runs directory not found at {self.runs_path}")
            return False
        
        detection_runs = []
        for run_dir in os.listdir(self.runs_path):
            run_path = os.path.join(self.runs_path, run_dir)
            if os.path.isdir(run_path):
                weights_path = os.path.join(run_path, 'weights')
                if os.path.exists(weights_path):
                    detection_runs.append({
                        'name': run_dir,
                        'best_model': os.path.join(weights_path, 'best.pt')
                    })
        
        if not detection_runs:
            print("No trained models found!")
            print("Please train at least one model first using testing.py or testing_camera.py")
            return False
        

        
        # Load all models
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                 (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0)]
        
        for i, run in enumerate(detection_runs):
            model_path = run['best_model']
            if os.path.exists(model_path):
                try:
                    model = YOLO(model_path)
                    self.models[run['name']] = model
                    self.model_colors[run['name']] = colors[i % len(colors)]
                    self.enabled_models[run['name']] = True  # Enable by default
                except Exception as e:
                    print(f"✗ Failed to load {run['name']}: {e}")
            else:
                print(f"✗ Model not found: {model_path}")
        
        if not self.models:
            print("No valid models could be loaded!")
            return False
        

        return True
    
    def run_ensemble_detection(self, frame, conf_threshold=0.25, iou_threshold=0.45):
        """Run detection with all models on a single frame"""
        claimed_detections = []  # Detections that have been claimed by first model
        
        for model_name, model in self.models.items():
            # Skip disabled models
            if not self.enabled_models.get(model_name, True):
                continue
                
            try:
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    save=False,
                    save_txt=False,
                    verbose=False
                )
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            new_detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'model': model_name,
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            
                            # Check if this detection overlaps with any already claimed detection
                            is_overlapping = False
                            for claimed in claimed_detections:
                                iou = self.calculate_iou(new_detection['bbox'], claimed['bbox'])
                                if iou > 0.3:  # If significant overlap, don't add this detection
                                    is_overlapping = True
                                    break
                            
                            # Only add if it doesn't significantly overlap with existing detections
                            if not is_overlapping:
                                claimed_detections.append(new_detection)
                        
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue
        
        return claimed_detections
    
    def detect_frame_motion(self, current_frame, motion_threshold=30.0):
        """Detect if there's significant motion between frames"""
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return True  # First frame, assume motion
        
        # Convert current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(self.previous_frame, current_gray)
        
        # Calculate mean difference
        mean_diff = cv2.mean(frame_diff)[0]
        
        # Update previous frame
        self.previous_frame = current_gray.copy()
        
        # Return True if motion detected
        return mean_diff > motion_threshold
    
    def update_stable_detections(self, new_detections, has_motion):
        """Update stable detections based on motion and new detections"""
        if has_motion or not self.stable_detections:
            # If there's motion or no stable detections, update with new detections
            self.stable_detections = new_detections.copy()
        else:
            # No motion detected, keep existing stable detections
            # But update confidence scores if new detections overlap with stable ones
            for stable in self.stable_detections:
                for new_det in new_detections:
                    iou = self.calculate_iou(stable['bbox'], new_det['bbox'])
                    if iou > 0.7:  # High overlap, update confidence
                        stable['confidence'] = max(stable['confidence'], new_det['confidence'])
                        break
        
        return self.stable_detections
    
    def apply_nms(self, detections, iou_threshold=0.5):
        """Apply non-maximum suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            final_detections.append(current)
            
            # Remove overlapping detections
            remaining = []
            for detection in detections:
                iou = self.calculate_iou(current['bbox'], detection['bbox'])
                if iou < iou_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return final_detections
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, frame, detections):
        """Draw detections on the frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            model = detection['model']
            
            # Get color for this model
            color = self.model_colors[model]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw small model indicator (optional - can be removed for cleaner look)
            # cv2.circle(frame, (x1+10, y1+10), 5, color, -1)
        
        return frame
    
    def add_info_panel(self, frame, detections, fps, frame_count, conf_threshold=0.25, iou_threshold=0.45, motion_threshold=30.0, motion_detection_enabled=True):
        """Add information panel to the frame"""
        # Count enabled models
        enabled_count = sum(1 for enabled in self.enabled_models.values() if enabled)
        
        # Create semi-transparent overlay for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        cv2.putText(frame, f"Live Ensemble Detection", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Models: {enabled_count}/{len(self.models)}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {frame_count}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {conf_threshold:.2f}", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"IOU: {iou_threshold:.2f}", (20, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        motion_status = "ON" if motion_detection_enabled else "OFF"
        motion_color = (255, 100, 255) if motion_detection_enabled else (100, 100, 100)
        cv2.putText(frame, f"Motion: {motion_status} ({motion_threshold:.1f})", (20, 175), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
        cv2.putText(frame, f"m=models | p=snapshot analysis", (20, 195), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def add_model_legend(self, frame):
        """Add model legend showing colors and status"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Calculate legend dimensions
        legend_height = len(self.models) * 25 + 40
        legend_width = 250
        
        # Position in bottom-right corner
        x_offset = width - legend_width - 20
        y_offset = height - legend_height - 20
        
        # Draw background rectangle
        cv2.rectangle(overlay, (x_offset-10, y_offset-10), 
                     (x_offset + legend_width, y_offset + legend_height), 
                     (0, 0, 0), -1)
        
        # Apply transparency
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw title
        cv2.putText(frame, "MODEL LEGEND:", (x_offset, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw controls info
        cv2.putText(frame, "m=toggle | t=all | a-z=individual", (x_offset, y_offset + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        
        # Draw each model
        for i, (model_name, color) in enumerate(self.model_colors.items()):
            y = y_offset + 55 + (i * 25)
            
            # Draw color box
            cv2.rectangle(frame, (x_offset, y-10), (x_offset + 20, y+5), color, -1)
            
            # Model status
            status = "ON" if self.enabled_models.get(model_name, True) else "OFF"
            status_color = (0, 255, 0) if self.enabled_models.get(model_name, True) else (0, 0, 255)
            
            # Draw keyboard shortcut
            shortcut = chr(ord('a') + i) if i < 26 else "?"
            
            # Draw model name, shortcut, and status
            model_display = model_name[:12] + "..." if len(model_name) > 12 else model_name
            cv2.putText(frame, f"{shortcut}: {model_display} ({status})", (x_offset + 25, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        return frame
    
    def take_snapshot_and_analyze(self, frame, conf_threshold, iou_threshold, motion_threshold, motion_detection_enabled=True):
        """Take a snapshot and run detailed analysis with current settings"""
        import time
        from datetime import datetime
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original snapshot (high quality)
        snapshot_filename = f'snapshot_original_{timestamp}.jpg'
        cv2.imwrite(snapshot_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"\n=== SNAPSHOT ANALYSIS ===")
        print(f"Timestamp: {timestamp}")
        print(f"Original snapshot saved: {snapshot_filename}")
        print(f"Settings used:")
        print(f"  - Confidence threshold: {conf_threshold:.3f}")
        print(f"  - IOU threshold: {iou_threshold:.3f}")
        print(f"  - Motion detection: {'ENABLED' if motion_detection_enabled else 'DISABLED'}")
        print(f"  - Motion threshold: {motion_threshold:.1f}")
        
        # Run detection with current settings
        detections = self.run_ensemble_detection(frame, conf_threshold, iou_threshold)
        
        print(f"  - Enabled models: {sum(1 for enabled in self.enabled_models.values() if enabled)}/{len(self.models)}")
        print(f"  - Total detections: {len(detections)}")
        
        # Analyze detections per model
        model_stats = {}
        for detection in detections:
            model = detection['model']
            if model not in model_stats:
                model_stats[model] = {'count': 0, 'avg_confidence': 0, 'confidences': []}
            model_stats[model]['count'] += 1
            model_stats[model]['confidences'].append(detection['confidence'])
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats['confidences']:
                stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
                stats['min_confidence'] = min(stats['confidences'])
                stats['max_confidence'] = max(stats['confidences'])
        
        print(f"\nModel Performance:")
        for model_name in self.models.keys():
            if self.enabled_models.get(model_name, True):
                if model_name in model_stats:
                    stats = model_stats[model_name]
                    print(f"  {model_name}:")
                    print(f"    - Detections: {stats['count']}")
                    print(f"    - Avg confidence: {stats['avg_confidence']:.3f}")
                    print(f"    - Range: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}")
                else:
                    print(f"  {model_name}: No detections")
            else:
                print(f"  {model_name}: DISABLED")
        
        # Draw detections on frame
        analyzed_frame = frame.copy()
        analyzed_frame = self.draw_detections(analyzed_frame, detections)
        
        # Add detailed info panel for snapshot
        analyzed_frame = self.add_snapshot_info_panel(analyzed_frame, detections, conf_threshold, iou_threshold, motion_threshold, model_stats, motion_detection_enabled)
        
        # Save analyzed snapshot
        analyzed_filename = f'snapshot_analyzed_{timestamp}.jpg'
        cv2.imwrite(analyzed_filename, analyzed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Analyzed snapshot saved: {analyzed_filename}")
        
        # Display snapshot for review
        cv2.namedWindow('Snapshot Analysis', cv2.WINDOW_NORMAL)
        height, width = analyzed_frame.shape[:2]
        if width > 1920:
            scale = 1920 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_frame = cv2.resize(analyzed_frame, (new_width, new_height))
        else:
            display_frame = analyzed_frame
            
        cv2.imshow('Snapshot Analysis', display_frame)
        print(f"Press any key to close snapshot analysis...")
        cv2.waitKey(0)
        cv2.destroyWindow('Snapshot Analysis')
        
        print(f"=== ANALYSIS COMPLETE ===\n")
        
        return detections
    
    def add_snapshot_info_panel(self, frame, detections, conf_threshold, iou_threshold, motion_threshold, model_stats, motion_detection_enabled=True):
        """Add detailed info panel for snapshot analysis"""
        height, width = frame.shape[:2]
        
        # Create large info panel
        overlay = frame.copy()
        panel_width = 400
        panel_height = 150 + len(self.models) * 20
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        y_pos = 35
        # Title
        cv2.putText(frame, "SNAPSHOT ANALYSIS", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30
        
        # Settings
        cv2.putText(frame, f"Confidence: {conf_threshold:.3f} | IOU: {iou_threshold:.3f}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 20
        motion_text = f"Motion: {'ON' if motion_detection_enabled else 'OFF'} ({motion_threshold:.1f})"
        cv2.putText(frame, motion_text, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
        y_pos += 25
        
        # Total detections
        cv2.putText(frame, f"Total Detections: {len(detections)}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25
        
        # Model performance
        cv2.putText(frame, "Model Performance:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += 20
        
        for model_name, color in self.model_colors.items():
            if self.enabled_models.get(model_name, True):
                if model_name in model_stats:
                    stats = model_stats[model_name]
                    text = f"{model_name[:15]}: {stats['count']} ({stats['avg_confidence']:.2f})"
                    cv2.putText(frame, text, (30, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.putText(frame, f"{model_name[:15]}: 0", (30, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            else:
                cv2.putText(frame, f"{model_name[:15]}: DISABLED", (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            y_pos += 18
        
        return frame
    
    def add_image_info_panel(self, frame, detections, conf_threshold, iou_threshold, detection_count):
        """Add information panel for single image analysis"""
        # Count enabled models
        enabled_count = sum(1 for enabled in self.enabled_models.values() if enabled)
        
        # Create semi-transparent overlay for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        cv2.putText(frame, f"Interactive Image Analysis", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Models: {enabled_count}/{len(self.models)}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {conf_threshold:.3f}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"IOU: {iou_threshold:.3f}", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(frame, f"Controls: +/- [ ] m t a-z p s r", (20, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def run_live_ensemble_detection(self, rtsp_url=None):
        """Run live ensemble detection on camera feed"""
        if not self.scan_and_load_models():
            return
        
        # Get RTSP URL
        if not rtsp_url:
            rtsp_url = input("Enter RTSP URL (or press Enter for default): ").strip()
            if not rtsp_url:
                rtsp_url = "rtsp://admin123:admin123@192.168.1.116/stream1"
        
        print(f"Connecting to camera: {rtsp_url}")
        
        # Open the RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print(f"Error: Could not connect to camera at {rtsp_url}")
            return
        

        
        # Performance tracking
        frame_count = 0
        saved_frames = 0
        start_time = time.time()
        fps = 0
        
        # Detection settings
        conf_threshold = 0.25
        iou_threshold = 0.45
        
        # Statistics
        total_detections = 0
        detection_history = []
        
        # UI state
        show_legend = True
        motion_threshold = 30.0  # Adjustable motion sensitivity
        motion_detection_enabled = True  # Toggle for motion detection
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from camera")
                break
            
            # Run ensemble detection
            new_detections = self.run_ensemble_detection(frame, conf_threshold, iou_threshold)
            
            # Handle motion detection and stable tracking
            if motion_detection_enabled:
                # Detect motion in frame
                has_motion = self.detect_frame_motion(frame, motion_threshold)
                # Update stable detections based on motion
                detections = self.update_stable_detections(new_detections, has_motion)
            else:
                # Motion detection disabled, use real-time detections
                detections = new_detections
            
            # Update statistics
            total_detections += len(detections)
            detection_history.append(len(detections))
            if len(detection_history) > 30:  # Keep last 30 frames
                detection_history.pop(0)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - start_time)
                start_time = current_time
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            # Add information panel
            frame = self.add_info_panel(frame, detections, fps, frame_count, conf_threshold, iou_threshold, motion_threshold, motion_detection_enabled)
            
            # Add model legend if enabled
            if show_legend:
                frame = self.add_model_legend(frame)
            
            # Display the frame
            cv2.imshow('Live Ensemble Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame with detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'ensemble_detection_frame_{timestamp}_{saved_frames}.jpg'
                cv2.imwrite(filename, frame)
 
                saved_frames += 1
            elif key == ord('p'):
                # Take snapshot and perform detailed analysis
                print("Taking snapshot for analysis...")
                self.take_snapshot_and_analyze(frame, conf_threshold, iou_threshold, motion_threshold, motion_detection_enabled)
            elif key == ord('r'):
                # Reset counters
                total_detections = 0
                detection_history = []

            elif key in [ord(str(i)) for i in range(1, 10)]:
                # Adjust confidence threshold (1-9 keys)
                conf_threshold = (key - ord('0')) * 0.1
                conf_threshold = max(0.1, min(0.9, conf_threshold))
            elif key == ord('+') or key == ord('='):
                # Increase confidence threshold
                conf_threshold = min(0.95, conf_threshold + 0.01)
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                conf_threshold = max(0.05, conf_threshold - 0.01)
            elif key == ord('['):
                # Increase IOU threshold
                iou_threshold = min(0.95, iou_threshold + 0.01)
            elif key == ord(']'):
                # Decrease IOU threshold
                iou_threshold = max(0.05, iou_threshold - 0.01)
            elif key == ord('u'):
                # Increase motion threshold (less sensitive)
                motion_threshold = min(100.0, motion_threshold + 1.0)
            elif key == ord('i'):
                # Decrease motion threshold (more sensitive)
                motion_threshold = max(5.0, motion_threshold - 1.0)
            elif key == ord('o'):
                # Toggle motion detection on/off
                motion_detection_enabled = not motion_detection_enabled
            elif key == ord('m'):
                # Toggle model legend
                show_legend = not show_legend
            elif key == ord('t'):
                # Toggle all models on/off
                all_enabled = all(self.enabled_models.values())
                for model_name in self.enabled_models:
                    self.enabled_models[model_name] = not all_enabled
            elif key >= ord('a') and key <= ord('z'):
                # Toggle individual models (a-z keys for first 26 models)
                model_index = key - ord('a')
                model_names = list(self.models.keys())
                if model_index < len(model_names):
                    model_name = model_names[model_index]
                    self.enabled_models[model_name] = not self.enabled_models[model_name]

        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
    def run_ensemble_detection_on_image(self, image_path, output_path=None):
        """Run interactive ensemble detection on a single image"""
        if not self.scan_and_load_models():
            return
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        # Read image
        original_frame = cv2.imread(image_path)
        if original_frame is None:
            print(f"Could not load image: {image_path}")
            return
        
        print(f"\nInteractive Image Analysis Mode")
        print(f"Image: {image_path}")
        print(f"Controls:")
        print(f"- Press 'q' or ESC to quit")
        print(f"- Press '+/-' to adjust confidence (±0.01)")
        print(f"- Press '[]' to adjust IOU (±0.01)")
        print(f"- Press '1-9' to set confidence (0.1-0.9)")
        print(f"- Press 'm' to toggle model legend")
        print(f"- Press 't' to toggle all models")
        print(f"- Press 'a-z' to toggle individual models")
        print(f"- Press 'p' to take snapshot analysis")
        print(f"- Press 's' to save current result")
        print(f"- Press 'r' to reset to default settings")
        
        # Detection settings
        conf_threshold = 0.25
        iou_threshold = 0.45
        motion_threshold = 30.0  # Not used but needed for info panel
        motion_detection_enabled = False  # Not applicable for single image
        
        # UI state
        show_legend = True
        saved_frames = 0
        
        # Create window
        cv2.namedWindow('Interactive Image Analysis', cv2.WINDOW_NORMAL)
        height, width = original_frame.shape[:2]
        if width > 1920:
            scale = 1920 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv2.resizeWindow('Interactive Image Analysis', new_width, new_height)
        
        while True:
            # Create working copy of image
            frame = original_frame.copy()
            
            # Run ensemble detection with current settings
            detections = self.run_ensemble_detection(frame, conf_threshold, iou_threshold)
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            # Add information panel (adapted for single image)
            frame = self.add_image_info_panel(frame, detections, conf_threshold, iou_threshold, len(detections))
            
            # Add model legend if enabled
            if show_legend:
                frame = self.add_model_legend(frame)
            
            # Display the frame
            cv2.imshow('Interactive Image Analysis', frame)
            
            # Handle key presses
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC key
                break
            elif key == ord('s'):
                # Save current result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'image_analysis_result_{timestamp}_{saved_frames}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Saved result as {filename}")
                saved_frames += 1
            elif key == ord('r'):
                # Reset to default settings
                conf_threshold = 0.25
                iou_threshold = 0.45
                for model_name in self.enabled_models:
                    self.enabled_models[model_name] = True
                print("Reset to default settings")
            elif key in [ord(str(i)) for i in range(1, 10)]:
                # Adjust confidence threshold (1-9 keys)
                conf_threshold = (key - ord('0')) * 0.1
                conf_threshold = max(0.1, min(0.9, conf_threshold))
            elif key == ord('+') or key == ord('='):
                # Increase confidence threshold
                conf_threshold = min(0.95, conf_threshold + 0.01)
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                conf_threshold = max(0.05, conf_threshold - 0.01)
            elif key == ord('['):
                # Increase IOU threshold
                iou_threshold = min(0.95, iou_threshold + 0.01)
            elif key == ord(']'):
                # Decrease IOU threshold
                iou_threshold = max(0.05, iou_threshold - 0.01)
            elif key == ord('m'):
                # Toggle model legend
                show_legend = not show_legend
            elif key == ord('t'):
                # Toggle all models on/off
                all_enabled = all(self.enabled_models.values())
                for model_name in self.enabled_models:
                    self.enabled_models[model_name] = not all_enabled
            elif key >= ord('a') and key <= ord('z'):
                # Toggle individual models (a-z keys for first 26 models)
                model_index = key - ord('a')
                model_names = list(self.models.keys())
                if model_index < len(model_names):
                    model_name = model_names[model_index]
                    self.enabled_models[model_name] = not self.enabled_models[model_name]
            elif key == ord('p'):
                # Take snapshot and perform detailed analysis
                print("Running detailed analysis...")
                self.take_snapshot_and_analyze(frame, conf_threshold, iou_threshold, motion_threshold, motion_detection_enabled)
        
        # Clean up
        cv2.destroyAllWindows()
        
        # Save final result if output path provided
        if output_path:
            final_frame = original_frame.copy()
            final_detections = self.run_ensemble_detection(final_frame, conf_threshold, iou_threshold)
            final_frame = self.draw_detections(final_frame, final_detections)
            final_frame = self.add_image_info_panel(final_frame, final_detections, conf_threshold, iou_threshold, len(final_detections))
            cv2.imwrite(output_path, final_frame)
            print(f"Final result saved to: {output_path}")
        
        print(f"Interactive analysis complete. Final detections: {len(detections)}")
        return detections

def main():
    detector = LiveEnsembleDetector()
    
    print("Live Ensemble Detection System")
    print("=" * 40)
    print("1. Live camera ensemble detection")
    print("2. Single image ensemble detection")
    print("3. Test ensemble detection on multiple images")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Live camera ensemble detection
        print("\nLive Camera Ensemble Detection")
        print("=" * 30)
        
        # Get RTSP URL
        rtsp_url = input("Enter RTSP URL (or press Enter for default): ").strip()
        if not rtsp_url:
            rtsp_url = "rtsp://admin123:admin123@192.168.1.116/stream1"
        
        detector.run_live_ensemble_detection(rtsp_url)
    
    elif choice == "2":
        # Single image ensemble detection
        print("\nSingle Image Ensemble Detection")
        print("=" * 30)
        
        image_path = input("Enter image path: ").strip()
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        output_path = input("Enter output path (or press Enter to display): ").strip()
        if not output_path:
            output_path = None
        
        detector.run_ensemble_detection_on_image(image_path, output_path)
    
    elif choice == "3":
        # Test on multiple images
        print("\nTest Ensemble Detection on Multiple Images")
        print("=" * 30)
        
        image_paths = input("Enter image paths (comma-separated): ").strip().split(',')
        image_paths = [img.strip() for img in image_paths if img.strip()]
        
        if not image_paths:
            print("No images provided!")
            return
        
        # Validate images exist
        valid_images = []
        for img in image_paths:
            if os.path.exists(img):
                valid_images.append(img)
            else:
                print(f"Warning: Image not found: {img}")
        
        if not valid_images:
            print("No valid images found!")
            return
        
        print(f"Testing ensemble detection on {len(valid_images)} images...")
        
        for i, image_path in enumerate(valid_images):
            print(f"\nProcessing image {i+1}/{len(valid_images)}: {image_path}")
            output_path = f"ensemble_result_{i+1}.jpg"
            detections = detector.run_ensemble_detection_on_image(image_path, output_path)
            print(f"Found {len(detections)} detections")
        
        print(f"\nAll images processed. Results saved as ensemble_result_*.jpg")
    
    else:
        print("Invalid choice. Please run the script again and select 1, 2, or 3.")

if __name__ == "__main__":
    main() 
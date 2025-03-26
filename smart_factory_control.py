import cv2
import mediapipe as mp
import numpy as np
import time
from collections import defaultdict
import threading
import datetime
import json
import torch
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class SmartFactoryController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize tracking
        self.gesture_stats = defaultdict(int)
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0
        self.current_gesture = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Industrial control states
        self.machine_status = "STANDBY"  # STANDBY, RUNNING, EMERGENCY
        self.quality_score = 100.0
        self.safety_status = "SECURE"
        self.defect_count = 0
        self.production_count = 0
        self.emergency_mode = False
        self.last_inspection_time = time.time()
        self.emergency_reset_start = 0  # Track when emergency reset started
        self.emergency_reset_duration = 3.0  # Seconds to hold palm to reset emergency
        
        # Load quality inspection model (simulated)
        self.quality_model = self.load_quality_model()
        
        # Initialize alert system
        self.alerts = []
        self.alert_thread = threading.Thread(target=self.alert_monitor, daemon=True)
        self.alert_thread.start()

    def load_quality_model(self):
        """Simulate loading a pre-trained quality inspection model"""
        # In real implementation, load actual model
        class MockModel:
            def __call__(self, x):
                return torch.tensor([np.random.normal(0.95, 0.05)])
            def eval(self):
                pass
        return MockModel()

    def alert_monitor(self):
        """Background thread for monitoring system alerts"""
        while True:
            if self.emergency_mode:
                self.alerts.append(f"⚠️ EMERGENCY STOP ACTIVATED - {datetime.datetime.now()}")
            time.sleep(1)

    def detect_gestures(self, hand_landmarks):
        """Enhanced gesture detection for industrial control"""
        if not hand_landmarks:
            return None

        # Get finger landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        # Emergency Stop (Fist)
        emergency_stop = (
            index_tip.y > index_pip.y and
            middle_tip.y > middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y
        )

        # Start Production (Peace Sign)
        start_production = (
            index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y
        )

        # Quality Check (Palm)
        quality_check = (
            index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y < ring_pip.y and
            pinky_tip.y < pinky_pip.y
        )

        if emergency_stop:
            return "emergency_stop"
        elif start_production:
            return "start_production"
        elif quality_check:
            return "quality_check"
        return None

    def apply_industrial_effect(self, frame, gesture):
        """Apply visual effects based on industrial context"""
        if gesture == "emergency_stop":
            # Red emergency overlay with reduced opacity
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # Reduced opacity
            
        elif gesture == "start_production":
            # Green production overlay with scanning effect
            if self.machine_status != "EMERGENCY":
                scan_line_pos = int((time.time() * 200) % frame.shape[0])
                cv2.line(frame, (0, scan_line_pos), (frame.shape[1], scan_line_pos), 
                        (0, 255, 0), 2)
                self.production_count += 1
                
        elif gesture == "quality_check":
            # Blue quality inspection overlay
            if time.time() - self.last_inspection_time > 2.0:
                self.last_inspection_time = time.time()
                # Simulate quality inspection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                frame_tensor = transform(frame_pil).unsqueeze(0)
                
                with torch.no_grad():
                    quality_score = float(self.quality_model(frame_tensor).item())
                self.quality_score = quality_score * 100
                
                if quality_score < 0.9:
                    self.defect_count += 1
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    
        return frame

    def draw_industrial_hud(self, frame):
        """Draw industrial HUD with production metrics"""
        # Create a black panel on the right side
        panel_width = 250
        original_frame = frame.copy()
        frame = cv2.copyMakeBorder(
            original_frame,
            0, 0, 0, panel_width,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        x_pos = int(frame.shape[1] - panel_width + 10)
        y_pos = 40
        line_spacing = 35
        
        # Draw title
        cv2.putText(frame, "SMART FACTORY", (int(x_pos), int(y_pos)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(frame, "CONTROL SYSTEM", (int(x_pos), int(y_pos)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += int(line_spacing * 1.2)

        # System Status with color
        status_text = f"Status: {self.machine_status}"
        status_color = (0, 255, 0) if self.machine_status == "RUNNING" else \
                      (0, 0, 255) if self.machine_status == "EMERGENCY" else \
                      (255, 255, 0)
        cv2.putText(frame, status_text, (int(x_pos), int(y_pos)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show emergency reset progress if active
        if self.emergency_mode and self.emergency_reset_start > 0:
            progress = min((time.time() - self.emergency_reset_start) / self.emergency_reset_duration * 100, 100)
            y_pos += 20
            cv2.putText(frame, f"Reset Progress: {progress:.0f}%", (int(x_pos), int(y_pos)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y_pos += line_spacing

        # Draw divider
        cv2.line(frame, (int(x_pos - 5), int(y_pos)), 
                (int(frame.shape[1] - 10), int(y_pos)), (200, 200, 200), 1)
        y_pos += line_spacing

        # Production Metrics
        cv2.putText(frame, f"Production: {self.production_count}", 
                   (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        cv2.putText(frame, f"Quality: {self.quality_score:.1f}%", 
                   (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        cv2.putText(frame, f"Defects: {self.defect_count}", 
                   (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += line_spacing

        # Draw divider
        cv2.line(frame, (int(x_pos - 5), int(y_pos)),
                (int(frame.shape[1] - 10), int(y_pos)), (200, 200, 200), 1)
        y_pos += line_spacing

        # Gesture Controls Guide
        cv2.putText(frame, "GESTURE CONTROLS:", 
                   (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += line_spacing
        cv2.putText(frame, "Fist - Emergency Stop", 
                   (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_pos += line_spacing
        cv2.putText(frame, "Peace - Start Production", 
                   (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_spacing
        cv2.putText(frame, "Palm - Quality Check", 
                   (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += line_spacing * 1.2

        # Current Hand Status
        if self.current_gesture:
            status_text = "Current Hand: "
            if self.current_gesture == "emergency_stop":
                status_text += "FIST"
                status_color = (0, 0, 255)
            elif self.current_gesture == "start_production":
                status_text += "PEACE"
                status_color = (0, 255, 0)
            elif self.current_gesture == "quality_check":
                status_text += "PALM"
                status_color = (255, 255, 0)
            cv2.putText(frame, status_text, (int(x_pos), int(y_pos)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        else:
            cv2.putText(frame, "Current Hand: NONE", (int(x_pos), int(y_pos)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        return frame

    def process_frame(self, frame):
        # Calculate FPS
        self.calculate_fps()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Create output frame
        output_frame = frame.copy()
        
        # Reset current gesture if no hands detected
        if not results.multi_hand_landmarks:
            self.current_gesture = None
            self.emergency_reset_start = 0  # Reset emergency reset timer
            if not self.emergency_mode:
                self.machine_status = "STANDBY"
        
        if results.multi_hand_landmarks:
            # Process the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks with industrial theme
            self.mp_draw.draw_landmarks(
                output_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)
            )
            
            # Detect gestures and apply effects
            gesture = self.detect_gestures(hand_landmarks)
            if gesture:
                # Update current gesture immediately
                self.current_gesture = gesture
                
                # Handle emergency reset with palm gesture
                if gesture == "quality_check" and self.emergency_mode:
                    if self.emergency_reset_start == 0:
                        self.emergency_reset_start = time.time()
                    elif time.time() - self.emergency_reset_start >= self.emergency_reset_duration:
                        self.emergency_mode = False
                        self.machine_status = "STANDBY"
                        self.emergency_reset_start = 0
                else:
                    self.emergency_reset_start = 0
                
                # Update machine status based on current gesture
                if gesture == "emergency_stop":
                    self.machine_status = "EMERGENCY"
                    self.emergency_mode = True
                elif gesture == "start_production":
                    if not self.emergency_mode:
                        self.machine_status = "RUNNING"
                    # Allow production count to increase even in emergency mode
                    if time.time() - self.last_gesture_time > self.gesture_cooldown:
                        self.production_count += 1
                elif gesture == "quality_check":
                    if not self.emergency_mode:
                        self.machine_status = "QUALITY CHECK"
                
                # Update stats with cooldown
                if time.time() - self.last_gesture_time > self.gesture_cooldown:
                    self.gesture_stats[gesture] += 1
                    self.last_gesture_time = time.time()
                
                output_frame = self.apply_industrial_effect(output_frame, gesture)
        
        # Draw industrial HUD
        output_frame = self.draw_industrial_hud(output_frame)
            
        return output_frame

    def calculate_fps(self):
        self.frame_count += 1
        if time.time() - self.start_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()

def main():
    print("Initializing Smart Factory Control System...")
    print("Loading AI models and calibrating sensors...")
    time.sleep(2)  # Simulate initialization
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Initialize the controller
    controller = SmartFactoryController()
    
    print("\nSmart Factory Control System Ready!")
    print("\nGesture Controls:")
    print("- Fist: Emergency Stop (Immediate machine shutdown)")
    print("- Peace Sign: Start Production Line")
    print("- Palm: Trigger Quality Inspection")
    print("\nPress 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        output_frame = controller.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Smart Factory Control System', output_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(200) & 0xFF == ord('q'):  # 5 FPS for better gesture recognition
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
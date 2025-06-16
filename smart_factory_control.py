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
import matplotlib.pyplot as plt
import pandas as pd
import os
import streamlit as st
from auth_manager import AuthManager

# Initialize session state for camera and production data
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'production_data' not in st.session_state:
    st.session_state.production_data = {
        'production_count': 0,
        'defect_count': 0,
        'batch_size': 0,
        'quality_score': 100.0,
        'machine_status': "STANDBY",
        'emergency_mode': False
    }

# Initialize authentication manager
auth_manager = AuthManager()

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
        
        # Initialize tracking from session state
        self.gesture_stats = defaultdict(int)
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0
        self.current_gesture = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Load production data from session state
        self.production_count = st.session_state.production_data['production_count']
        self.defect_count = st.session_state.production_data['defect_count']
        self.batch_size = st.session_state.production_data['batch_size']
        self.quality_score = st.session_state.production_data['quality_score']
        self.machine_status = st.session_state.production_data['machine_status']
        self.emergency_mode = st.session_state.production_data['emergency_mode']
        
        self.last_inspection_time = time.time()
        self.emergency_reset_start = 0
        self.emergency_reset_duration = 3.0
        
        # Load quality inspection model (simulated)
        self.quality_model = self.load_quality_model()

        # Safety check tracking
        self.last_batch_size = 0
        self.safety_check_done = False
        self.safety_check_records = "safety_check_records.csv"
        
        # Ensure CSV file exists
        if not os.path.exists(self.safety_check_records):
            with open(self.safety_check_records, 'w') as f:
                f.write("Batch Size,Defect Rate,Timestamp\n")
        
        # Initialize alert system
        self.alerts = []
        self.alert_thread = threading.Thread(target=self.alert_monitor, daemon=True)
        self.alert_thread.start()

    def update_session_state(self):
        """Update session state with current production data"""
        st.session_state.production_data.update({
            'production_count': self.production_count,
            'defect_count': self.defect_count,
            'batch_size': self.batch_size,
            'quality_score': self.quality_score,
            'machine_status': self.machine_status,
            'emergency_mode': self.emergency_mode
        })

    def reset_production(self):
        """Reset all production data"""
        self.production_count = 0
        self.defect_count = 0
        self.batch_size = 0
        self.quality_score = 100.0
        self.machine_status = "STANDBY"
        self.emergency_mode = False
        self.update_session_state()

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

        # Plot Batch vs. Defect Rate (Index Finger Raised)
        plot_defect_rate = (
            index_tip.y < index_pip.y and
            middle_tip.y > middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y
        )

        if emergency_stop:
            return "emergency_stop"
        elif start_production:
            return "start_production"
        elif quality_check:
            return "quality_check"
        elif plot_defect_rate:
            return "plot_defect_rate"
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
                cv2.line(frame, (0, scan_line_pos), (frame.shape[1], scan_line_pos), (0, 255, 0), 2)
                self.production_count += 1
                self.batch_size = self.production_count  # Synchronize batch size with production count
                
        elif gesture == "quality_check":
            if self.production_count < 10:
                cv2.putText(frame, "Min 10 products needed for safety check", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif self.safety_check_done and self.batch_size == self.last_batch_size:
                cv2.putText(frame, "Safety check already done for this batch", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                if time.time() - self.last_inspection_time > 2.0:
                    self.last_inspection_time = time.time()
                    self.last_batch_size = self.batch_size
                    self.safety_check_done = True

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
                    self.quality_score = min(quality_score * 100, 100)  # Ensure max 100%

                    if quality_score < 90:  # Adjust threshold for defect detection
                        self.defect_count += 1

                    # Calculate defect rate
                    defect_rate = (self.defect_count / self.batch_size) * 100

                    # Save safety check record
                    with open(self.safety_check_records, 'a') as f:
                        f.write(f"{self.batch_size},{defect_rate},{datetime.datetime.now()}\n")
                        
        elif gesture == "plot_defect_rate":
            self.plot_defect_rate()

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

    def plot_defect_rate(self):
        """Plot batch size vs. defect rate."""
        if os.path.exists(self.safety_check_records):
            data = pd.read_csv(self.safety_check_records)
            if not data.empty:
                # Create figure with larger size
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot with improved styling
                ax.plot(data["Batch Size"], data["Defect Rate"], 
                       marker='o', linewidth=2, markersize=8,
                       color='#2ecc71', markeredgecolor='white',
                       markeredgewidth=2)
                
                # Customize grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Set labels and title with better fonts
                ax.set_xlabel("Batch Size", fontsize=12, fontweight='bold')
                ax.set_ylabel("Defect Rate (%)", fontsize=12, fontweight='bold')
                ax.set_title("Batch Size vs. Defect Rate Analysis", 
                           fontsize=14, fontweight='bold', pad=20)
                
                # Add mean defect rate line
                mean_rate = data["Defect Rate"].mean()
                ax.axhline(y=mean_rate, color='#e74c3c', linestyle='--', alpha=0.8,
                          label=f'Mean Rate: {mean_rate:.2f}%')
                
                # Customize ticks
                ax.tick_params(axis='both', labelsize=10)
                
                # Add legend
                ax.legend(fontsize=10)
                
                # Adjust layout
                plt.tight_layout()
                
                return fig
            else:
                return None
        else:
            return None

    def generate_report(self):
        """Generate a CSV report of the safety checks"""
        if os.path.exists(self.safety_check_records):
            data = pd.read_csv(self.safety_check_records)
            if not data.empty:
                # Add summary statistics
                summary = pd.DataFrame({
                    'Metric': ['Total Batches', 'Average Defect Rate', 'Max Defect Rate', 'Min Defect Rate'],
                    'Value': [
                        len(data),
                        f"{data['Defect Rate'].mean():.2f}%",
                        f"{data['Defect Rate'].max():.2f}%",
                        f"{data['Defect Rate'].min():.2f}%"
                    ]
                })
                
                # Create report filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"defect_analysis_report_{timestamp}.csv"
                
                # Save summary and data
                with open(report_filename, 'w') as f:
                    f.write("DEFECT ANALYSIS REPORT\n")
                    f.write(f"Generated on: {datetime.datetime.now()}\n\n")
                    f.write("SUMMARY STATISTICS\n")
                    summary.to_csv(f, index=False)
                    f.write("\nDETAILED DATA\n")
                    data.to_csv(f, index=False)
                
                return report_filename
        return None

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
                # Update batch size for production
                if gesture == "start_production":
                    if not self.emergency_mode:
                        self.batch_size += 1
                        self.safety_check_done = False  # Reset safety check for new batch
                
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
    st.set_page_config(
        page_title="Smart Factory Control",
        page_icon="🏭",
        layout="wide"
    )

    # Authentication check
    if not auth_manager.authenticate():
        return

    # Main application interface
    st.title("Smart Factory Control System")
    
    try:
        # Initialize the camera if not already initialized
        if st.session_state.camera is None:
            st.session_state.camera = cv2.VideoCapture(0)
        
        # Initialize the controller
        controller = SmartFactoryController()
        
        # Create two columns for the interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Live Camera Feed")
            frame_placeholder = st.empty()
            
        with col2:
            # Create containers for dynamic content
            status_container = st.container()
            controls_container = st.container()
            
            with status_container:
                st.header("System Status")
                status_placeholder = st.empty()
            
            with controls_container:
                st.header("Controls")
                st.write("Gesture Controls:")
                st.write("- Fist: Emergency Stop")
                st.write("- Peace Sign: Start Production")
                st.write("- Palm: Quality Check")
                
                # Add Analysis section
                st.header("Analysis")
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    if st.button("Generate Graph"):
                        fig = controller.plot_defect_rate()
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.warning("No data available for visualization")
                
                with analysis_col2:
                    if st.button("Export Report"):
                        report_file = controller.generate_report()
                        if report_file:
                            with open(report_file, 'r') as f:
                                st.download_button(
                                    label="Download Report",
                                    data=f.read(),
                                    file_name=report_file,
                                    mime="text/csv"
                                )
                            st.success(f"Report generated: {report_file}")
                        else:
                            st.warning("No data available for report generation")
                
                if st.button("Emergency Reset"):
                    controller.reset_production()
                    st.success("Emergency mode reset successfully")
        
        while True:
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Failed to grab frame")
                break
                
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            output_frame = controller.process_frame(frame)
            
            # Display the frame
            frame_placeholder.image(output_frame, channels="BGR", use_column_width=True)
            
            # Update status with empty container to prevent glitching
            with status_placeholder.container():
                st.write(f"Machine Status: {controller.machine_status}")
                st.write(f"Production Count: {controller.production_count}")
                st.write(f"Quality Score: {controller.quality_score:.1f}%")
                st.write(f"Defect Count: {controller.defect_count}")
                st.write(f"Batch Size: {controller.batch_size}")
                
                if controller.emergency_mode:
                    st.error("⚠️ EMERGENCY STOP ACTIVATED")
            
            # Update session state
            controller.update_session_state()
            
            # Check for inactivity
            if auth_manager.check_inactivity():
                st.warning("Session expired due to inactivity. Please log in again.")
                break
            
            # Update activity timestamp
            auth_manager.update_activity()
            
            # Add a small delay to prevent high CPU usage
            time.sleep(0.1)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    finally:
        # Release camera resources if initialized
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None

if __name__ == "__main__":
    main() 
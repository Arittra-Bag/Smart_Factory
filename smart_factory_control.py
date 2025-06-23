import cv2
import mediapipe as mp
import numpy as np
import time
from collections import defaultdict
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
# import mysql.connector
import threading
import csv
from tensorflow.keras.models import load_model
import numpy as np
import random


# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="demo123",
#     database="smart_factory",
#     auth_plugin='mysql_native_password',
#     use_pure=True
# )

# cursor = db.cursor()

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

        self.production_mode = False
        self.defect_model = load_model("defect_detector_model.h5")


        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize gesture tracking
        self.gesture_stats = defaultdict(int)
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0
        self.current_gesture = None
        self.fps = 0
        self.frame_count = 0
        self.completed_batches = 0
        self.start_time = time.time()
        self.current_batch_count = 0


        # Load production data from session state
        self.production_count = st.session_state.production_data.get('production_count', 0)
        self.defect_count = st.session_state.production_data.get('defect_count', 0)
        self.batch_size = st.session_state.production_data.get('batch_size', 0)
        self.quality_score = st.session_state.production_data.get('quality_score', 0)
        self.machine_status = st.session_state.production_data.get('machine_status', "STANDBY")
        self.emergency_mode = st.session_state.production_data.get('emergency_mode', False)

        # 🆕 Maintain total production throughout the day (do not reset per batch)
        self.total_production = st.session_state.production_data.get('total_production', self.production_count)

        self.last_inspection_time = time.time()
        self.emergency_reset_start = 0
        self.emergency_reset_duration = 3.0  # 3 seconds for emergency reset

        # Load quality inspection model (simulated)
        self.quality_model = self.load_quality_model()

        # Safety check tracking
        self.last_batch_size = 0
        self.safety_check_done = False
        self.safety_check_records = "safety_check_records.csv"

        # Emergency reset tracking
        self.emergency_reset_progress = 0
        self.emergency_reset_active = False

        # Set a fixed batch size per day
        # self.fixed_batch_size = 10  # You can prompt user input for this if needed

        # Auto-increment timing for production
        # self.last_production_time = time.time()
        # self.production_interval = 2  # seconds


        
        # Ensure CSV file exists
        # if not os.path.exists(self.safety_check_records):
        #     with open(self.safety_check_records, 'w') as f:
        #         f.write("Batch Size,Defect Rate,Timestamp\n")

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.datetime.now()
        
        # Check the last written date from the file
        last_logged_date = None
        if os.path.exists(self.safety_check_records):
            with open(self.safety_check_records, "r") as f:
                for line in reversed(f.readlines()):
                    if line.startswith("----"):
                        last_logged_date = line.strip().replace("---- ", "")
                        break

        # Now write the new batch log
        with open(self.safety_check_records, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write a new date header if needed
            if last_logged_date != today:
                file.write(f"\n---- {today} ----\n")
                writer.writerow(["Batch Size", "Defect Rate", "Timestamp"])

            # Write the actual batch info
            # defect_rate = (self.defect_count / self.production_count) * 100 if self.production_count else 0
            # writer.writerow([
            #     self.production_count,
            #     round(defect_rate, 2),
            #     timestamp
            # ])

        # Initialize alert system
        self.alerts = []
        self.alert_thread = threading.Thread(target=self.alert_monitor, daemon=True)
        self.alert_thread.start()

    def predict_defect(self, image):
        """Use the trained model to predict defects in the image"""
        try:
            # Resize to match model input (128x128 as per training)
            img = cv2.resize(image, (128, 128))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make prediction using the trained model
            prediction = self.defect_model.predict(img, verbose=0)[0][0]
            
            # Convert prediction to defect rate (0-100%)
            defect_rate = prediction * 100
            
            # Determine if defective (threshold can be adjusted)
            is_defective = prediction >= 0.5
            
            return {
                'defect_rate': defect_rate,
                'is_defective': is_defective,
                'prediction': prediction
            }
            
        except Exception as e:
            print(f"❌ Error in defect prediction: {e}")
            # Fallback to random if model fails
            return {
                'defect_rate': random.uniform(0.0, 15.0),
                'is_defective': False,
                'prediction': 0.0
            }


    def update_session_state(self):
        """Update session state with current production data"""
        st.session_state.production_data.update({
            'production_count': self.production_count,
            # 'completed_batches': self.completed_batches,
            'total_production': self.total_production,
            'defect_count': self.defect_count,
            'batch_size': self.batch_size,
            'quality_score': self.quality_score,
            'machine_status': self.machine_status,
            'emergency_mode': self.emergency_mode
        })

    # def save_daily_log(self):
    #     sql = """
    #         INSERT INTO production_logs (
    #             machine_status, production_count, quality_score,
    #             defect_count, batch_size, emergency_stop
    #         ) VALUES (%s, %s, %s, %s, %s, %s)
    #     """

    #     data = (
    #         self.machine_status,
    #         self.production_count,
    #         self.quality_score,
    #         self.defect_count,
    #         self.batch_size,
    #         self.emergency_mode
    #     )

    #     try:
    #         cursor.execute(sql, data)
    #         db.commit()
    #         st.success("Daily production log saved successfully.")
    #     except Exception as e:
    #         st.error(f"Database insert error: {e}")



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

        # Landmarks
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        # Finger state
        is_thumb_up = thumb_tip.y < thumb_ip.y and abs(thumb_tip.x - wrist.x) > 0.1
        is_index_up = index_tip.y < index_pip.y - 0.02
        is_middle_up = middle_tip.y < middle_pip.y - 0.02
        is_ring_up = ring_tip.y < ring_pip.y - 0.02
        is_pinky_up = pinky_tip.y < pinky_pip.y - 0.02

        is_index_down = index_tip.y > index_pip.y + 0.02
        is_middle_down = middle_tip.y > middle_pip.y + 0.02
        is_ring_down = ring_tip.y > ring_pip.y + 0.02
        is_pinky_down = pinky_tip.y > pinky_pip.y + 0.02
        is_thumb_down = thumb_tip.y > thumb_ip.y

        # ✊ Emergency Stop (All fingers down - Fist)
        if is_index_down and is_middle_down and is_ring_down and is_pinky_down:
            return "emergency_stop"

        # ✌️ Peace (only index + middle up, and well spaced) - Start Production
        if (is_index_up and is_middle_up and not is_ring_up and not is_pinky_up and not is_thumb_up):
            spread = abs(index_tip.x - middle_tip.x)
            if spread > 0.05:
                return "start_production"

        # ✋ Palm (all five fingers up) - Quality Check
        if is_index_up and is_middle_up and is_ring_up and is_pinky_up and is_thumb_up:
            return "quality_check"

        # 📊 Plot defect rate (only index up)
        if is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
            return "plot_defect_rate"

        # 👍 Thumbs up only if others are clearly down
        if is_thumb_up and not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
            return "thumbs_up"

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
                
        elif gesture == "quality_check":
            if self.emergency_mode and self.emergency_reset_active:
                # Show emergency reset progress
                progress_text = f"Emergency Reset: {self.emergency_reset_progress:.0f}%"
                cv2.putText(frame, progress_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw progress bar
                bar_width = 300
                bar_height = 20
                bar_x = 10
                bar_y = 50
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (100, 100, 100), -1)
                
                # Progress bar
                progress_width = int((self.emergency_reset_progress / 100) * bar_width)
                progress_color = (0, 255, 255) if self.emergency_reset_progress < 100 else (0, 255, 0)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                             progress_color, -1)
                
                # Add pulsing effect
                pulse_intensity = int(127 + 127 * np.sin(time.time() * 5))
                cv2.circle(frame, (bar_x + bar_width + 20, bar_y + bar_height//2), 8, 
                          (0, pulse_intensity, pulse_intensity), -1)
                
            elif not self.emergency_mode:
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

                        # Note: Actual defect detection is now handled in process_frame method

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
            y_pos += 15
            cv2.putText(frame, "Hold PALM for 3s to reset", (int(x_pos), int(y_pos)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        elif self.emergency_mode:
            y_pos += 20
            cv2.putText(frame, "Show PALM for 3s to reset", (int(x_pos), int(y_pos)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
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
            if self.production_mode:
                cv2.putText(frame, "Current Hand: NONE (Production Running)", (int(x_pos), int(y_pos)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Current Hand: NONE", (int(x_pos), int(y_pos)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        # Add production mode indicator
        y_pos += line_spacing
        if self.production_mode:
            cv2.putText(frame, "MODE: CONTINUOUS PRODUCTION", (int(x_pos), int(y_pos)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif self.emergency_mode:
            cv2.putText(frame, "MODE: EMERGENCY STOP", (int(x_pos), int(y_pos)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(frame, "MODE: STANDBY", (int(x_pos), int(y_pos)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return frame

    def plot_defect_rate(self):
        """Plot batch size vs. defect rate from the database."""
        try:
            query = "SELECT batch_size, defect_rate FROM safetyfi_logs"
            cursor.execute(query)
            result = cursor.fetchall()

            if result:
                # Convert result to DataFrame
                data = pd.DataFrame(result, columns=["Batch Size", "Defect Rate"])

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data["Batch Size"], data["Defect Rate"],
                        marker='o', linewidth=2, markersize=8,
                        color='#2ecc71', markeredgecolor='white',
                        markeredgewidth=2)

                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xlabel("Batch Size", fontsize=12, fontweight='bold')
                ax.set_ylabel("Defect Rate (%)", fontsize=12, fontweight='bold')
                ax.set_title("Batch Size vs. Defect Rate Analysis",
                            fontsize=14, fontweight='bold', pad=20)

                mean_rate = data["Defect Rate"].mean()
                ax.axhline(y=mean_rate, color='#e74c3c', linestyle='--', alpha=0.8,
                        label=f'Mean Rate: {mean_rate:.2f}%')

                ax.tick_params(axis='both', labelsize=10)
                ax.legend(fontsize=10)
                plt.tight_layout()

                return fig
            else:
                return None
        except Exception as e:
            st.error(f"Error fetching data from DB: {e}")
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

    def log_safety_check(self, batch_size, defect_rate, timestamp):
        """Log safety check data with proper date organization"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        header = f"---- {today} ----\n"
        columns = "Batch Size,Defect Rate,Timestamp\n"
        entry = f"{batch_size},{defect_rate:.2f},{timestamp}\n"

        # Read existing content
        if os.path.exists(self.safety_check_records):
            with open(self.safety_check_records, "r") as f:
                lines = f.readlines()
        else:
            lines = []

        # Find today's header
        today_section_start = -1
        today_section_end = len(lines)
        
        for i, line in enumerate(lines):
            if line.strip() == f"---- {today} ----":
                today_section_start = i
                # Find where this section ends (next date header or end of file)
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("----"):
                        today_section_end = j
                        break
                break

        if today_section_start == -1:
            # Today's header not found, append at end
            if lines and not lines[-1].endswith('\n'):
                lines.append('\n')
            lines.append(header)
            lines.append(columns)
            lines.append(entry)
        else:
            # Today's section exists, insert entry after the columns line
            insert_pos = today_section_start + 2  # After header and columns
            lines.insert(insert_pos, entry)

        # Write back
        with open(self.safety_check_records, "w") as f:
            f.writelines(lines)
        
        print(f"📊 Logged: Batch={batch_size}, Defect Rate={defect_rate:.2f}%, Time={timestamp}")

    def process_frame(self, frame):
        # Calculate FPS
        self.calculate_fps()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Create output frame
        output_frame = frame.copy()
        
        # Reset current gesture if no hands detected, but DON'T change production state
        if not results.multi_hand_landmarks:
            self.current_gesture = None
            self.emergency_reset_start = 0  # Reset emergency reset timer
            self.emergency_reset_active = False  # Reset emergency reset tracking
            self.emergency_reset_progress = 0  # Reset progress
            # REMOVED: Automatic reset to STANDBY - production continues until explicitly stopped
        
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
                print(f"Detected Gesture: {gesture}")

                # Emergency stop - ALWAYS stops production regardless of current state
                if gesture == "emergency_stop":
                    self.machine_status = "EMERGENCY"
                    self.emergency_mode = True
                    self.production_mode = False
                    self.emergency_reset_active = False
                    self.emergency_reset_start = 0
                    print("🛑 EMERGENCY STOP ACTIVATED")

                # Emergency reset with palm gesture (10 seconds hold)
                elif gesture == "quality_check" and self.emergency_mode:
                    if not self.emergency_reset_active:
                        self.emergency_reset_active = True
                        self.emergency_reset_start = time.time()
                        print("🔄 Emergency reset initiated - hold palm for 10 seconds")
                    
                    # Calculate progress
                    elapsed_time = time.time() - self.emergency_reset_start
                    self.emergency_reset_progress = min((elapsed_time / self.emergency_reset_duration) * 100, 100)
                    
                    # Check if reset duration completed
                    if elapsed_time >= self.emergency_reset_duration:
                        print("✅ Emergency reset completed!")
                        
                        # Log final batch metrics before reset
                        if self.batch_size > 0:
                            # Run final defect detection on current frame
                            final_defect_result = self.predict_defect(output_frame)
                            self.log_safety_check(
                                self.batch_size, 
                                final_defect_result['defect_rate'], 
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                            print(f"📊 Final batch logged: Size={self.batch_size}, Defect Rate={final_defect_result['defect_rate']:.2f}%, Defective={final_defect_result['is_defective']}")
                        
                        self.perform_emergency_reset()

                # Start production - only if not in emergency mode
                elif gesture == "start_production":
                    if not self.emergency_mode:
                        self.machine_status = "RUNNING"
                        self.production_mode = True
                        self.safety_check_done = False
                        self.batch_size = self.production_count  # Initialize batch size to current production count
                        print("✅ Production STARTED")

                # Quality check - stops production and performs quality check (only if not in emergency mode)
                elif gesture == "quality_check" and not self.emergency_mode:
                    self.machine_status = "QUALITY CHECK"
                    self.production_mode = False
                    print("🔍 Quality Check initiated")

                    # Ensure batch size is set to current production count
                    if self.batch_size == 0:
                        self.batch_size = self.production_count
                    
                    # Add visual feedback for defect detection
                    cv2.putText(output_frame, "ANALYZING FOR DEFECTS...", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Run actual defect detection on current frame
                    defect_result = self.predict_defect(output_frame)
                    print(f"🧪 Quality Check Result: Defect Rate={defect_result['defect_rate']:.2f}%, Defective={defect_result['is_defective']}")
                    
                    # Show result on frame
                    result_text = f"DEFECT RATE: {defect_result['defect_rate']:.1f}%"
                    result_color = (0, 0, 255) if defect_result['is_defective'] else (0, 255, 0)
                    cv2.putText(output_frame, result_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
                    
                    # Log it (only if batch size is valid)
                    if self.batch_size > 0:
                        self.log_safety_check(
                            self.batch_size, 
                            defect_result['defect_rate'], 
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        
                        # Update defect count based on actual detection
                        if defect_result['is_defective']:
                            self.defect_count += 1
                            print(f"⚠️ Defect detected! Total defects: {self.defect_count}")
                    else:
                        print("⚠️ Cannot log: Batch size is 0")

                # Update stats with cooldown for all other gestures
                if time.time() - self.last_gesture_time > self.gesture_cooldown and gesture != "start_production":
                    self.gesture_stats[gesture] += 1
                    self.last_gesture_time = time.time()

                output_frame = self.apply_industrial_effect(output_frame, gesture)

        # Auto-increment production while in production mode (continuous production)
        if self.production_mode and not self.emergency_mode:
            current_time = time.time()
            if not hasattr(self, 'last_production_time'):
                self.last_production_time = current_time
            
            # Increment production every 2 seconds while in production mode
            if current_time - self.last_production_time >= 2.0:
                self.production_count += 1
                self.total_production += 1
                self.batch_size = self.production_count  # Keep batch size synchronized with production count
                self.last_production_time = current_time
                print(f"🏭 Producing... Count: {self.production_count}")
            
            # Add continuous production indicator on frame
            cv2.putText(output_frame, "PRODUCTION RUNNING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Add pulsing green dot indicator
            pulse_intensity = int(127 + 127 * np.sin(time.time() * 3))
            cv2.circle(output_frame, (250, 25), 8, (0, pulse_intensity, 0), -1)

        # Draw industrial HUD
        output_frame = self.draw_industrial_hud(output_frame)
            
        return output_frame

    def calculate_fps(self):
        self.frame_count += 1
        if time.time() - self.start_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()

    def perform_emergency_reset(self):
        """Perform emergency reset: log metrics to CSV and reset all production data"""
        try:
            # Calculate final metrics before reset
            defect_rate = (self.defect_count / max(self.production_count, 1)) * 100
            total_production = self.total_production
            
            # Log emergency reset to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                'timestamp': timestamp,
                'event': 'EMERGENCY_RESET',
                'total_production': total_production,
                'production_count': self.production_count,
                'defect_count': self.defect_count,
                'defect_rate': round(defect_rate, 2),
                'quality_score': round(self.quality_score, 1),
                'batch_size': self.batch_size,
                'machine_status': self.machine_status
            }
            
            # Write to CSV
            csv_file = "emergency_reset_log.csv"
            file_exists = os.path.exists(csv_file)
            
            with open(csv_file, mode='a', newline='') as f:
                fieldnames = ['timestamp', 'event', 'total_production', 'production_count', 
                             'defect_count', 'defect_rate', 'quality_score', 'batch_size', 'machine_status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(log_entry)
            
            print(f"📊 Emergency reset logged to {csv_file}")
            print(f"📈 Final metrics - Production: {total_production}, Defects: {self.defect_count}, Rate: {defect_rate:.2f}%")
            
            # Reset all production metrics
            self.production_count = 0
            self.defect_count = 0
            self.batch_size = 0
            self.quality_score = 100.0
            self.machine_status = "STANDBY"
            self.emergency_mode = False
            self.production_mode = False
            self.total_production = 0
            self.safety_check_done = False
            self.last_batch_size = 0
            
            # Reset emergency reset tracking
            self.emergency_reset_active = False
            self.emergency_reset_start = 0
            self.emergency_reset_progress = 0
            
            # Update session state
            self.update_session_state()
            
            print("🔄 All production metrics reset successfully")
            
        except Exception as e:
            print(f"❌ Error during emergency reset: {e}")
            # Still reset the system even if logging fails
            self.emergency_mode = False
            self.machine_status = "STANDBY"
            self.production_mode = False

    def test_defect_model(self):
        """Test the defect detection model with a sample image"""
        try:
            # Create a test image (you can replace this with actual test images)
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            print("🧪 Testing defect detection model...")
            result = self.predict_defect(test_image)
            
            print(f"✅ Model test successful!")
            print(f"   - Defect Rate: {result['defect_rate']:.2f}%")
            print(f"   - Is Defective: {result['is_defective']}")
            print(f"   - Raw Prediction: {result['prediction']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False

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
                
                # Add model testing section
                st.header("Model Testing")
                if st.button("Test Defect Model"):
                    if controller.test_defect_model():
                        st.success("✅ Defect detection model is working correctly!")
                    else:
                        st.error("❌ Defect detection model test failed!")
                
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
            frame_placeholder.image(output_frame, channels="BGR", use_container_width=True)
            
            # Update status with empty container to prevent glitching
            with status_placeholder.container():
                # st.write(f"Fixed Batch Size: {controller.fixed_batch_size}")
                # st.write(f"Completed Batches: {controller.completed_batches}")
                st.write(f"Machine Status: {controller.machine_status}")
                st.write(f"Production Count: {controller.production_count}")
                st.write(f"Quality Score: {controller.quality_score:.1f}%")
                st.write(f"Defect Count: {controller.defect_count}")
                st.write(f"Batch Size: {controller.batch_size}")

                # st.write(f"Production Count: {controller.production_count}")
                
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
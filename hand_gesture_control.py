import cv2
import mediapipe as mp
import numpy as np
import time
from collections import defaultdict

class HandGestureController:
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
        
        # Initialize gesture tracking
        self.gesture_stats = defaultdict(int)
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.0  # seconds
        self.current_gesture = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.active_effect = None  # Track current active effect
        self.effect_start_time = 0  # Track when effect started
        self.effect_duration = 2.0  # Duration to maintain effect in seconds

        self.production_started = False  # Track production status
        self.last_action_time = 0
        self.action_cooldown = 1.5  # Seconds between gesture activations



    def calculate_fps(self):
        self.frame_count += 1
        if time.time() - self.start_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()

    def detect_gestures(self, hand_landmarks):
        """Detect various hand gestures"""
        if not hand_landmarks:
            return None

        # Get finger landmarks
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        # Check for different gestures
        fingers_closed = (
            index_tip.y > index_pip.y and
            middle_tip.y > middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y
        )

        # Peace sign (index and middle fingers up)
        peace_sign = (
            index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y > ring_pip.y + 0.02 and  # Clearly bent
            pinky_tip.y > pinky_pip.y + 0.02 and
            abs(index_tip.x - middle_tip.x) > 0.03  # Fingers spread apart
        )


        # Thumbs up (thumb up, other fingers closed)
        thumbs_up = (
            thumb_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y and
            all([
                index_tip.y > index_pip.y + 0.02,
                middle_tip.y > middle_pip.y + 0.02,
                ring_tip.y > ring_pip.y + 0.02,
                pinky_tip.y > pinky_pip.y + 0.02
            ]) and
            abs(thumb_tip.x - wrist.x) > 0.1 and  # Ensure thumb is extended sideways
            abs(thumb_tip.y - wrist.y) > 0.1      # Ensure it’s "up"
        )


        # Palm open (all fingers extended)
        palm_open = (
            index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y < ring_pip.y and
            pinky_tip.y < pinky_pip.y
        )

        if fingers_closed:
            return "fist"
        elif peace_sign:
            return "peace"
        elif thumbs_up:
            return "thumbs_up"
        elif palm_open:
            return "palm"
        return None

    def apply_gesture_effect(self, frame, gesture):
        """Apply visual effects based on the detected gesture"""
        if gesture == "fist":
            # Grey overlay effect
            overlay = frame.copy()
            grey_color = (128, 128, 128)
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), grey_color, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        elif gesture == "peace":
            # Rainbow effect
            overlay = frame.copy()
            rainbow = np.zeros_like(frame)
            rainbow[:, :, 0] = np.sin(np.linspace(0, 2*np.pi, frame.shape[1])) * 127 + 128
            rainbow[:, :, 1] = np.sin(np.linspace(0, 2*np.pi, frame.shape[1]) + 2*np.pi/3) * 127 + 128
            rainbow[:, :, 2] = np.sin(np.linspace(0, 2*np.pi, frame.shape[1]) + 4*np.pi/3) * 127 + 128
            cv2.addWeighted(rainbow, 0.3, frame, 0.7, 0, frame)
        elif gesture == "thumbs_up":
            # Glow effect
            overlay = frame.copy()
            glow = cv2.GaussianBlur(frame, (21, 21), 0)
            cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)
        elif gesture == "palm":
            # No special effect for palm, just update status
            pass
        return frame

    def draw_stats(self, frame):
        """Draw statistics and information on the frame"""
        # Create a black panel on the right side for the legend
        panel_width = 200
        original_frame = frame.copy()
        frame = cv2.copyMakeBorder(
            original_frame,
            0, 0, 0, panel_width,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        # Starting position for the legend
        x_pos = int(frame.shape[1] - panel_width + 10)
        y_pos = 40  # Increased initial y position
        line_spacing = 35  # Increased line spacing
        
        # Draw title with smaller font and proper positioning
        cv2.putText(frame, "GESTURE", (int(x_pos), int(y_pos)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        cv2.putText(frame, "CONTROLS", (int(x_pos), int(y_pos)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += int(line_spacing * 1.2)

        # Draw FPS with background
        fps_text = f"FPS: {self.fps}"
        cv2.putText(frame, fps_text, (int(x_pos), int(y_pos)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += int(line_spacing * 1.2)

        # Draw divider
        cv2.line(frame, 
                 (int(x_pos - 5), int(y_pos - line_spacing//2)),
                 (int(frame.shape[1] - 10), int(y_pos - line_spacing//2)),
                 (200, 200, 200), 1)

        # Draw gesture counts with better icons
        for gesture, count in self.gesture_stats.items():
            if gesture == "fist":
                text = f"Fist: {count}"
            elif gesture == "peace":
                text = f"Peace: {count}"
            elif gesture == "thumbs_up":
                text = f"Thumbs: {count}"
            else:
                text = f"Palm: {count}"
            
            cv2.putText(frame, text, (int(x_pos), int(y_pos)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += line_spacing

        # Draw divider
        cv2.line(frame, 
                 (int(x_pos - 5), int(y_pos)),
                 (int(frame.shape[1] - 10), int(y_pos)),
                 (200, 200, 200), 1)
        y_pos += line_spacing

        # Draw current gesture with highlighted background
        if self.current_gesture:
            # Draw background rectangle for current gesture
            text = f"Current: {self.current_gesture}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, 
                         (int(x_pos - 5), int(y_pos - text_height - 5)),
                         (int(x_pos + text_width + 5), int(y_pos + 5)),
                         (0, 100, 100), -1)
            cv2.putText(frame, text, (int(x_pos), int(y_pos)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_pos += int(line_spacing * 1.5)

        # Draw warning for multiple hands at the bottom
        if self.multiple_hands_detected:
            warning_y = frame.shape[0] - 40
            cv2.putText(frame, "Multiple hands!", (int(x_pos), int(warning_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
        
        # Check for multiple hands
        self.multiple_hands_detected = results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                output_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            gesture = self.detect_gestures(hand_landmarks)

            # Only trigger action when the gesture changes
        if gesture:
            current_time = time.time()

            if gesture != getattr(self, "last_detected_gesture", None):
                self.last_detected_gesture = gesture
                self.last_gesture_time = current_time

                print(f"Detected Gesture: {gesture}")

                # ✌️ Start production if peace and not already running
                if gesture == "start_production" and not self.production_started:
                    self.production_started = True
                    print("✅ Production started")

                # ✊ Emergency stop
                elif gesture == "emergency_stop" and self.production_started:
                    self.production_started = False
                    print("🛑 Emergency stop")

                # ✋ Quality check (same as stop for now)
                elif gesture == "quality_check" and self.production_started:
                    self.production_started = False
                    print("🔍 Quality check triggered")

                # Count gesture
                self.gesture_stats[gesture] += 1

            output_frame = self.apply_gesture_effect(output_frame, gesture)

        elif not results.multi_hand_landmarks:
            self.last_detected_gesture = None
            self.current_gesture = None


        # Draw overlay stats
        output_frame = self.draw_stats(output_frame)
        return output_frame


    # def process_frame(self, frame):
    #     # Calculate FPS
    #     self.calculate_fps()
        
    #     # Convert BGR to RGB
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # Process the frame
    #     results = self.hands.process(rgb_frame)
        
    #     # Create output frame
    #     output_frame = frame.copy()
        
    #     # Check for multiple hands
    #     self.multiple_hands_detected = results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1
        
    #     if results.multi_hand_landmarks:
    #         # Only process the first hand
    #         hand_landmarks = results.multi_hand_landmarks[0]
            
    #         # Draw hand landmarks
    #         self.mp_draw.draw_landmarks(
    #             output_frame,
    #             hand_landmarks,
    #             self.mp_hands.HAND_CONNECTIONS,
    #             self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
    #             self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
    #         )
            
    #         gesture = self.detect_gestures(hand_landmarks)

    #         # Update gesture statistics if cooldown passed
    #         if gesture and (time.time() - self.last_gesture_time > self.gesture_cooldown):
    #             self.gesture_stats[gesture] += 1
    #             self.last_gesture_time = time.time()
    #             self.current_gesture = gesture

    #             # 🟢 Start production with peace sign
    #             if gesture == "peace" and not self.production_started:
    #                 self.production_started = True
    #                 print("✅ Production started (flag turned ON)")

    #             # 🛑 Stop/reset production with palm
    #             elif gesture == "palm" and self.production_started:
    #                 self.production_started = False
    #                 print("🛑 Production stopped (flag turned OFF)")

    #         # Always apply effect for feedback
    #         output_frame = self.apply_gesture_effect(output_frame, self.current_gesture)

            
        #     # Detect gestures
        #     gesture = self.detect_gestures(hand_landmarks)
            
        #     # Update gesture statistics and effects
        #     if gesture:
        #         # Update current gesture and stats if changed
        #         if gesture != self.current_gesture:
        #             self.current_gesture = gesture
        #             if time.time() - self.last_gesture_time > self.gesture_cooldown:
        #                 self.gesture_stats[gesture] += 1
        #                 self.last_gesture_time = time.time()
                
        #         # Always apply the current gesture's effect
        #         output_frame = self.apply_gesture_effect(output_frame, gesture)
        #     else:
        #         self.current_gesture = None
        # else:
        #     self.current_gesture = None
            
        # # Draw statistics
        # output_frame = self.draw_stats(output_frame)
            
        # return output_frame

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Initialize the hand gesture controller
    controller = HandGestureController()
    
    print("Enhanced Hand Gesture Control Started")
    print("Available gestures:")
    print("- Close fist: Grey overlay")
    print("- Peace sign: Rainbow effect")
    print("- Thumbs up: Glow effect")
    print("- Open palm: Status update")
    print("Press 'q' to quit")
    
    # Set target FPS
    target_fps = 10
    frame_delay = int(1000 / target_fps)  # Convert FPS to milliseconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        output_frame = controller.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Enhanced Hand Gesture Control', output_frame)
        
        # Add delay to maintain 5 FPS
        key = cv2.waitKey(frame_delay)
        
        # Break the loop if 'q' is pressed
        if key & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
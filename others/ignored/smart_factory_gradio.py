import gradio as gr
import cv2
import numpy as np
from smart_factory_control import SmartFactoryController
import time

class GradioFactoryInterface:
    def __init__(self):
        self.controller = SmartFactoryController()
        self.cap = None
        self.is_running = False

    def initialize_camera(self):
        """Initialize the camera if not already initialized"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        return self.cap.isOpened()

    def release_camera(self):
        """Release the camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def process_video(self):
        """Generator function for video processing"""
        self.initialize_camera()
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process frame through our controller
            output_frame = self.controller.process_frame(frame)
            
            # Convert BGR to RGB for Gradio
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            
            # Get current metrics for display
            yield (
                output_frame,  # video feed
                self.controller.machine_status,  # status
                self.controller.production_count,  # production count
                f"{self.controller.quality_score:.1f}%",  # quality score
                self.controller.defect_count,  # defects
                self.controller.current_gesture or "NONE"  # current gesture
            )

    def stop_video(self):
        """Stop video processing"""
        self.is_running = False
        self.release_camera()
        return None, None, None, None, None, None

def create_interface():
    interface = GradioFactoryInterface()
    
    with gr.Blocks(title="Smart Factory Control System", theme=gr.themes.Base()) as demo:
        gr.Markdown("# Smart Factory Control System")
        gr.Markdown("""
        ### Gesture Controls:
        - 👊 Fist: Emergency Stop
        - ✌️ Peace: Start Production
        - ✋ Palm: Quality Check (Hold for 3s to reset Emergency)
        """)
        
        with gr.Row():
            # Video feed on the left
            video_output = gr.Image(label="Factory Control Feed", width=640, height=480)
            
            # Metrics panel on the right
            with gr.Column():
                status = gr.Textbox(label="System Status", interactive=False)
                production = gr.Number(label="Production Count", interactive=False)
                quality = gr.Textbox(label="Quality Score", interactive=False)
                defects = gr.Number(label="Defect Count", interactive=False)
                gesture = gr.Textbox(label="Current Gesture", interactive=False)

        with gr.Row():
            start_btn = gr.Button("Start System", variant="primary")
            stop_btn = gr.Button("Stop System", variant="secondary")

        # Connect buttons to actions
        outputs = [video_output, status, production, quality, defects, gesture]
        
        start_btn.click(
            fn=interface.process_video,
            outputs=outputs,
            api_name="start"
        )
        
        stop_btn.click(
            fn=interface.stop_video,
            outputs=outputs,
            api_name="stop"
        )

        # Add closing callback
        demo.close = interface.release_camera

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.queue().launch(server_port=7860, share=False) 

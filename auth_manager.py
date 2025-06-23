import pyotp
import qrcode
import os
from datetime import datetime, timedelta
import streamlit as st
from dotenv import load_dotenv
import time

class AuthManager:
    def __init__(self):
        load_dotenv()
        # Initialize session state if not exists
        if 'is_authenticated' not in st.session_state:
            st.session_state.is_authenticated = False
        if 'last_activity_time' not in st.session_state:
            st.session_state.last_activity_time = time.time()
            
        self.secret_key = os.getenv('OTP_SECRET_KEY', pyotp.random_base32())
        self.totp = pyotp.TOTP(self.secret_key)
        self.inactivity_timeout = 120  # 2 minutes in seconds

    def get_remaining_time(self):
        """Get remaining time before auto-logout"""
        elapsed_time = time.time() - st.session_state.last_activity_time
        remaining_time = max(0, self.inactivity_timeout - elapsed_time)
        return int(remaining_time)

    def format_remaining_time(self, seconds):
        """Format remaining time as MM:SS"""
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def generate_qr_code(self):
        """Generate QR code for Google Authenticator setup"""
        provisioning_uri = self.totp.provisioning_uri(
            name="Smart Factory Control",
            issuer_name="Factory Management"
        )
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img.save("google_auth_qr.png")
        return img

    def verify_otp(self, otp):
        """Verify the provided OTP"""
        return self.totp.verify(otp)

    def update_activity(self):
        """Update the last activity timestamp"""
        st.session_state.last_activity_time = time.time()

    def check_inactivity(self):
        """Check if user has been inactive for too long"""
        if time.time() - st.session_state.last_activity_time > self.inactivity_timeout:
            st.session_state.is_authenticated = False
            return True
        return False

    def authenticate(self):
        """Handle the authentication process"""
        # st.title("Smart Factory Control - Authentication")
        
        # if not st.session_state.is_authenticated:
        #     # First-time setup
        #     if not os.path.exists("google_auth_qr.png"):
        #         st.write("First-time setup: Please scan this QR code with Google Authenticator")
        #         qr_img = self.generate_qr_code()
        #         st.image("google_auth_qr.png", caption="Scan with Google Authenticator")
            
        #     # OTP input
        #     otp = st.text_input("Enter OTP from Google Authenticator", type="password")
            
        #     if st.button("Authenticate"):
        #         if self.verify_otp(otp):
        #             st.session_state.is_authenticated = True
        #             self.update_activity()
        #             st.success("Authentication successful!")
        #             st.rerun()
        #         else:
        #             st.error("Invalid OTP. Please try again.")
            
        #     st.info("Note: You will be automatically logged out after 2 minutes of inactivity.")
        # else:
        #     # Create a container for the timer in the sidebar
        #     with st.sidebar:
        #         st.markdown("### Session Timer")
        #         remaining_time = self.get_remaining_time()
        #         time_str = self.format_remaining_time(remaining_time)
                
        #         # Display timer with color based on remaining time
        #         if remaining_time > 30:
        #             st.info(f"⏱️ Auto-logout in: {time_str}")
        #         elif remaining_time > 10:
        #             st.warning(f"⏱️ Auto-logout in: {time_str}")
        #         else:
        #             st.error(f"⏱️ Auto-logout in: {time_str}")
            
        #     # Show logout button
        #     if st.sidebar.button("Logout"):
        #         st.session_state.is_authenticated = False
        #         st.rerun()
            
        #     # Check for inactivity
        #     if self.check_inactivity():
        #         st.warning("Session expired due to inactivity. Please log in again.")
        #         st.rerun()
            
        #     self.update_activity()
        #     return True
        
        return True 
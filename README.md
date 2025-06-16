# Smart Factory Control System

A computer vision-based industrial control system that uses hand gesture recognition for contactless factory operations management. The system integrates real-time quality inspection, production monitoring, and safety controls through an intuitive gesture interface.

## System Architecture

### Core Components

1. **Smart Factory Controller** (`smart_factory_control.py`)
   - Main application orchestrator
   - Streamlit-based web interface
   - Real-time camera processing and gesture recognition
   - Production metrics tracking and quality control

2. **Authentication Manager** (`auth_manager.py`)
   - Time-based One-Time Password (TOTP) authentication
   - Google Authenticator integration
   - Session management with auto-logout
   - QR code generation for initial setup

3. **Hand Gesture Controller** (`hand_gesture_control.py`)
   - MediaPipe-based hand landmark detection
   - Multi-gesture recognition system
   - Visual effects and feedback processing

## Technical Implementation

### Quality Inspection System

#### Safety Check Protocol
- **Minimum Batch Requirement**: 10 products before quality inspection
- **Cooldown Period**: 2-second interval between inspections
- **Defect Threshold**: 90% quality score threshold
- **Data Persistence**: CSV logging of all safety checks

### Production Monitoring

#### Real-Time Metrics Tracking
- **Production Count**: Incremental counter with gesture-based control
- **Batch Size**: Synchronized with production count
- **Quality Score**: AI-derived quality assessment (0-100%)
- **Defect Count**: Automatic defect detection and counting
- **Machine Status**: State machine with four states:
  - `STANDBY`: Default idle state
  - `RUNNING`: Active production mode
  - `QUALITY CHECK`: Inspection in progress
  - `EMERGENCY`: Emergency stop activated

### Security and Authentication

#### TOTP Implementation
- **Algorithm**: Time-based One-Time Password (RFC 6238)
- **Secret Generation**: `pyotp.random_base32()`
- **Verification Window**: 30-second time steps
- **QR Code**: Automatic generation for Google Authenticator setup

#### Session Management
- **Inactivity Timeout**: 120 seconds (2 minutes) (proposed)
- **Activity Tracking**: Automatic timestamp updates
- **Auto-logout**: Forced logout on inactivity (proposed)
- **Session Persistence**: Streamlit session state integration

### Data Analytics and Reporting

#### Defect Rate Analysis
- **Data Storage**: CSV format with timestamp logging
- **Visualization**: Matplotlib-based plotting
- **Metrics Calculated**:
  - Batch size vs. defect rate correlation
  - Mean defect rate with trend line
  - Statistical summary (min, max, average)

#### Report Generation
- **Format**: CSV with embedded summary statistics
- **Filename Convention**: `defect_analysis_report_YYYYMMDD_HHMMSS.csv`
- **Content Structure**:
  1. Report metadata and timestamp
  2. Summary statistics table
  3. Detailed raw data

### User Interface

#### Streamlit Web Application
- **Layout**: Wide layout with two-column design
- **Real-time Updates**: Live camera feed with processed overlay
- **Interactive Controls**: Button-based manual controls
- **Status Dashboard**: Real-time metrics display
- **Download Functionality**: Report export with download button

#### Industrial HUD Overlay
- **Panel Design**: 250px right-side panel with black background
- **Information Display**:
  - System status with color coding
  - Production metrics
  - Gesture control guide
  - Current hand gesture indicator
  - Emergency reset progress bar

## Installation and Setup

### Prerequisites
- Python 3.8+
- Webcam or USB camera
- Google Authenticator mobile app

### Dependencies Installation
```bash
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file for environment variables:
```
OTP_SECRET_KEY = JBSWY3DPEHPK3PXP
```

## Usage Instructions

### Initial Setup
1. **Launch Application**:
   ```bash
   streamlit run smart_factory_control.py
   ```

2. **Authentication Setup**:
   - Scan the generated QR code with Google Authenticator
   - Enter the 6-digit TOTP code
   - System will authenticate and start the main interface

### Gesture Controls
- **Emergency Stop**: Make a fist to immediately halt all operations
- **Start Production**: Show peace sign to begin/continue production
- **Quality Check**: Open palm to trigger quality inspection
- **Data Visualization**: Point with index finger to generate defect rate plots

### Safety Protocols
- **Emergency Reset**: Hold palm gesture for 3 seconds during emergency mode
- **Batch Safety Check**: Minimum 10 products required before quality inspection
- **Session Timeout**: Automatic logout after 2 minutes of inactivity (proposed)

### Data Export
- **Generate Graph**: Click button to create defect rate visualization
- **Export Report**: Generate and download comprehensive CSV report

## System Performance

### Processing Specifications
- **Frame Rate**: Real-time processing at camera FPS
- **Gesture Detection Latency**: <100ms
- **Quality Inspection Time**: ~2 seconds per check
- **Memory Usage**: Optimized for continuous operation

## Error Handling and Troubleshooting

### Common Issues
1. **Camera Access**: Ensure camera permissions are granted
2. **Authentication Failures**: Verify system time synchronization
3. **Gesture Recognition**: Ensure adequate lighting and hand visibility
4. **Session Timeouts**: Regular interaction required to maintain session

### Debug Mode
Enable Streamlit debug mode for development:
```bash
streamlit run smart_factory_control.py --logger.level=debug
```

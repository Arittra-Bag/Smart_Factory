# AI Co-Pilot Feature

## 🤖 Overview

The AI Co-Pilot is an intelligent factory monitoring and analysis system that provides real-time insights into factory operations using Google Gemini AI. It combines live data monitoring, simulation control, and AI-powered chat analysis.

## 🚀 Features

### 📊 Live Dashboard
- **Real-time Data Display**: Shows live factory metrics every second
- **Visual Status Indicators**: Color-coded alerts for temperature, vibration, and defect rates
- **Threshold Monitoring**: Automatic warning and critical level indicators
- **Factory State Tracking**: Current operational state display

### 🎛️ Simulator Control
- **State Management**: Control factory simulation states (Normal, Overheating, Belt_Slipping)
- **Real-time Control**: Instant state changes via UI buttons
- **Visual Feedback**: Active state highlighting and confirmation

### 💬 AI Chat Interface
- **Intelligent Analysis**: AI-powered factory data interpretation
- **Context-Aware Responses**: Analysis based on last 60 seconds of data
- **Conversation History**: Persistent chat with timestamps
- **Quick Questions**: Pre-defined question buttons for common queries

## 🛠️ Setup Instructions

### 1. Environment Configuration
Add your Google Gemini API key to the `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Start the Backend
```bash
python app.py
```

### 3. Start the Simulator
```bash
python simulator.py
```

### 4. Start the Frontend
```bash
cd SFC_UI
npm run dev
```

### 5. Access AI Co-Pilot
Navigate to the AI Co-Pilot page via:
- Home page → "AI Co-Pilot" button
- Header navigation → "AI Co-Pilot" tab

## 📡 API Endpoints

### Simulation Control
- `POST /api/simulation/state` - Change simulator state
- `GET /api/simulation/live_data` - Get latest factory data

### AI Co-Pilot
- `POST /api/copilot/chat` - Send query to AI for analysis

## 🎯 Usage Guide

### Live Dashboard
1. **Monitor Metrics**: Watch real-time temperature, vibration, and defect rates
2. **Status Colors**: 
   - 🟢 Green: Normal operation
   - 🟡 Yellow: Warning levels
   - 🔴 Red: Critical levels
3. **Factory State**: Current operational mode display

### Simulator Control
1. **Normal Mode**: Stable operation with minor fluctuations
2. **Overheating Mode**: Gradually increasing temperature and defect rates
3. **Belt Slipping Mode**: Increased vibration and defect rates

### AI Chat
1. **Ask Questions**: Type questions about factory data
2. **Quick Questions**: Use pre-defined question buttons
3. **Get Insights**: Receive AI-powered analysis and recommendations
4. **View History**: Scroll through conversation history

## 🔧 Technical Details

### Data Flow
1. **Simulator** → Generates factory data → `live_data.log`
2. **Flask API** → Reads data → Provides to frontend
3. **AI Co-Pilot** → Analyzes data → Provides insights
4. **Frontend** → Displays data → User interaction

### Thresholds
- **Temperature**: Warning: 65°C, Critical: 75°C
- **Vibration**: Warning: 25, Critical: 30
- **Defect Rate**: Warning: 2.0%, Critical: 3.0%

### AI Analysis Context
- **Data Window**: Last 60 seconds of factory data
- **Analysis Focus**: Trends, anomalies, recommendations
- **Response Format**: Professional, actionable insights

## 🧪 Testing

Run the test suite to verify all endpoints:
```bash
python test_copilot_endpoints.py
```

## 🐛 Troubleshooting

### Common Issues
1. **No Live Data**: Ensure simulator is running
2. **AI Not Responding**: Check GEMINI_API_KEY configuration
3. **State Not Changing**: Verify simulator_control.txt permissions
4. **Frontend Errors**: Check API base URL configuration

### Debug Steps
1. Check console logs for error messages
2. Verify all services are running
3. Test API endpoints directly
4. Check file permissions for data files

## 📈 Future Enhancements

- **Predictive Analytics**: Forecast potential issues
- **Voice Commands**: Speech-to-text integration
- **Mobile App**: Native mobile interface
- **Advanced AI Models**: Multiple AI provider support
- **Custom Thresholds**: User-configurable alert levels

## 🔐 Security Notes

- API keys are stored in environment variables
- No sensitive data is logged or transmitted
- All communications use HTTPS in production
- File-based communication is temporary and cleaned up

---

**AI Co-Pilot** - Your intelligent factory companion! 🤖🏭 
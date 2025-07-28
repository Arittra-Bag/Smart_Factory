import React, { useState, useEffect } from 'react';
import { getLiveSimulationData, setSimulationState, sendCopilotMessage } from '../../api';

const AiCopilot = () => {
  // State for live data
  const [liveData, setLiveData] = useState({
    machine_temp: 0,
    vibration: 0,
    defect_rate: 0,
    state: 'Normal',
    operator: 'Unknown',
    timestamp: ''
  });

  // State for chat
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // State for simulator control
  const [simulatorState, setSimulatorState] = useState('Normal');

  // Thresholds for warning indicators
  const thresholds = {
    machine_temp: { warning: 65, critical: 75 },
    vibration: { warning: 25, critical: 30 },
    defect_rate: { warning: 2.0, critical: 3.0 }
  };

  // Fetch live data every second
  useEffect(() => {
    const fetchLiveData = async () => {
      try {
        const data = await getLiveSimulationData();
        setLiveData(data);
      } catch (error) {
        console.error('Error fetching live data:', error);
      }
    };

    // Initial fetch
    fetchLiveData();

    // Set up interval
    const interval = setInterval(fetchLiveData, 1000);

    // Cleanup
    return () => clearInterval(interval);
  }, []);

  // Handle simulator state change
  const handleSimulatorStateChange = async (newState) => {
    try {
      const response = await setSimulationState(newState);
      setSimulatorState(newState);
      console.log(`Simulator state changed to: ${newState}`);
    } catch (error) {
      console.error('Error changing simulator state:', error);
    }
  };

  // Handle sending chat message
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString()
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await sendCopilotMessage(inputMessage);

      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: response.response,
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Get status color based on value and threshold
  const getStatusColor = (value, threshold) => {
    if (value >= threshold.critical) return 'text-red-600 bg-red-50';
    if (value >= threshold.warning) return 'text-yellow-600 bg-yellow-50';
    return 'text-green-600 bg-green-50';
  };

  // Get status icon
  const getStatusIcon = (value, threshold) => {
    if (value >= threshold.critical) return '🔴';
    if (value >= threshold.warning) return '🟡';
    return '🟢';
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">AI Co-Pilot</h1>
        <p className="text-gray-600">Real-time factory monitoring and AI-powered analysis</p>
      </div>

      {/* Live Dashboard Section */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          📊 Live Factory Dashboard
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Machine Temperature */}
          <div className={`p-4 rounded-lg border ${getStatusColor(liveData.machine_temp, thresholds.machine_temp)}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Temperature</p>
                <p className="text-2xl font-bold">{liveData.machine_temp}°C</p>
              </div>
              <span className="text-2xl">{getStatusIcon(liveData.machine_temp, thresholds.machine_temp)}</span>
            </div>
          </div>

          {/* Vibration */}
          <div className={`p-4 rounded-lg border ${getStatusColor(liveData.vibration, thresholds.vibration)}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Vibration</p>
                <p className="text-2xl font-bold">{liveData.vibration}</p>
              </div>
              <span className="text-2xl">{getStatusIcon(liveData.vibration, thresholds.vibration)}</span>
            </div>
          </div>

          {/* Defect Rate */}
          <div className={`p-4 rounded-lg border ${getStatusColor(liveData.defect_rate, thresholds.defect_rate)}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Defect Rate</p>
                <p className="text-2xl font-bold">{liveData.defect_rate}%</p>
              </div>
              <span className="text-2xl">{getStatusIcon(liveData.defect_rate, thresholds.defect_rate)}</span>
            </div>
          </div>

          {/* Factory State */}
          <div className="p-4 rounded-lg border bg-blue-50 border-blue-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Factory State</p>
                <p className="text-2xl font-bold text-blue-600">{liveData.state}</p>
              </div>
              <span className="text-2xl">🏭</span>
            </div>
          </div>
        </div>

        {/* Additional Info */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
          <div>👤 Operator: {liveData.operator}</div>
          <div>⏰ Last Update: {liveData.timestamp}</div>
        </div>
      </div>

      {/* Simulator Control Section */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          🎛️ Simulator Control
        </h2>
        
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => handleSimulatorStateChange('Normal')}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              simulatorState === 'Normal'
                ? 'bg-green-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            🟢 Simulate: Normal
          </button>
          
          <button
            onClick={() => handleSimulatorStateChange('Overheating')}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              simulatorState === 'Overheating'
                ? 'bg-red-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            🔥 Simulate: Overheating
          </button>
          
          <button
            onClick={() => handleSimulatorStateChange('Belt_Slipping')}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              simulatorState === 'Belt_Slipping'
                ? 'bg-yellow-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            ⚠️ Simulate: Belt Slipping
          </button>
        </div>
      </div>

      {/* Chat Interface Section */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          🤖 AI Co-Pilot Chat
        </h2>
        
        {/* Chat Messages */}
        <div className="h-96 overflow-y-auto border rounded-lg p-4 mb-4 bg-gray-50">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <p className="text-lg mb-2">👋 Welcome to AI Co-Pilot!</p>
              <p>Ask me anything about the factory data and I'll provide insights.</p>
              <p className="text-sm mt-2">Try asking: "What is the current factory status?"</p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-blue-600 text-white'
                        : message.type === 'error'
                        ? 'bg-red-100 text-red-800'
                        : 'bg-white text-gray-800 border'
                    }`}
                  >
                    <div className="text-sm font-medium mb-1">
                      {message.type === 'user' ? 'You' : message.type === 'error' ? 'Error' : 'AI Co-Pilot'}
                    </div>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    <div className="text-xs opacity-70 mt-1">{message.timestamp}</div>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white text-gray-800 border px-4 py-2 rounded-lg">
                    <div className="text-sm font-medium mb-1">AI Co-Pilot</div>
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      <span>Analyzing factory data...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Chat Input */}
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about factory data, trends, or recommendations..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>

        {/* Quick Questions */}
        <div className="mt-4">
          <p className="text-sm text-gray-600 mb-2">Quick questions:</p>
          <div className="flex flex-wrap gap-2">
            {[
              "What is the current factory status?",
              "Are there any concerning trends?",
              "What should the operator do?",
              "Is the temperature normal?"
            ].map((question, index) => (
              <button
                key={index}
                onClick={() => setInputMessage(question)}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AiCopilot; 
import React, { useState } from 'react';
import { useNotifications } from '../../hooks/useNotifications';
import NotificationManager from './NotificationManager';

const NotificationDemo: React.FC = () => {
  const {
    isGranted,
    sendNotification,
    sendDefectNotification,
    sendProductionCompleteNotification,
    sendEmergencyNotification,
  } = useNotifications();

  const [customTitle, setCustomTitle] = useState('Custom Notification');
  const [customMessage, setCustomMessage] = useState('This is a custom notification message');

  const handleSendCustomNotification = () => {
    sendNotification(customTitle, customMessage);
  };

  const handleTestDefectNotification = () => {
    const defectTypes = ['Scratch', 'Dent', 'Crack', 'Discoloration'];
    const randomDefect = defectTypes[Math.floor(Math.random() * defectTypes.length)];
    const confidence = Math.random() * 30 + 70; // 70-100%
    sendDefectNotification(randomDefect, confidence);
  };

  const handleTestProductionComplete = () => {
    const batchSize = Math.floor(Math.random() * 50) + 10; // 10-60 items
    const duration = `${Math.floor(Math.random() * 30) + 5} minutes`;
    sendProductionCompleteNotification(batchSize, duration);
  };

  const handleTestEmergency = () => {
    const emergencies = [
      'Machine overheating detected!',
      'Safety system triggered!',
      'Power supply failure!',
      'Emergency stop activated!'
    ];
    const randomEmergency = emergencies[Math.floor(Math.random() * emergencies.length)];
    sendEmergencyNotification(randomEmergency);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Notification System Demo</h2>
        <NotificationManager />
      </div>

      {!isGranted && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span className="text-sm font-medium text-yellow-800">
              Enable notifications to test the system
            </span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Custom Notification */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-gray-900">Custom Notification</h3>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Title
              </label>
              <input
                type="text"
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Notification title"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Message
              </label>
              <textarea
                value={customMessage}
                onChange={(e) => setCustomMessage(e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Notification message"
              />
            </div>
            <button
              onClick={handleSendCustomNotification}
              disabled={!isGranted}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Send Custom Notification
            </button>
          </div>
        </div>

        {/* Predefined Notifications */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-gray-900">Predefined Notifications</h3>
          <div className="space-y-3">
            <button
              onClick={handleTestDefectNotification}
              disabled={!isGranted}
              className="w-full px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              🚨 Test Defect Detection
            </button>
            
            <button
              onClick={handleTestProductionComplete}
              disabled={!isGranted}
              className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              ✅ Test Production Complete
            </button>
            
            <button
              onClick={handleTestEmergency}
              disabled={!isGranted}
              className="w-full px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              🚨 Test Emergency Alert
            </button>
          </div>
        </div>
      </div>

      {/* Features List */}
      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-lg font-medium text-gray-900 mb-3">Features</h3>
        <ul className="space-y-2 text-sm text-gray-600">
          <li className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            <span>Works on desktop and mobile Chrome browsers</span>
          </li>
          <li className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            <span>Notifications display even when tab is not focused</span>
          </li>
          <li className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            <span>Fallback alerts when notifications are disabled</span>
          </li>
          <li className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            <span>Vibration support for mobile devices</span>
          </li>
          <li className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            <span>Auto-close notifications after 5 seconds</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default NotificationDemo; 
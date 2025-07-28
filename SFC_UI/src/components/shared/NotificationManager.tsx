import React, { useState, useEffect } from 'react';
import { useNotifications } from '../../hooks/useNotifications';

interface NotificationManagerProps {
  showReminder?: boolean;
  onPermissionChange?: (permission: 'default' | 'granted' | 'denied') => void;
}

const NotificationManager: React.FC<NotificationManagerProps> = ({
  showReminder = true,
  onPermissionChange,
}) => {
  const {
    permission,
    isSupported,
    isGranted,
    requestPermission,
    sendNotification,
  } = useNotifications();

  const [showReminderPopup, setShowReminderPopup] = useState(false);
  const [isRequesting, setIsRequesting] = useState(false);

  // Show reminder popup after 3 seconds if permission is not granted
  useEffect(() => {
    if (showReminder && isSupported && permission === 'default') {
      const timer = setTimeout(() => {
        setShowReminderPopup(true);
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [showReminder, isSupported, permission]);

  // Notify parent component of permission changes
  useEffect(() => {
    onPermissionChange?.(permission);
  }, [permission, onPermissionChange]);

  const handleRequestPermission = async () => {
    setIsRequesting(true);
    try {
      await requestPermission();
      setShowReminderPopup(false);
    } catch (error) {
      console.error('Failed to request permission:', error);
    } finally {
      setIsRequesting(false);
    }
  };

  const handleTestNotification = () => {
    sendNotification(
      '🔔 Test Notification',
      'This is a test notification from your Smart Factory app!'
    );
  };

  const handleDismissReminder = () => {
    setShowReminderPopup(false);
  };

  if (!isSupported) {
    return (
      <div className="flex items-center space-x-2 text-amber-600 bg-amber-50 px-3 py-2 rounded-lg">
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
        <span className="text-sm font-medium">Notifications not supported in this browser</span>
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Notification Status Icon */}
      <div className="flex items-center space-x-2">
        <div className="relative">
          <button
            onClick={handleTestNotification}
            disabled={!isGranted}
            className={`p-2 rounded-full transition-all duration-200 ${
              isGranted
                ? 'bg-green-100 text-green-600 hover:bg-green-200'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            }`}
            title={isGranted ? 'Click to test notification' : 'Notifications not enabled'}
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z" />
            </svg>
          </button>
          
          {/* Status indicator */}
          <div
            className={`absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-white ${
              isGranted ? 'bg-green-500' : 'bg-gray-400'
            }`}
          />
        </div>
        
        <div className="text-sm">
          <span className={`font-medium ${isGranted ? 'text-green-600' : 'text-gray-500'}`}>
            {isGranted ? 'Notifications Active' : 'Notifications Disabled'}
          </span>
        </div>
      </div>

      {/* Permission Request Button */}
      {!isGranted && (
        <button
          onClick={handleRequestPermission}
          disabled={isRequesting}
          className="ml-3 px-3 py-1 text-xs bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isRequesting ? 'Requesting...' : 'Enable'}
        </button>
      )}

      {/* Reminder Popup */}
      {showReminderPopup && (
        <div className="absolute top-full left-0 mt-2 w-80 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
          <div className="p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg className="w-6 h-6 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-medium text-gray-900">Enable Notifications</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Get real-time alerts for defects, production updates, and emergencies even when the app is not focused.
                </p>
                <div className="mt-3 flex space-x-2">
                  <button
                    onClick={handleRequestPermission}
                    disabled={isRequesting}
                    className="flex-1 px-3 py-2 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isRequesting ? 'Requesting...' : 'Enable Notifications'}
                  </button>
                  <button
                    onClick={handleDismissReminder}
                    className="px-3 py-2 text-xs font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotificationManager; 
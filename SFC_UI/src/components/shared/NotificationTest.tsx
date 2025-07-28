import React from 'react';
import { useNotifications } from '../../hooks/useNotifications';

const NotificationTest: React.FC = () => {
  const {
    permission,
    isSupported,
    isGranted,
    requestPermission,
    sendNotification,
    sendDefectNotification,
    sendProductionCompleteNotification,
    sendEmergencyNotification,
  } = useNotifications();

  const handleRequestPermission = async () => {
    try {
      const result = await requestPermission();
      console.log('Permission result:', result);
    } catch (error) {
      console.error('Permission request failed:', error);
    }
  };

  const handleTestNotification = () => {
    sendNotification('Test Notification', 'This is a test notification from the Smart Factory app!');
  };

  const handleTestDefect = () => {
    sendDefectNotification('Scratch', 87.5);
  };

  const handleTestProduction = () => {
    sendProductionCompleteNotification(25, '12 minutes');
  };

  const handleTestEmergency = () => {
    sendEmergencyNotification('Test emergency alert - this is a drill!');
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">Notification System Test</h2>
      
      {/* Status Display */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium mb-2">System Status</h3>
        <div className="space-y-1 text-sm">
          <div>Supported: <span className={isSupported ? 'text-green-600' : 'text-red-600'}>{isSupported ? 'Yes' : 'No'}</span></div>
          <div>Permission: <span className="font-mono">{permission}</span></div>
          <div>Granted: <span className={isGranted ? 'text-green-600' : 'text-red-600'}>{isGranted ? 'Yes' : 'No'}</span></div>
        </div>
      </div>

      {/* Permission Request */}
      {!isGranted && (
        <div className="mb-6">
          <button
            onClick={handleRequestPermission}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Request Notification Permission
          </button>
        </div>
      )}

      {/* Test Buttons */}
      <div className="space-y-3">
        <button
          onClick={handleTestNotification}
          disabled={!isGranted}
          className="w-full px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Test Basic Notification
        </button>
        
        <button
          onClick={handleTestDefect}
          disabled={!isGranted}
          className="w-full px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Test Defect Notification
        </button>
        
        <button
          onClick={handleTestProduction}
          disabled={!isGranted}
          className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Test Production Complete
        </button>
        
        <button
          onClick={handleTestEmergency}
          disabled={!isGranted}
          className="w-full px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Test Emergency Alert
        </button>
      </div>

      {/* Instructions */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-medium text-blue-900 mb-2">Testing Instructions</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Click "Request Permission" to enable notifications</li>
          <li>• Test each notification type with the buttons above</li>
          <li>• Switch to another tab to test background notifications</li>
          <li>• Check browser console for any errors</li>
        </ul>
      </div>
    </div>
  );
};

export default NotificationTest; 
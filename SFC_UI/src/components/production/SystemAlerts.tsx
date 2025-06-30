import React from 'react';
import { AlertTriangle, Info, CheckCircle, Clock } from 'lucide-react';

interface Alert {
  id: string;
  type: 'warning' | 'info' | 'success' | 'error';
  message: string;
  timestamp: string;
}

interface SystemAlertsProps {
  alerts: Alert[];
}

export default function SystemAlerts({ alerts }: SystemAlertsProps) {
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'warning': return AlertTriangle;
      case 'info': return Info;
      case 'success': return CheckCircle;
      case 'error': return AlertTriangle;
      default: return Info;
    }
  };

  const getAlertColors = (type: string) => {
    switch (type) {
      case 'warning': return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'info': return 'bg-blue-50 border-blue-200 text-blue-800';
      case 'success': return 'bg-green-50 border-green-200 text-green-800';
      case 'error': return 'bg-red-50 border-red-200 text-red-800';
      default: return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  const systemHealth = {
    cpu: 23,
    memory: 67,
    camera: 'Connected',
    database: 'Connected'
  };

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* System Alerts */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Alerts</h3>
        
        <div className="space-y-3">
          {alerts.map((alert) => {
            const Icon = getAlertIcon(alert.type);
            return (
              <div key={alert.id} className={`${getAlertColors(alert.type)} border rounded-lg p-3`}>
                <div className="flex items-start space-x-3">
                  <Icon className="h-4 w-4 sm:h-5 sm:w-5 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs sm:text-sm font-medium break-words">{alert.message}</p>
                    <div className="flex items-center mt-1 text-xs opacity-75">
                      <Clock className="h-3 w-3 mr-1 flex-shrink-0" />
                      <span className="truncate">{alert.timestamp}</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* System Health */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
        
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>CPU Usage</span>
              <span>{systemHealth.cpu}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${
                  systemHealth.cpu < 50 ? 'bg-green-500' : 
                  systemHealth.cpu < 80 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${systemHealth.cpu}%` }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Memory Usage</span>
              <span>{systemHealth.memory}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-300 ${
                  systemHealth.memory < 50 ? 'bg-green-500' : 
                  systemHealth.memory < 80 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${systemHealth.memory}%` }}
              />
            </div>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Camera</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full flex-shrink-0" />
                <span className="text-sm text-green-600">{systemHealth.camera}</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Database</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full flex-shrink-0" />
                <span className="text-sm text-green-600">{systemHealth.database}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="border-t pt-4 mt-4 text-xs text-gray-500 space-y-1">
          <div>Last Update: {new Date().toLocaleTimeString()}</div>
          <div>Uptime: 142h 34m</div>
        </div>
      </div>
    </div>
  );
}
import React from 'react';
import { Play, Pause, Square, CheckCircle, RotateCcw } from 'lucide-react';
import { ProductionState } from '../../types';

interface ControlPanelProps {
  productionState: ProductionState;
  machineStatus: 'STANDBY' | 'RUNNING' | 'EMERGENCY' | 'QUALITY_CHECK';
  emergencyResetProgress: number;
  onStartProduction: () => void;
  onPauseProduction: () => void;
  onEmergencyStop: () => void;
  onQualityCheck: () => void;
  onModeChange: (mode: 'production' | 'simulation' | 'test') => void;
  isLoading: boolean;
}

export default function ControlPanel({ 
  productionState, 
  machineStatus,
  emergencyResetProgress,
  onStartProduction, 
  onPauseProduction, 
  onEmergencyStop,
  onQualityCheck,
  onModeChange,
  isLoading
}: ControlPanelProps) {
  const controlButtons = [
    {
      label: 'Start Production',
      shortLabel: 'Start',
      icon: Play,
      onClick: onStartProduction,
      disabled: productionState.isRunning || isLoading,
      color: 'bg-green-600 hover:bg-green-700'
    },
    {
      label: 'Emergency Stop',
      shortLabel: 'Emergency',
      icon: Square,
      onClick: onEmergencyStop,
      disabled: isLoading,
      color: 'bg-red-600 hover:bg-red-700'
    },
    {
      label: 'Quality Check',
      shortLabel: 'Quality',
      icon: CheckCircle,
      onClick: onQualityCheck,
      disabled: isLoading,
      color: 'bg-blue-600 hover:bg-blue-700'
    }
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6">
      <h3 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Production Control Panel</h3>
      
      {/* System Status */}
      <div className="mb-4 sm:mb-6 p-3 sm:p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 sm:w-4 sm:h-4 rounded-full ${
              machineStatus === 'RUNNING' ? 'bg-green-500' :
              machineStatus === 'EMERGENCY' ? 'bg-red-500' :
              machineStatus === 'QUALITY_CHECK' ? 'bg-blue-500' :
              'bg-yellow-500'
            }`} />
            <div>
              <div className="text-sm sm:text-base font-semibold text-gray-900">System Status</div>
              <div className="text-xs sm:text-sm text-gray-600">{machineStatus.replace('_', ' ')}</div>
            </div>
          </div>
          {isLoading && (
            <div className="text-blue-600 text-sm">Processing...</div>
          )}
        </div>
      </div>

      {/* Emergency Reset Progress */}
      {emergencyResetProgress > 0 && (
        <div className="mb-4 sm:mb-6 p-3 sm:p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex justify-between text-sm text-red-800 mb-2">
            <span>Emergency Reset Progress</span>
            <span>{emergencyResetProgress}%</span>
          </div>
          <div className="w-full bg-red-200 rounded-full h-2 sm:h-3">
            <div 
              className="bg-red-600 h-2 sm:h-3 rounded-full transition-all duration-200" 
              style={{ width: `${emergencyResetProgress}%` }}
            />
          </div>
        </div>
      )}
      
      {/* Control Buttons */}
      <div className="grid grid-cols-2 gap-2 sm:gap-3 lg:gap-4 mb-6">
        {controlButtons.map((button, index) => (
          <button
            key={index}
            onClick={button.onClick}
            disabled={button.disabled}
            className={`${button.color} text-white font-medium py-3 sm:py-4 px-2 sm:px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-1 sm:space-x-2 hover:shadow-md active:scale-95`}
          >
            <button.icon className="h-4 w-4 sm:h-5 sm:w-5" />
            <span className="text-xs sm:text-sm font-medium">
              <span className="hidden sm:inline">{button.label}</span>
              <span className="sm:hidden">{button.shortLabel}</span>
            </span>
          </button>
        ))}
      </div>

      {/* Test Results */}
      {productionState.mode === 'test' && productionState.testAccuracy && (
        <div className="border-t pt-4 sm:pt-6 mt-4 sm:mt-6">
          <h4 className="text-sm sm:text-base font-semibold text-gray-800 mb-3 sm:mb-4">Test Results</h4>
          <div className="bg-green-50 border border-green-200 rounded-lg p-3 sm:p-4">
            <div className="flex items-center justify-between mb-2 sm:mb-3">
              <span className="text-sm sm:text-base text-green-700 font-medium">Test Accuracy</span>
              <span className="text-lg sm:text-xl font-bold text-green-800">{productionState.testAccuracy}%</span>
            </div>
            <div className="w-full bg-green-200 rounded-full h-2 sm:h-3">
              <div 
                className="bg-green-600 h-2 sm:h-3 rounded-full transition-all duration-300" 
                style={{ width: `${productionState.testAccuracy}%` }}
              />
            </div>
          </div>
          
          <div className="mt-3 sm:mt-4 text-xs sm:text-sm text-gray-600 space-y-1 sm:space-y-2">
            <div className="flex items-center space-x-2">
              <span className="text-green-500">✓</span>
              <span>Model Status: Healthy</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-green-500">✓</span>
              <span>Defect Detection: Active</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-green-500">✓</span>
              <span>Last Test: 2 minutes ago</span>
            </div>
          </div>
        </div>
      )}

      {/* Reset System */}
      <div className="border-t pt-4 sm:pt-6 mt-4 sm:mt-6">
        <button className="w-full flex items-center justify-center space-x-2 py-3 sm:py-4 px-4 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors hover:shadow-sm">
          <RotateCcw className="h-4 w-4 sm:h-5 sm:w-5" />
          <span className="text-sm sm:text-base font-medium">Reset System</span>
        </button>
      </div>
    </div>
  );
}
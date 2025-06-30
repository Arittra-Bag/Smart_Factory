import React from 'react';
import { Activity, AlertTriangle, Gauge, Cpu, Hand } from 'lucide-react';
import { ProductionState } from '../../types';

interface RealTimeMetricsProps {
  productionState: ProductionState;
  machineStatus: string;
}

export default function RealTimeMetrics({ productionState, machineStatus }: RealTimeMetricsProps) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6">
      <h3 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Real-time Production Metrics</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 mb-4 sm:mb-6">
        {/* Machine Status */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 sm:p-4 flex items-center space-x-3">
          <Cpu className="h-6 w-6 text-blue-600" />
          <div>
            <div className="text-lg font-bold">{machineStatus.replace('_', ' ')}</div>
            <div className="text-xs text-gray-500">Status</div>
          </div>
        </div>
        {/* Production Count */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 sm:p-4 flex items-center space-x-3">
          <Activity className="h-6 w-6 text-green-600" />
          <div>
            <div className="text-lg font-bold">{productionState.productionCount}</div>
            <div className="text-xs text-gray-500">Production Count</div>
          </div>
        </div>
        {/* Defect Count */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 sm:p-4 flex items-center space-x-3">
          <AlertTriangle className="h-6 w-6 text-red-600" />
          <div>
            <div className="text-lg font-bold">{productionState.defectCount}</div>
            <div className="text-xs text-gray-500">Defect Count</div>
          </div>
        </div>
        {/* Quality Score */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 sm:p-4 flex items-center space-x-3">
          <Gauge className="h-6 w-6 text-purple-600" />
          <div>
            <div className="text-lg font-bold">{productionState.qualityScore.toFixed(1)}%</div>
            <div className="text-xs text-gray-500">Quality Score</div>
          </div>
        </div>
        {/* Current Gesture */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 sm:p-4 flex items-center space-x-3">
          <Hand className="h-6 w-6 text-yellow-600" />
          <div>
            <div className="text-lg font-bold capitalize">{productionState.currentGesture || 'None'}</div>
            <div className="text-xs text-gray-500">Current Gesture</div>
          </div>
        </div>
      </div>
    </div>
  );
}
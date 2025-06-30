import React, { useState } from 'react';
import { AlertTriangle, Eye, Download } from 'lucide-react';
import { EmergencyEvent } from '../../types';

interface EmergencyLogTableProps {
  events: EmergencyEvent[];
}

export default function EmergencyLogTable({ events }: EmergencyLogTableProps) {
  const [filterType, setFilterType] = useState<'all' | 'EMERGENCY_STOP' | 'QUALITY_ALERT' | 'SYSTEM_RESET'>('all');

  const filteredEvents = events.filter(event => 
    filterType === 'all' || event.eventType === filterType
  );

  const getEventTypeColor = (type: string) => {
    switch (type) {
      case 'EMERGENCY_STOP': return 'bg-red-100 text-red-800';
      case 'QUALITY_ALERT': return 'bg-yellow-100 text-yellow-800';
      case 'SYSTEM_RESET': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'RUNNING': return 'text-green-600';
      case 'STANDBY': return 'text-yellow-600';
      case 'EMERGENCY': return 'text-red-600';
      case 'QUALITY_CHECK': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6 mb-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 space-y-2 sm:space-y-0">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="h-5 w-5 text-red-500" />
          <h3 className="text-lg font-semibold text-gray-900">Emergency Reset Log</h3>
        </div>
        <button className="flex items-center justify-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors w-full sm:w-auto">
          <Download className="h-4 w-4" />
          <span>Export Log</span>
        </button>
      </div>

      {/* Filter */}
      <div className="mb-4">
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value as any)}
          className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-red-500 focus:border-red-500 w-full sm:w-auto text-sm"
        >
          <option value="all">All Event Types</option>
          <option value="EMERGENCY_STOP">Emergency Stop</option>
          <option value="QUALITY_ALERT">Quality Alert</option>
          <option value="SYSTEM_RESET">System Reset</option>
        </select>
      </div>

      {/* Table - Desktop */}
      <div className="hidden lg:block overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Event Type</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Production</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Defects</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality Score</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Machine Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredEvents.map((event) => (
              <tr key={event.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{event.timestamp}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getEventTypeColor(event.eventType)}`}>
                    {event.eventType.replace('_', ' ')}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  <div>
                    <div className="font-medium">{event.productionCount}</div>
                    <div className="text-gray-500 text-xs">Total: {event.totalProduction}</div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div>
                    <div className="font-medium text-red-600">{event.defectCount}</div>
                    <div className="text-gray-500 text-xs">{event.defectRate}% rate</div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`text-sm font-medium ${
                    event.qualityScore >= 95 ? 'text-green-600' : 
                    event.qualityScore >= 90 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {event.qualityScore}%
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`text-sm font-medium ${getStatusColor(event.machineStatus)}`}>
                    {event.machineStatus.replace('_', ' ')}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  <button className="text-blue-600 hover:text-blue-800">
                    <Eye className="h-4 w-4" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Cards - Mobile/Tablet */}
      <div className="lg:hidden space-y-4">
        {filteredEvents.map((event) => (
          <div key={event.id} className="bg-gray-50 rounded-lg p-4 border">
            <div className="flex justify-between items-start mb-3">
              <div>
                <div className="font-semibold text-gray-900 text-sm">{event.timestamp}</div>
                <span className={`inline-block mt-1 px-2 py-1 text-xs font-medium rounded-full ${getEventTypeColor(event.eventType)}`}>
                  {event.eventType.replace('_', ' ')}
                </span>
              </div>
              <button className="text-blue-600 hover:text-blue-800 p-2">
                <Eye className="h-4 w-4" />
              </button>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <div className="text-xs text-gray-500">Production</div>
                <div className="font-medium">{event.productionCount}</div>
                <div className="text-xs text-gray-500">Total: {event.totalProduction}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Defects</div>
                <div className="font-medium text-red-600">{event.defectCount}</div>
                <div className="text-xs text-gray-500">{event.defectRate}% rate</div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Quality Score</div>
                <div className={`font-medium ${
                  event.qualityScore >= 95 ? 'text-green-600' : 
                  event.qualityScore >= 90 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {event.qualityScore}%
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Machine Status</div>
                <div className={`font-medium ${getStatusColor(event.machineStatus)}`}>
                  {event.machineStatus.replace('_', ' ')}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredEvents.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No emergency events found for the selected filter.
        </div>
      )}
    </div>
  );
}
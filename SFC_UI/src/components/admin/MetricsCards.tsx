import React from 'react';
import { TrendingUp, TrendingDown, Gauge, AlertTriangle, Shield, Clock } from 'lucide-react';
import { SystemMetrics } from '../../types';

interface MetricsCardsProps {
  metrics: SystemMetrics;
}

export default function MetricsCards({ metrics }: MetricsCardsProps) {
  const cards = [
    {
      title: 'Total Production',
      value: metrics.totalProduction.toLocaleString(),
      trend: '+5.2%',
      trending: 'up',
      icon: TrendingUp,
      color: 'blue'
    },
    {
      title: 'Quality Score',
      value: `${metrics.currentQualityScore}%`,
      trend: '+1.1%',
      trending: 'up',
      icon: Gauge,
      color: metrics.currentQualityScore >= 90 ? 'green' : metrics.currentQualityScore >= 70 ? 'yellow' : 'red'
    },
    {
      title: 'Defect Rate',
      value: `${metrics.activeDefectRate}%`,
      trend: '-0.3%',
      trending: 'down',
      icon: AlertTriangle,
      color: metrics.activeDefectRate <= 2 ? 'green' : metrics.activeDefectRate <= 5 ? 'yellow' : 'red'
    },
    {
      title: 'Machine Status',
      value: metrics.machineStatus.replace('_', ' '),
      status: metrics.machineStatus,
      icon: Shield,
      color: 'blue',
      showStatusBadge: false
    },
    {
      title: 'Emergency Events',
      value: metrics.emergencyEvents.toString(),
      period: 'Today',
      icon: AlertTriangle,
      color: metrics.emergencyEvents === 0 ? 'green' : 'red'
    },
    {
      title: 'System Uptime',
      value: `${Math.floor(metrics.systemUptime)}h ${Math.floor((metrics.systemUptime % 1) * 60)}m`,
      percentage: '99.2%',
      icon: Clock,
      color: 'green'
    }
  ];

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'green': return 'bg-green-50 border-green-200 text-green-800';
      case 'yellow': return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'red': return 'bg-red-50 border-red-200 text-red-800';
      case 'blue': return 'bg-blue-50 border-blue-200 text-blue-800';
      default: return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'RUNNING': return 'bg-green-100 text-green-800';
      case 'STANDBY': return 'bg-yellow-100 text-yellow-800';
      case 'EMERGENCY': return 'bg-red-100 text-red-800';
      case 'QUALITY_CHECK': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3 sm:gap-4 mb-6 sm:mb-8">
      {cards.map((card, index) => (
        <div key={index} className={`${getColorClasses(card.color)} border rounded-lg p-3 sm:p-4 shadow-sm`}>
          <div className="flex items-center justify-between mb-2">
            <card.icon className="h-4 w-4 sm:h-5 sm:w-5" />
            {card.trending && (
              <div className={`flex items-center text-xs ${
                card.trending === 'up' ? 'text-green-600' : 'text-red-600'
              }`}>
                {card.trending === 'up' ? (
                  <TrendingUp className="h-3 w-3 mr-1" />
                ) : (
                  <TrendingDown className="h-3 w-3 mr-1" />
                )}
                <span className="hidden sm:inline">{card.trend}</span>
              </div>
            )}
          </div>
          <div className="mb-1">
            <div className="text-lg sm:text-2xl font-bold">
              {card.value}
              {card.status && card.showStatusBadge !== false && (
                <span className={`ml-1 sm:ml-2 px-1 sm:px-2 py-1 text-xs rounded-full ${getStatusColor(card.status)}`}>
                  <span className="hidden sm:inline">{card.status.replace('_', ' ')}</span>
                  <span className="sm:hidden">{card.status.split('_')[0]}</span>
                </span>
              )}
            </div>
          </div>
          <div className="text-xs sm:text-sm opacity-75">
            <span className="block sm:inline">{card.title}</span>
            {card.period && <span className="block sm:inline"> ({card.period})</span>}
            {card.percentage && <span className="block sm:inline"> - {card.percentage}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
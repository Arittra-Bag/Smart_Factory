import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { TrendingUp, Download, BarChart3 } from 'lucide-react';
import { SafetyRecord } from '../../types';
import { getAnalytics } from '../../api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface AnalyticsSectionProps {
  records: SafetyRecord[];
}

export default function AnalyticsSection({ records }: AnalyticsSectionProps) {
  const [selectedDate, setSelectedDate] = useState('');
  const [chartData, setChartData] = useState<any>(null);
  const [statistics, setStatistics] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chartRef = useRef<any>(null);

  const uniqueDates = [...new Set(records.map(record => record.date))];

  useEffect(() => {
    if (selectedDate) {
      fetchAnalytics();
    } else {
      setChartData(null);
      setStatistics(null);
    }
  }, [selectedDate]);

  const fetchAnalytics = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const analyticsData = await getAnalytics(selectedDate);
      
      if (analyticsData.batch_sizes && analyticsData.defect_rates) {
    // Create chart data
    const data = {
          labels: analyticsData.batch_sizes.map((_: any, index: number) => `Batch ${index + 1}`),
      datasets: [
        {
          label: 'Defect Rate (%)',
              data: analyticsData.defect_rates,
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          tension: 0.1,
        },
        {
          label: 'Mean Defect Rate',
              data: Array(analyticsData.defect_rates.length).fill(analyticsData.mean_defect_rate),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderDash: [5, 5],
          tension: 0,
        },
        {
          label: 'Batch Size (normalized)',
              data: analyticsData.batch_sizes.map((size: number) => (size / 100) * 5), // Normalize to similar scale
          borderColor: 'rgb(16, 185, 129)',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          tension: 0.1,
        }
      ],
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top' as const,
          labels: {
            usePointStyle: true,
            padding: 20,
            font: {
              size: window.innerWidth < 640 ? 10 : 12
            }
          }
        },
        title: {
          display: true,
          text: `Production Analysis - ${selectedDate}`,
          font: {
            size: window.innerWidth < 640 ? 14 : 16
          }
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Defect Rate (%) / Normalized Batch Size',
            font: {
              size: window.innerWidth < 640 ? 10 : 12
            }
          },
          ticks: {
            font: {
              size: window.innerWidth < 640 ? 10 : 12
            }
          }
        },
        x: {
          title: {
            display: true,
            text: 'Production Batches',
            font: {
              size: window.innerWidth < 640 ? 10 : 12
            }
          },
          ticks: {
            font: {
              size: window.innerWidth < 640 ? 10 : 12
            }
          }
        }
      },
    };

    setChartData({ data, options });
        
        // Set statistics
        setStatistics({
          meanDefectRate: analyticsData.mean_defect_rate.toFixed(2),
          avgBatchSize: analyticsData.avg_batch_size.toFixed(1),
          totalBatches: analyticsData.batch_sizes.length,
          passRate: ((analyticsData.defect_rates.filter((rate: number) => rate <= 5).length / analyticsData.defect_rates.length) * 100).toFixed(1)
        });
      }
    } catch (err) {
      setError('Failed to load analytics data');
      console.error('Analytics error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportChart = () => {
    if (chartRef.current) {
      const url = chartRef.current.toBase64Image();
      const link = document.createElement('a');
      link.href = url;
      link.download = `production_chart_${selectedDate}.png`;
      link.click();
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6 mb-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 space-y-2 sm:space-y-0">
        <div className="flex items-center space-x-2">
          <BarChart3 className="h-5 w-5 text-blue-500" />
          <h3 className="text-lg font-semibold text-gray-900">Production Analytics</h3>
        </div>
        <button 
          disabled={!chartData}
          onClick={handleExportChart}
          className="flex items-center justify-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto"
        >
          <Download className="h-4 w-4" />
          <span>Export Chart</span>
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Date Selector */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">Select Date for Analysis:</label>
        <select
          value={selectedDate}
          onChange={(e) => setSelectedDate(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-full sm:w-auto"
        >
          <option value="">Choose a date...</option>
          {uniqueDates.map(date => (
            <option key={date} value={date}>{date}</option>
          ))}
        </select>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-12">
          <div className="animate-pulse">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 text-blue-500" />
            <p className="text-lg text-gray-700">Loading analytics data...</p>
          </div>
        </div>
      )}

      {/* Chart */}
      {chartData && !isLoading && (
        <div className="relative h-80 sm:h-96 mb-6">
          <Line ref={chartRef} data={chartData.data} options={chartData.options} />
        </div>
      )}

      {/* Statistics */}
      {statistics && !isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 sm:p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xl sm:text-2xl font-bold text-blue-800">{statistics.meanDefectRate}%</div>
                <div className="text-xs sm:text-sm text-blue-600">Mean Defect Rate</div>
              </div>
              <TrendingUp className="h-6 w-6 sm:h-8 sm:w-8 text-blue-500" />
            </div>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-3 sm:p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xl sm:text-2xl font-bold text-green-800">{statistics.avgBatchSize}</div>
                <div className="text-xs sm:text-sm text-green-600">Avg Batch Size</div>
              </div>
              <TrendingUp className="h-6 w-6 sm:h-8 sm:w-8 text-green-500" />
            </div>
          </div>
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 sm:p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xl sm:text-2xl font-bold text-purple-800">{statistics.totalBatches}</div>
                <div className="text-xs sm:text-sm text-purple-600">Total Batches</div>
              </div>
              <TrendingUp className="h-6 w-6 sm:h-8 sm:w-8 text-purple-500" />
            </div>
          </div>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 sm:p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xl sm:text-2xl font-bold text-yellow-800">{statistics.passRate}%</div>
                <div className="text-xs sm:text-sm text-yellow-600">Pass Rate</div>
              </div>
              <TrendingUp className="h-6 w-6 sm:h-8 sm:w-8 text-yellow-500" />
            </div>
          </div>
        </div>
      )}

      {!selectedDate && !isLoading && !error && (
        <div className="text-center py-12 text-gray-500">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 text-gray-400" />
          <p>Select a date to view production analytics</p>
        </div>
      )}
    </div>
  );
}
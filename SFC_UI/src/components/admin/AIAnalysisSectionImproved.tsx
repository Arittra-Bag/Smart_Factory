import React, { useState, useEffect } from 'react';
import { Brain, Download, Zap, TrendingUp, AlertCircle, CheckCircle, Calendar, BarChart3, FileText, Eye, EyeOff } from 'lucide-react';
import { getAISummary, exportAISummaryPDF, getAnalytics, exportGraphImage, exportCSVData } from '../../api';

export default function AIAnalysisSectionImproved() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const [analyticsData, setAnalyticsData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [isExportingGraph, setIsExportingGraph] = useState(false);
  const [isExportingCSV, setIsExportingCSV] = useState(false);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().slice(0, 10));
  const [showRawData, setShowRawData] = useState(false);

  const generateAnalysis = async () => {
    setIsGenerating(true);
    setError(null);
    
    try {
      console.log('🔍 Starting AI analysis generation...');
      console.log('🔍 Selected date:', selectedDate);
      console.log('🔍 API Base URL:', import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000');
      
      // Use selected date for both AI summary and analytics
      const [summary, analytics] = await Promise.all([
        getAISummary(selectedDate),
        getAnalytics(selectedDate)
      ]);
      
      console.log('🔍 API Response - Summary:', summary);
      console.log('🔍 API Response - Analytics:', analytics);
      console.log('🔍 Summary type:', typeof summary);
      console.log('🔍 Summary keys:', summary ? Object.keys(summary) : 'null/undefined');
      
      // Validate the response
      if (!summary) {
        throw new Error('No summary data received from server');
      }
      
      if (summary.error) {
        throw new Error(`Server error: ${summary.error}`);
      }
      
      // Ensure the summary has the expected structure
      const validatedSummary = {
        executiveSummary: summary.executiveSummary || 'No executive summary available.',
        trendAnalysis: Array.isArray(summary.trendAnalysis) ? summary.trendAnalysis : [],
        qualityAssessment: summary.qualityAssessment || {
          score: 0,
          trend: 'unknown',
          riskLevel: 'unknown'
        },
        recommendations: Array.isArray(summary.recommendations) ? summary.recommendations : [],
        alerts: Array.isArray(summary.alerts) ? summary.alerts : []
      };
      
      console.log('🔍 Validated summary:', validatedSummary);
      
      setAnalysis(validatedSummary);
      setAnalyticsData(analytics);
      
      console.log('🔍 Analysis state set successfully');
    } catch (err: any) {
      console.error('❌ AI analysis error:', err);
      console.error('❌ Error details:', {
        message: err.message,
        status: err.response?.status,
        statusText: err.response?.statusText,
        data: err.response?.data,
        config: err.config,
        stack: err.stack
      });
      
      // Provide more specific error messages
      let errorMessage = 'Failed to generate AI analysis';
      if (err.response?.status === 404) {
        errorMessage = 'AI analysis endpoint not found. Please check if the backend is running.';
      } else if (err.response?.status === 500) {
        errorMessage = 'Server error occurred while generating AI analysis.';
      } else if (err.code === 'ECONNREFUSED') {
        errorMessage = 'Cannot connect to backend server. Please ensure the Flask backend is running on port 5000.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setIsGenerating(false);
    }
  };

  function formatAISummary(analysis: any): string {
    let text = '';
    if (analysis.executiveSummary) {
      text += `Executive Summary:\n${analysis.executiveSummary}\n\n`;
    }
    if (analysis.trendAnalysis && analysis.trendAnalysis.length) {
      text += 'Trend Analysis:\n';
      analysis.trendAnalysis.forEach((item: string) => {
        text += `- ${item}\n`;
      });
      text += '\n';
    }
    if (analysis.qualityAssessment) {
      text += 'Quality Assessment:\n';
      text += `Score: ${analysis.qualityAssessment.score}\n`;
      text += `Trend: ${analysis.qualityAssessment.trend}\n`;
      text += `Risk Level: ${analysis.qualityAssessment.riskLevel}\n\n`;
    }
    if (analysis.recommendations && analysis.recommendations.length) {
      text += 'Recommendations:\n';
      analysis.recommendations.forEach((item: string) => {
        text += `- ${item}\n`;
      });
      text += '\n';
    }
    if (analysis.alerts && analysis.alerts.length) {
      text += 'Alerts:\n';
      analysis.alerts.forEach((alert: any) => {
        text += `- [${alert.type}] ${alert.message}\n`;
      });
      text += '\n';
    }
    return text.trim();
  }

  const handleExport = async () => {
    if (!analysis) return;
    setIsExporting(true);
    try {
      const payload = {
        ai_analysis: formatAISummary(analysis),
        date: selectedDate,
        batch_sizes: analyticsData?.batch_sizes || [],
        defect_rates: analyticsData?.defect_rates || []
      };
      const blob = await exportAISummaryPDF(payload);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `defect_analysis_report_${selectedDate}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
    } catch (e) {
      alert('Failed to export PDF');
    }
    setIsExporting(false);
  };

  const handleExportGraph = async (format: 'png' | 'jpg' = 'png') => {
    setIsExportingGraph(true);
    try {
      const blob = await exportGraphImage(selectedDate, format);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `defect_analysis_graph_${selectedDate}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
    } catch (e) {
      alert('Failed to export graph');
    }
    setIsExportingGraph(false);
  };

  const handleExportCSV = async () => {
    setIsExportingCSV(true);
    try {
      const blob = await exportCSVData(selectedDate);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `defect_analysis_data_${selectedDate}.csv`);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
    } catch (e) {
      alert('Failed to export CSV');
    }
    setIsExportingCSV(false);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 sm:mb-6 space-y-2 sm:space-y-0">
        <div className="flex items-center space-x-2">
          <Brain className="h-5 w-5 sm:h-6 sm:w-6 text-purple-500" />
          <h3 className="text-lg sm:text-xl font-semibold text-gray-900">AI-Powered Analysis</h3>
        </div>
        <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
          {/* Date Picker */}
          <div className="flex items-center space-x-2">
            <Calendar className="h-4 w-4 text-gray-500" />
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div className="flex space-x-2">
            <button 
              onClick={generateAnalysis}
              disabled={isGenerating}
              className="flex items-center justify-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Zap className="h-4 w-4" />
              <span>{isGenerating ? 'Generating...' : 'Generate Analysis'}</span>
            </button>
            <button 
              disabled={!analysis || isExporting}
              onClick={handleExport}
              className="flex items-center justify-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Download className="h-4 w-4" />
              <span>{isExporting ? 'Exporting...' : 'Export PDF'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Debug Toggle */}
      {analysis && (
        <div className="mb-4 flex justify-end">
          <button
            onClick={() => setShowRawData(!showRawData)}
            className="flex items-center space-x-2 px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
          >
            {showRawData ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            <span>{showRawData ? 'Hide' : 'Show'} Raw Data</span>
          </button>
        </div>
      )}

      {/* Raw Data Debug View */}
      {showRawData && analysis && (
        <div className="mb-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Raw API Response:</h4>
          <pre className="text-xs text-gray-600 overflow-auto max-h-40">
            {JSON.stringify(analysis, null, 2)}
          </pre>
        </div>
      )}

      {/* Export Options */}
      {analysis && (
        <div className="mb-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Export Options:</h4>
          <div className="flex flex-wrap gap-2">
            <button 
              onClick={() => handleExportGraph('png')}
              disabled={isExportingGraph}
              className="flex items-center space-x-2 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
              <BarChart3 className="h-4 w-4" />
              <span>{isExportingGraph ? 'Exporting...' : 'Export Graph (PNG)'}</span>
            </button>
            <button 
              onClick={() => handleExportGraph('jpg')}
              disabled={isExportingGraph}
              className="flex items-center space-x-2 px-3 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
              <BarChart3 className="h-4 w-4" />
              <span>{isExportingGraph ? 'Exporting...' : 'Export Graph (JPG)'}</span>
            </button>
            <button 
              onClick={handleExportCSV}
              disabled={isExportingCSV}
              className="flex items-center space-x-2 px-3 py-2 bg-teal-600 text-white rounded-md hover:bg-teal-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
              <FileText className="h-4 w-4" />
              <span>{isExportingCSV ? 'Exporting...' : 'Export CSV Data'}</span>
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <p className="text-red-800">{error}</p>
          </div>
        </div>
      )}

      {isGenerating && (
        <div className="text-center py-12">
          <div className="animate-pulse">
            <Brain className="h-12 w-12 mx-auto mb-4 text-purple-500" />
            <p className="text-lg text-gray-700">Analyzing production data...</p>
            <p className="text-sm text-gray-500 mb-4">Processing quality metrics, trends, and patterns</p>
            
            {/* Progress indicators */}
            <div className="max-w-md mx-auto space-y-2">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>Loading production data</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span>Generating analytics charts</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                <span>AI analysis in progress (may take 30-60 seconds)</span>
              </div>
            </div>
            
            <div className="mt-6">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-purple-600 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {analysis && (
        <div className="space-y-6">
          {/* Date Information */}
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
            <div className="flex items-center space-x-2">
              <Calendar className="h-4 w-4 text-gray-600" />
              <span className="text-sm font-medium text-gray-700">Analysis Date:</span>
              <span className="text-sm text-gray-900 font-semibold">{selectedDate}</span>
            </div>
          </div>

          {/* Executive Summary */}
          {analysis.executiveSummary && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="text-md font-semibold text-blue-800 mb-2 flex items-center">
                <Brain className="h-4 w-4 mr-2" />
                Executive Summary
              </h4>
              <p className="text-blue-700 leading-relaxed">{analysis.executiveSummary}</p>
            </div>
          )}

          {/* Trend Analysis */}
          {analysis.trendAnalysis && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="text-md font-semibold text-green-800 mb-2 flex items-center">
                <TrendingUp className="h-4 w-4 mr-2" />
                Trend Analysis
              </h4>
              <p className="text-green-700 leading-relaxed">{analysis.trendAnalysis}</p>
            </div>
          )}

          {/* Quality Assessment */}
          {analysis.qualityAssessment && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h4 className="text-md font-semibold text-purple-800 mb-2 flex items-center">
                <BarChart3 className="h-4 w-4 mr-2" />
                Quality Assessment
              </h4>
              <p className="text-purple-700 leading-relaxed">{analysis.qualityAssessment}</p>
            </div>
          )}

          {/* Root Cause Analysis */}
          {analysis.rootCauseAnalysis && (
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <h4 className="text-md font-semibold text-orange-800 mb-2 flex items-center">
                <AlertCircle className="h-4 w-4 mr-2" />
                Root Cause Analysis
              </h4>
              <p className="text-orange-700 leading-relaxed">{analysis.rootCauseAnalysis}</p>
            </div>
          )}

          {/* Risk Assessment */}
          {analysis.riskAssessment && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <h4 className="text-md font-semibold text-red-800 mb-2 flex items-center">
                <AlertCircle className="h-4 w-4 mr-2" />
                Risk Assessment
              </h4>
              <p className="text-red-700 leading-relaxed">{analysis.riskAssessment}</p>
            </div>
          )}

          {/* Performance Metrics */}
          {analysis.performanceMetrics && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="text-md font-semibold text-yellow-800 mb-2 flex items-center">
                <BarChart3 className="h-4 w-4 mr-2" />
                Performance Metrics
              </h4>
              <p className="text-yellow-700 leading-relaxed">{analysis.performanceMetrics}</p>
            </div>
          )}

          {/* Recommendations */}
          {analysis.recommendations && analysis.recommendations.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <h4 className="text-md font-semibold text-yellow-800 mb-3 flex items-center">
                <CheckCircle className="h-4 w-4 mr-2" />
                Recommendations
              </h4>
              <ul className="space-y-2">
                {analysis.recommendations.map((rec: string, index: number) => (
                  <li key={index} className="text-yellow-700 text-sm flex items-start">
                    <span className="mr-2 text-yellow-600">•</span>
                    <span className="leading-relaxed">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* AI Alerts */}
          {analysis.alerts && analysis.alerts.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-md font-semibold text-gray-800 flex items-center">
                <AlertCircle className="h-4 w-4 mr-2" />
                AI Alerts
              </h4>
              {analysis.alerts.map((alert: any, index: number) => (
                <div key={index} className={`p-3 rounded-lg border ${
                  alert.type === 'success' ? 'bg-green-50 border-green-200 text-green-800' :
                  alert.type === 'warning' ? 'bg-yellow-50 border-yellow-200 text-yellow-800' :
                  alert.type === 'error' ? 'bg-red-50 border-red-200 text-red-800' :
                  'bg-blue-50 border-blue-200 text-blue-800'
                }`}>
                  <div className="flex items-center space-x-2">
                    {alert.type === 'success' && <CheckCircle className="h-4 w-4" />}
                    {alert.type === 'warning' && <AlertCircle className="h-4 w-4" />}
                    {alert.type === 'error' && <AlertCircle className="h-4 w-4" />}
                    {alert.type === 'info' && <TrendingUp className="h-4 w-4" />}
                    <span className="text-sm font-medium">{alert.message}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {!analysis && !isGenerating && !error && (
        <div className="text-center py-12 text-gray-500">
          <Brain className="h-12 w-12 mx-auto mb-4 text-gray-400" />
          <p>Click "Generate Analysis" to get AI-powered insights</p>
        </div>
      )}
    </div>
  );
} 
import React, { useEffect, useState } from 'react';
import MetricsCards from './MetricsCards';
import SafetyRecordsTable from './SafetyRecordsTable';
import EmergencyLogTable from './EmergencyLogTable';
import AnalyticsSection from './AnalyticsSection';
import AIAnalysisSection from './AIAnalysisSectionImproved';
import { getSystemMetrics, getSafetyRecords, getEmergencyEvents } from '../../api';
import { SystemMetrics, SafetyRecord, EmergencyEvent } from '../../types';

export default function AdminDashboard() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [metricsError, setMetricsError] = useState(null);

  const [safetyRecords, setSafetyRecords] = useState<SafetyRecord[]>([]);
  const [safetyLoading, setSafetyLoading] = useState(true);
  const [safetyError, setSafetyError] = useState(null);

  const [emergencyEvents, setEmergencyEvents] = useState<EmergencyEvent[]>([]);
  const [emergencyLoading, setEmergencyLoading] = useState(true);
  const [emergencyError, setEmergencyError] = useState(null);

  useEffect(() => {
    setMetricsLoading(true);
    getSystemMetrics()
      .then(setMetrics)
      .catch(setMetricsError)
      .finally(() => setMetricsLoading(false));

    setSafetyLoading(true);
    getSafetyRecords()
      .then(setSafetyRecords)
      .catch(setSafetyError)
      .finally(() => setSafetyLoading(false));

    // setEmergencyLoading(true);
    // getEmergencyEvents()
    //   .then(setEmergencyEvents)
    //   .catch(setEmergencyError)
    //   .finally(() => setEmergencyLoading(false));
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 space-y-6 sm:space-y-8">
      {/* Metrics Cards */}
      {metricsLoading ? <div>Loading metrics...</div> : metricsError ? <div>Error loading metrics.</div> : metrics && <MetricsCards metrics={metrics} />}
      
      {/* Safety Records Table */}
      {safetyLoading ? <div>Loading safety records...</div> : safetyError ? <div>Error loading safety records.</div> : <SafetyRecordsTable records={safetyRecords} />}
      
      {/* Emergency Log Table */}
      {/* {emergencyLoading ? <div>Loading emergency logs...</div> : emergencyError ? <div>Error loading emergency logs.</div> : <EmergencyLogTable events={emergencyEvents} />} */}
      
      {/* Analytics Section */}
      <AnalyticsSection records={safetyRecords} />
      
      {/* AI Analysis Section */}
      <AIAnalysisSection />
    </div>
  );
}
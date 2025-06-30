// MOCK DATA DEPRECATED - Use API endpoints instead
// export const safetyRecords = ...
// export const emergencyEvents = ...
// export const systemMetrics = ...
// export const alerts = ...

import { SafetyRecord, EmergencyEvent, SystemMetrics, Alert } from '../types';

export const safetyRecords: SafetyRecord[] = [
  {
    id: '1',
    date: '2024-01-15',
    batchSize: 150,
    defectRate: 2.3,
    timestamp: '2024-01-15 08:30:00',
    qualityScore: 97.7,
    status: 'Pass'
  },
  {
    id: '2',
    date: '2024-01-15',
    batchSize: 145,
    defectRate: 1.8,
    timestamp: '2024-01-15 12:15:00',
    qualityScore: 98.2,
    status: 'Pass'
  },
  {
    id: '3',
    date: '2024-01-16',
    batchSize: 160,
    defectRate: 4.2,
    timestamp: '2024-01-16 09:45:00',
    qualityScore: 95.8,
    status: 'Fail'
  },
  {
    id: '4',
    date: '2024-01-16',
    batchSize: 155,
    defectRate: 3.1,
    timestamp: '2024-01-16 14:20:00',
    qualityScore: 96.9,
    status: 'Pass'
  },
  {
    id: '5',
    date: '2024-01-17',
    batchSize: 142,
    defectRate: 1.5,
    timestamp: '2024-01-17 10:00:00',
    qualityScore: 98.5,
    status: 'Pass'
  }
];

export const emergencyEvents: EmergencyEvent[] = [
  {
    id: '1',
    timestamp: '2024-01-15 14:23:12',
    eventType: 'EMERGENCY_STOP',
    totalProduction: 2340,
    productionCount: 145,
    defectCount: 3,
    defectRate: 2.1,
    qualityScore: 97.9,
    batchSize: 150,
    machineStatus: 'EMERGENCY'
  },
  {
    id: '2',
    timestamp: '2024-01-16 11:45:33',
    eventType: 'QUALITY_ALERT',
    totalProduction: 2485,
    productionCount: 160,
    defectCount: 7,
    defectRate: 4.4,
    qualityScore: 95.6,
    batchSize: 160,
    machineStatus: 'QUALITY_CHECK'
  },
  {
    id: '3',
    timestamp: '2024-01-17 09:12:45',
    eventType: 'SYSTEM_RESET',
    totalProduction: 2627,
    productionCount: 142,
    defectCount: 2,
    defectRate: 1.4,
    qualityScore: 98.6,
    batchSize: 142,
    machineStatus: 'STANDBY'
  }
];

export const systemMetrics: SystemMetrics = {
  totalProduction: 2769,
  currentQualityScore: 97.8,
  activeDefectRate: 2.2,
  machineStatus: 'RUNNING',
  emergencyEvents: 3,
  systemUptime: 142.5
};

export const alerts: Alert[] = [
  { id: '1', type: 'warning', message: 'Defect rate approaching threshold (2.5%)', timestamp: '2024-01-17 15:23:12' },
  { id: '2', type: 'info', message: 'Batch 2769 completed successfully', timestamp: '2024-01-17 15:18:45' },
  { id: '3', type: 'success', message: 'Quality check passed - 98.5% score', timestamp: '2024-01-17 15:12:33' },
];
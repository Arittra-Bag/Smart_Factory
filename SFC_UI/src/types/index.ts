export interface SafetyRecord {
  id: string;
  date: string;
  batchSize: number;
  defectRate: number;
  timestamp: string;
  qualityScore: number;
  status: 'Pass' | 'Fail';
}

export interface EmergencyEvent {
  id: string;
  timestamp: string;
  eventType: string;
  totalProduction: number;
  productionCount: number;
  defectCount: number;
  defectRate: number;
  qualityScore: number;
  batchSize: number;
  machineStatus: string;
}

export interface SystemMetrics {
  totalProduction: number;
  currentQualityScore: number;
  activeDefectRate: number;
  machineStatus: 'STANDBY' | 'RUNNING' | 'EMERGENCY' | 'QUALITY_CHECK';
  emergencyEvents: number;
  systemUptime: number;
  defectCount: number;
  batchSize: number;
  currentGesture: GestureType | null;
  fps: number;
}

export interface ProductionState {
  isRunning: boolean;
  productionCount: number;
  batchSize: number;
  defectCount: number;
  qualityScore: number;
  fps: number;
  mode: 'production' | 'simulation' | 'test';
  testAccuracy?: number;
  currentGesture?: GestureType | null;
}

export type GestureType = 'fist' | 'peace' | 'palm' | 'index' | null;

export interface Alert {
  id: string;
  type: 'warning' | 'info' | 'success' | 'error';
  message: string;
  timestamp: string;
}
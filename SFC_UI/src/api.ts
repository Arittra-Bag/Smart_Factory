import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export async function getSystemMetrics() {
  const res = await api.get('/api/metrics');
  return res.data;
}

export async function getSafetyRecords() {
  const res = await api.get('/api/safety-records');
  return res.data;
}

export async function getEmergencyEvents() {
  const res = await api.get('/api/emergency-logs');
  return res.data;
}

export async function getAnalytics(date?: string) {
  const res = await api.get('/api/analytics', { params: { date } });
  return res.data;
}

export async function getAISummary(date?: string) {
  const res = await api.get('/api/ai-summary', { 
    params: { date },
    timeout: 60000 // 60 seconds for AI analysis
  });
  return res.data;
}

export async function getAlerts() {
  const res = await api.get('/api/alerts');
  return res.data;
}

// Production control functions
export async function startProduction() {
  const res = await api.post('/api/production/start');
  return res.data;
}

export async function pauseProduction() {
  const res = await api.post('/api/production/pause');
  return res.data;
}

export async function emergencyStop() {
  const res = await api.post('/api/production/emergency_stop');
  return res.data;
}

export async function qualityCheck() {
  const res = await api.post('/api/production/quality_check');
  return res.data;
}

// Real-time frame processing
export async function processFrame(frameBlob: Blob) {
  const formData = new FormData();
  formData.append('frame', frameBlob, 'frame.jpg');
  
  const res = await api.post('/api/process-frame', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 5000, // 5 seconds for frame processing
  });
  return res.data;
}

// Smart factory control integration
export async function getSmartFactoryStatus() {
  const res = await api.get('/api/smart-factory/status');
  return res.data;
}

export async function getCurrentProductionImage() {
  const res = await api.get('/api/smart-factory/production-image', { responseType: 'blob' });
  return res.data;
}

export async function getCurrentDefectResult() {
  const res = await api.get('/api/smart-factory/defect-result');
  return res.data;
}

export async function getProductionMetrics() {
  const res = await api.get('/api/smart-factory/metrics');
  return res.data;
}

// Defect prediction
export async function predictDefect(imageFile: File) {
  const formData = new FormData();
  formData.append('image', imageFile);
  const res = await api.post('/api/defect/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
}

// Ensemble Detection API
export async function detectImage(imageFile: File, settings: any, enabledModels: Record<string, boolean>) {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('settings', JSON.stringify(settings));
  formData.append('enabled_models', JSON.stringify(enabledModels));
  
  const res = await api.post('/api/detect-image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 30000, // 30 seconds for detection processing
  });
  return res.data;
}

// Test mode functions
export async function startTestMode() {
  const res = await api.post('/api/test/start');
  return res.data;
}

export async function stopTestMode() {
  const res = await api.post('/api/test/stop');
  return res.data;
}

export async function getTestResults() {
  const res = await api.get('/api/test/results');
  return res.data;
}

// Add more API functions as needed for control, defect prediction, etc.

export async function exportAISummaryPDF(payload: {
  ai_analysis: string;
  date?: string;
  batch_sizes?: number[];
  defect_rates?: number[];
}) {
  const res = await api.post('/api/ai-summary-pdf', payload, { responseType: 'blob' });
  return res.data;
}

export async function exportGraphImage(date: string, format: 'png' | 'jpg' = 'png') {
  const res = await api.get(`/api/export/graph/${date}`, { 
    params: { format },
    responseType: 'blob' 
  });
  return res.data;
}

export async function exportCSVData(date: string) {
  const res = await api.get(`/api/export/csv/${date}`, { responseType: 'blob' });
  return res.data;
}

export async function recognizeGestureFromImage(imageFile: File | Blob) {
  const formData = new FormData();
  formData.append('image', imageFile);
  const res = await api.post('/api/gesture/recognize', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
}

export async function getProductionImage(): Promise<Blob> {
  const res = await api.get('/api/production/image', { responseType: 'blob' });
  return res.data;
}

export async function getProductionDefect() {
  const res = await api.get('/api/production/defect');
  return res.data;
} 
import React, { useState, useEffect, useCallback } from 'react';
import ControlPanel from './ControlPanel';
import VideoUploadProcessor from './VideoUploadProcessor';
import { getProductionMetrics, startProduction, pauseProduction, emergencyStop, qualityCheck } from '../../api';
import { ProductionState, GestureType } from '../../types';

export default function ProductionPage() {
  const [productionState, setProductionState] = useState<ProductionState>({
    isRunning: false,
    productionCount: 0,
    batchSize: 0,
    defectCount: 0,
    qualityScore: 0,
    fps: 0,
    mode: 'production',
    testAccuracy: undefined,
    currentGesture: null
  });
  const [machineStatus, setMachineStatus] = useState<'STANDBY' | 'RUNNING' | 'EMERGENCY' | 'QUALITY_CHECK'>('STANDBY');
  const [emergencyResetProgress, setEmergencyResetProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  // Poll backend for real-time metrics
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const metrics = await getProductionMetrics();
        setProductionState(prev => ({
          ...prev,
          productionCount: metrics.production_count || prev.productionCount,
          batchSize: metrics.batch_size || prev.batchSize,
          defectCount: metrics.defect_count || prev.defectCount,
          qualityScore: metrics.quality_score || prev.qualityScore,
          fps: metrics.fps || prev.fps,
          currentGesture: metrics.current_gesture || prev.currentGesture,
        }));
        setMachineStatus(metrics.machine_status || machineStatus);
        if (metrics.emergency_reset_progress !== undefined) {
          setEmergencyResetProgress(metrics.emergency_reset_progress);
        }
      } catch (err) {
        // Optionally handle error
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [machineStatus]);

  // Button handlers
  const handleStartProduction = useCallback(async () => {
    setIsLoading(true);
    try {
      await startProduction();
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handlePauseProduction = useCallback(async () => {
    setIsLoading(true);
    try {
      await pauseProduction();
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleEmergencyStop = useCallback(async () => {
    setIsLoading(true);
    try {
      await emergencyStop();
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleQualityCheck = useCallback(async () => {
    setIsLoading(true);
    try {
      await qualityCheck();
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Gesture handler
  const handleGestureDetected = useCallback((gesture: GestureType) => {
    if (!gesture) return;
    switch (gesture) {
      case 'fist':
        handleEmergencyStop();
        break;
      case 'peace':
        handleStartProduction();
        break;
      case 'palm':
        handleQualityCheck();
        break;
      default:
        break;
    }
    setProductionState(prev => ({ ...prev, currentGesture: gesture }));
  }, [handleEmergencyStop, handleStartProduction, handleQualityCheck]);

  // Mode change handler
  const handleModeChange = (mode: 'production' | 'simulation' | 'test') => {
    setProductionState(prev => ({ ...prev, mode }));
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div>
        <VideoUploadProcessor
          productionState={productionState}
          machineStatus={machineStatus}
          onGestureDetected={handleGestureDetected}
          currentGesture={productionState.currentGesture}
        />
      </div>
      <div>
        <ControlPanel
          productionState={productionState}
          machineStatus={machineStatus}
          emergencyResetProgress={emergencyResetProgress}
          onStartProduction={handleStartProduction}
          onPauseProduction={handlePauseProduction}
          onEmergencyStop={handleEmergencyStop}
          onQualityCheck={handleQualityCheck}
          onModeChange={handleModeChange}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
} 
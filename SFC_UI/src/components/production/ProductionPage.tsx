import React, { useState, useEffect, useCallback } from 'react';
import ControlPanel from './ControlPanel';
import VideoUploadProcessor from './VideoUploadProcessor';
import { getProductionMetrics, startProduction, pauseProduction, emergencyStop, qualityCheck } from '../../api';
import { ProductionState, GestureType } from '../../types';
import { useNotifications } from '../../hooks/useNotifications';

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
  
  // Notification system
  const { sendNotification, sendDefectNotification, sendProductionCompleteNotification, sendEmergencyNotification } = useNotifications();

  // Poll backend for real-time metrics
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const metrics = await getProductionMetrics();
        setProductionState(prev => {
          const newState = {
            ...prev,
            productionCount: metrics.production_count || prev.productionCount,
            batchSize: metrics.batch_size || prev.batchSize,
            defectCount: metrics.defect_count || prev.defectCount,
            qualityScore: metrics.quality_score || prev.qualityScore,
            fps: metrics.fps || prev.fps,
            currentGesture: metrics.current_gesture || prev.currentGesture,
          };
          
          // Send notification when defect count increases
          if (metrics.defect_count && metrics.defect_count > prev.defectCount) {
            const newDefects = metrics.defect_count - prev.defectCount;
            sendDefectNotification('Quality Issue', 85 + Math.random() * 10);
          }
          
          return newState;
        });
        setMachineStatus(metrics.machine_status || machineStatus);
        if (metrics.emergency_reset_progress !== undefined) {
          setEmergencyResetProgress(metrics.emergency_reset_progress);
        }
      } catch (err) {
        // Optionally handle error
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [machineStatus, sendDefectNotification]);

  // Button handlers
  const handleStartProduction = useCallback(async () => {
    setIsLoading(true);
    try {
      await startProduction();
      sendNotification('🚀 Production Started', 'Manufacturing process has been initiated successfully.');
    } finally {
      setIsLoading(false);
    }
  }, [sendNotification]);

  const handlePauseProduction = useCallback(async () => {
    setIsLoading(true);
    try {
      await pauseProduction();
      sendNotification('⏸️ Production Paused', 'Manufacturing process has been paused.');
    } finally {
      setIsLoading(false);
    }
  }, [sendNotification]);

  const handleEmergencyStop = useCallback(async () => {
    setIsLoading(true);
    try {
      await emergencyStop();
      sendEmergencyNotification('Emergency stop activated! Production halted immediately.');
    } finally {
      setIsLoading(false);
    }
  }, [sendEmergencyNotification]);

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
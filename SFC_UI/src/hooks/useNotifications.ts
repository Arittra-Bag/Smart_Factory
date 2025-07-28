import { useState, useEffect, useCallback } from 'react';
import notificationService, { NotificationPermission, NotificationOptions } from '../services/notificationService';

interface UseNotificationsReturn {
  permission: NotificationPermission;
  isSupported: boolean;
  isGranted: boolean;
  requestPermission: () => Promise<NotificationPermission>;
  sendNotification: (title: string, message: string) => void;
  sendDefectNotification: (defectType: string, confidence: number) => void;
  sendProductionCompleteNotification: (batchSize: number, duration: string) => void;
  sendEmergencyNotification: (message: string) => void;
  showFallbackAlert: (title: string, message: string) => void;
}

export const useNotifications = (): UseNotificationsReturn => {
  const [permission, setPermission] = useState<NotificationPermission>('default');
  const [isSupported, setIsSupported] = useState(false);

  useEffect(() => {
    // Initialize notification state
    setIsSupported(notificationService.isNotificationSupported());
    setPermission(notificationService.getPermissionStatus());
  }, []);

  const requestPermission = useCallback(async (): Promise<NotificationPermission> => {
    try {
      const newPermission = await notificationService.requestPermission();
      setPermission(newPermission);
      return newPermission;
    } catch (error) {
      console.error('Failed to request notification permission:', error);
      throw error;
    }
  }, []);

  const sendNotification = useCallback((title: string, message: string) => {
    if (permission === 'granted') {
      notificationService.sendSimpleNotification(title, message);
    } else {
      notificationService.showFallbackAlert(title, message);
    }
  }, [permission]);

  const sendDefectNotification = useCallback((defectType: string, confidence: number) => {
    if (permission === 'granted') {
      notificationService.sendDefectNotification(defectType, confidence);
    } else {
      notificationService.showFallbackAlert(
        '🚨 Defect Detected',
        `A ${defectType} defect was detected with ${confidence.toFixed(1)}% confidence`
      );
    }
  }, [permission]);

  const sendProductionCompleteNotification = useCallback((batchSize: number, duration: string) => {
    if (permission === 'granted') {
      notificationService.sendProductionCompleteNotification(batchSize, duration);
    } else {
      notificationService.showFallbackAlert(
        '✅ Production Complete',
        `Batch of ${batchSize} items completed in ${duration}`
      );
    }
  }, [permission]);

  const sendEmergencyNotification = useCallback((message: string) => {
    if (permission === 'granted') {
      notificationService.sendEmergencyNotification(message);
    } else {
      notificationService.showFallbackAlert('🚨 EMERGENCY', message);
    }
  }, [permission]);

  const showFallbackAlert = useCallback((title: string, message: string) => {
    notificationService.showFallbackAlert(title, message);
  }, []);

  return {
    permission,
    isSupported,
    isGranted: permission === 'granted',
    requestPermission,
    sendNotification,
    sendDefectNotification,
    sendProductionCompleteNotification,
    sendEmergencyNotification,
    showFallbackAlert,
  };
}; 
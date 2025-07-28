import { useState, useEffect, useCallback } from 'react';
import { useToast } from '../components/shared/ToastManager';
import notificationService from '../services/notificationService';
import pushNotificationService from '../services/pushNotificationService';

export interface NotificationPreferences {
  useToast: boolean;
  usePush: boolean;
  useBrowser: boolean;
  priorityLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface EnhancedNotificationOptions {
  title: string;
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  priority?: 'low' | 'medium' | 'high' | 'critical';
  requireInteraction?: boolean;
  silent?: boolean;
  vibrate?: number[];
  data?: any;
}

export const useEnhancedNotifications = (preferences: NotificationPreferences = {
  useToast: true,
  usePush: true,
  useBrowser: false, // Disabled by default to avoid blocking
  priorityLevel: 'medium'
}) => {
  const toast = useToast();
  const [pushSupported, setPushSupported] = useState(false);
  const [pushSubscribed, setPushSubscribed] = useState(false);

  useEffect(() => {
    // Check push notification support
    setPushSupported(pushNotificationService.isPushSupported());
    
    // Check if already subscribed
    if (pushNotificationService.getSubscription()) {
      setPushSubscribed(true);
    }
  }, []);

  const sendNotification = useCallback(async (options: EnhancedNotificationOptions) => {
    const {
      title,
      message,
      type = 'info',
      priority = 'medium',
      requireInteraction = false,
      silent = false,
      vibrate,
      data
    } = options;

    // Determine which notification methods to use based on priority and preferences
    const shouldUseToast = preferences.useToast && priority !== 'critical';
    const shouldUsePush = preferences.usePush && pushSupported && pushSubscribed;
    const shouldUseBrowser = preferences.useBrowser && priority === 'critical';

    // Send toast notification (non-blocking)
    if (shouldUseToast) {
      switch (type) {
        case 'success':
          toast.showSuccess(title, message);
          break;
        case 'error':
          toast.showError(title, message);
          break;
        case 'warning':
          toast.showWarning(title, message);
          break;
        case 'info':
        default:
          toast.showInfo(title, message);
          break;
      }
    }

    // Send push notification (for phone)
    if (shouldUsePush) {
      try {
        await pushNotificationService.sendLocalPushNotification({
          title,
          body: message,
          requireInteraction: priority === 'critical',
          silent,
          vibrate: priority === 'critical' ? [500, 200, 500, 200, 500] : vibrate,
          data: { ...data, priority, type }
        });
      } catch (error) {
        console.error('Failed to send push notification:', error);
      }
    }

    // Send browser notification (blocking, only for critical)
    if (shouldUseBrowser) {
      try {
        notificationService.sendNotification({
          title,
          message,
          requireInteraction: true,
          silent: false,
          vibrate: [500, 200, 500, 200, 500]
        });
      } catch (error) {
        console.error('Failed to send browser notification:', error);
      }
    }
  }, [toast, pushSupported, pushSubscribed, preferences]);

  // Convenience methods for different notification types
  const sendSuccess = useCallback((title: string, message: string, priority?: 'low' | 'medium' | 'high' | 'critical') => {
    sendNotification({ title, message, type: 'success', priority });
  }, [sendNotification]);

  const sendError = useCallback((title: string, message: string, priority?: 'low' | 'medium' | 'high' | 'critical') => {
    sendNotification({ title, message, type: 'error', priority });
  }, [sendNotification]);

  const sendWarning = useCallback((title: string, message: string, priority?: 'low' | 'medium' | 'high' | 'critical') => {
    sendNotification({ title, message, type: 'warning', priority });
  }, [sendNotification]);

  const sendInfo = useCallback((title: string, message: string, priority?: 'low' | 'medium' | 'high' | 'critical') => {
    sendNotification({ title, message, type: 'info', priority });
  }, [sendNotification]);

  // Smart notification methods for factory events
  const sendDefectNotification = useCallback((defectType: string, confidence: number) => {
    const priority = confidence > 90 ? 'high' : 'medium';
    sendNotification({
      title: '🚨 Defect Detected',
      message: `A ${defectType} defect was detected with ${confidence.toFixed(1)}% confidence`,
      type: 'error',
      priority,
      requireInteraction: priority === 'high'
    });
  }, [sendNotification]);

  const sendProductionCompleteNotification = useCallback((batchSize: number, duration: string) => {
    sendNotification({
      title: '✅ Production Complete',
      message: `Batch of ${batchSize} items completed in ${duration}`,
      type: 'success',
      priority: 'low'
    });
  }, [sendNotification]);

  const sendEmergencyNotification = useCallback((message: string) => {
    sendNotification({
      title: '🚨 EMERGENCY',
      message,
      type: 'error',
      priority: 'critical',
      requireInteraction: true,
      vibrate: [500, 200, 500, 200, 500]
    });
  }, [sendNotification]);

  const sendProductionStartNotification = useCallback((message: string) => {
    sendNotification({
      title: '🚀 Production Started',
      message,
      type: 'success',
      priority: 'medium'
    });
  }, [sendNotification]);

  const sendProductionPauseNotification = useCallback((message: string) => {
    sendNotification({
      title: '⏸️ Production Paused',
      message,
      type: 'warning',
      priority: 'medium'
    });
  }, [sendNotification]);

  // Push notification management
  const subscribeToPush = useCallback(async () => {
    try {
      const subscription = await pushNotificationService.subscribeToPush();
      if (subscription) {
        setPushSubscribed(true);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to subscribe to push notifications:', error);
      return false;
    }
  }, []);

  const unsubscribeFromPush = useCallback(async () => {
    try {
      await pushNotificationService.unsubscribe();
      setPushSubscribed(false);
      return true;
    } catch (error) {
      console.error('Failed to unsubscribe from push notifications:', error);
      return false;
    }
  }, []);

  return {
    // Notification methods
    sendNotification,
    sendSuccess,
    sendError,
    sendWarning,
    sendInfo,
    
    // Smart factory notifications
    sendDefectNotification,
    sendProductionCompleteNotification,
    sendEmergencyNotification,
    sendProductionStartNotification,
    sendProductionPauseNotification,
    
    // Push notification management
    pushSupported,
    pushSubscribed,
    subscribeToPush,
    unsubscribeFromPush,
    
    // Status
    preferences
  };
}; 
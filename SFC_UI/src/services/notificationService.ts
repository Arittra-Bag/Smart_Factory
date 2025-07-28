export interface NotificationOptions {
  title: string;
  message: string;
  icon?: string;
  badge?: string;
  tag?: string;
  requireInteraction?: boolean;
  silent?: boolean;
  vibrate?: number[];
}

// Browser Notification API options
export interface BrowserNotificationOptions {
  body: string;
  icon?: string;
  badge?: string;
  tag?: string;
  requireInteraction?: boolean;
  silent?: boolean;
  vibrate?: number[];
}

export type NotificationPermission = 'default' | 'granted' | 'denied';

class NotificationService {
  private permission: NotificationPermission = 'default';
  private isSupported: boolean = false;

  constructor() {
    this.isSupported = 'Notification' in window;
    if (this.isSupported) {
      this.permission = Notification.permission;
    }
  }

  /**
   * Check if notifications are supported in the current browser
   */
  isNotificationSupported(): boolean {
    return this.isSupported;
  }

  /**
   * Get current notification permission status
   */
  getPermissionStatus(): NotificationPermission {
    return this.permission;
  }

  /**
   * Request notification permission from the user
   */
  async requestPermission(): Promise<NotificationPermission> {
    if (!this.isSupported) {
      throw new Error('Notifications are not supported in this browser');
    }

    try {
      const permission = await Notification.requestPermission();
      this.permission = permission;
      return permission;
    } catch (error) {
      console.error('Error requesting notification permission:', error);
      throw error;
    }
  }

  /**
   * Send a notification with the given options
   */
  sendNotification(options: NotificationOptions): void {
    if (!this.isSupported) {
      console.warn('Notifications are not supported in this browser');
      return;
    }

    if (this.permission !== 'granted') {
      console.warn('Notification permission not granted');
      return;
    }

    try {
      const notificationOptions: BrowserNotificationOptions = {
        body: options.message,
        icon: options.icon || '/favicon.ico',
        badge: options.badge,
        tag: options.tag,
        requireInteraction: options.requireInteraction || false,
        silent: options.silent || false,
      };

      // Add vibrate only if supported (mobile devices)
      if ('vibrate' in navigator && options.vibrate) {
        notificationOptions.vibrate = options.vibrate;
      }

      const notification = new Notification(options.title, notificationOptions);

      // Auto-close notification after 5 seconds unless requireInteraction is true
      if (!options.requireInteraction) {
        setTimeout(() => {
          notification.close();
        }, 5000);
      }

      // Handle notification click
      notification.onclick = () => {
        notification.close();
        // Focus the window/tab
        window.focus();
      };
    } catch (error) {
      console.error('Error sending notification:', error);
      throw error;
    }
  }

  /**
   * Send a simple notification with title and message
   */
  sendSimpleNotification(title: string, message: string): void {
    this.sendNotification({ title, message });
  }

  /**
   * Send a defect detection notification
   */
  sendDefectNotification(defectType: string, confidence: number): void {
    this.sendNotification({
      title: '🚨 Defect Detected',
      message: `A ${defectType} defect was detected with ${confidence.toFixed(1)}% confidence`,
      tag: 'defect-detection',
      requireInteraction: true,
      vibrate: [200, 100, 200, 100, 200],
    });
  }

  /**
   * Send a production completion notification
   */
  sendProductionCompleteNotification(batchSize: number, duration: string): void {
    this.sendNotification({
      title: '✅ Production Complete',
      message: `Batch of ${batchSize} items completed in ${duration}`,
      tag: 'production-complete',
      vibrate: [100, 200, 100],
    });
  }

  /**
   * Send an emergency notification
   */
  sendEmergencyNotification(message: string): void {
    this.sendNotification({
      title: '🚨 EMERGENCY',
      message,
      tag: 'emergency',
      requireInteraction: true,
      vibrate: [500, 200, 500, 200, 500],
    });
  }

  /**
   * Check if the app is currently focused
   */
  isAppFocused(): boolean {
    return document.hasFocus();
  }

  /**
   * Add a fallback alert if notifications are not available
   */
  showFallbackAlert(title: string, message: string): void {
    alert(`${title}\n\n${message}`);
  }
}

// Create a singleton instance
const notificationService = new NotificationService();

export default notificationService; 
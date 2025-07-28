export interface PushNotificationOptions {
  title: string;
  body: string;
  icon?: string;
  badge?: string;
  tag?: string;
  data?: any;
  requireInteraction?: boolean;
  silent?: boolean;
  vibrate?: number[];
  actions?: Array<{
    action: string;
    title: string;
    icon?: string;
  }>;
}

class PushNotificationService {
  private isSupported: boolean = false;
  private registration: ServiceWorkerRegistration | null = null;
  private subscription: PushSubscription | null = null;

  constructor() {
    this.isSupported = 'serviceWorker' in navigator && 'PushManager' in window;
  }

  /**
   * Check if push notifications are supported
   */
  isPushSupported(): boolean {
    return this.isSupported;
  }

  /**
   * Register service worker for push notifications
   */
  async registerServiceWorker(): Promise<ServiceWorkerRegistration | null> {
    if (!this.isSupported) {
      console.warn('Push notifications are not supported in this browser');
      return null;
    }

    try {
      this.registration = await navigator.serviceWorker.register('/sw.js');
      console.log('Service Worker registered:', this.registration);
      return this.registration;
    } catch (error) {
      console.error('Service Worker registration failed:', error);
      return null;
    }
  }

  /**
   * Request push notification permission
   */
  async requestPermission(): Promise<NotificationPermission> {
    if (!this.isSupported) {
      throw new Error('Push notifications are not supported');
    }

    try {
      const permission = await Notification.requestPermission();
      return permission;
    } catch (error) {
      console.error('Failed to request push notification permission:', error);
      throw error;
    }
  }

  /**
   * Subscribe to push notifications
   */
  async subscribeToPush(): Promise<PushSubscription | null> {
    if (!this.registration) {
      await this.registerServiceWorker();
    }

    if (!this.registration) {
      return null;
    }

    try {
      // You'll need to replace this with your actual VAPID public key
      const vapidPublicKey = 'YOUR_VAPID_PUBLIC_KEY';
      const convertedVapidKey = this.urlBase64ToUint8Array(vapidPublicKey);

      this.subscription = await this.registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: convertedVapidKey,
      });

      console.log('Push subscription:', this.subscription);
      return this.subscription;
    } catch (error) {
      console.error('Failed to subscribe to push notifications:', error);
      return null;
    }
  }

  /**
   * Send push notification to server
   */
  async sendPushNotification(options: PushNotificationOptions): Promise<void> {
    if (!this.subscription) {
      console.warn('No push subscription available');
      return;
    }

    try {
      // Send to your backend server
      const response = await fetch('/api/push-notifications', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          subscription: this.subscription,
          notification: options,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send push notification');
      }

      console.log('Push notification sent successfully');
    } catch (error) {
      console.error('Failed to send push notification:', error);
      throw error;
    }
  }

  /**
   * Send local push notification (for testing)
   */
  async sendLocalPushNotification(options: PushNotificationOptions): Promise<void> {
    if (!this.registration) {
      await this.registerServiceWorker();
    }

    if (!this.registration) {
      return;
    }

    try {
      const notificationOptions: NotificationOptions = {
        body: options.body,
        icon: options.icon || '/favicon.ico',
        badge: options.badge,
        tag: options.tag,
        data: options.data,
        requireInteraction: options.requireInteraction || false,
        silent: options.silent || false,
      };

      // Add vibrate only if supported
      if ('vibrate' in navigator && options.vibrate) {
        (notificationOptions as any).vibrate = options.vibrate;
      }

      // Add actions if supported
      if (options.actions) {
        (notificationOptions as any).actions = options.actions;
      }

      await this.registration.showNotification(options.title, notificationOptions);
    } catch (error) {
      console.error('Failed to show local push notification:', error);
      throw error;
    }
  }

  /**
   * Get current subscription
   */
  getSubscription(): PushSubscription | null {
    return this.subscription;
  }

  /**
   * Unsubscribe from push notifications
   */
  async unsubscribe(): Promise<void> {
    if (this.subscription) {
      try {
        await this.subscription.unsubscribe();
        this.subscription = null;
        console.log('Unsubscribed from push notifications');
      } catch (error) {
        console.error('Failed to unsubscribe:', error);
        throw error;
      }
    }
  }

  /**
   * Convert VAPID key to Uint8Array
   */
  private urlBase64ToUint8Array(base64String: string): Uint8Array {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
      .replace(/-/g, '+')
      .replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }

  /**
   * Check if user has granted permission
   */
  hasPermission(): boolean {
    return Notification.permission === 'granted';
  }

  /**
   * Get permission status
   */
  getPermissionStatus(): NotificationPermission {
    return Notification.permission;
  }
}

// Create singleton instance
const pushNotificationService = new PushNotificationService();

export default pushNotificationService; 
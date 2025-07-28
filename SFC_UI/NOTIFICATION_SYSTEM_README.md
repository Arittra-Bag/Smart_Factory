# Smart Factory Notification System

A comprehensive browser notification system for the Smart Factory Control application that provides real-time alerts for production events, defects, and emergencies.

## Features

### ✅ Core Features
- **Browser Notifications API Integration**: Uses the native browser Notifications API
- **Cross-Platform Support**: Works on desktop and mobile Chrome browsers
- **Permission Management**: Automatic permission requests with user-friendly UI
- **Fallback Support**: Graceful degradation to browser alerts when notifications are disabled
- **Mobile Vibration**: Vibration support for mobile devices
- **Auto-Close**: Notifications automatically close after 5 seconds (configurable)

### 🚨 Smart Notifications
- **Defect Detection**: Real-time alerts when quality issues are detected
- **Production Updates**: Notifications for production start, pause, and completion
- **Emergency Alerts**: High-priority notifications for safety and system issues
- **Custom Notifications**: Flexible API for sending custom notifications

### 🎨 UI Components
- **NotificationManager**: Status indicator with permission controls
- **NotificationDemo**: Interactive demo showcasing all notification types
- **Visual Indicators**: Color-coded status indicators and icons
- **Responsive Design**: Works seamlessly on desktop and mobile

## Architecture

### File Structure
```
src/
├── services/
│   └── notificationService.ts    # Core notification service
├── hooks/
│   └── useNotifications.ts       # React hook for notifications
├── components/shared/
│   ├── NotificationManager.tsx   # UI component for notification status
│   └── NotificationDemo.tsx      # Demo component for testing
```

### Core Components

#### 1. NotificationService (`services/notificationService.ts`)
The main service class that handles all notification operations:

```typescript
// Check if notifications are supported
notificationService.isNotificationSupported()

// Request permission
await notificationService.requestPermission()

// Send notifications
notificationService.sendSimpleNotification(title, message)
notificationService.sendDefectNotification(defectType, confidence)
notificationService.sendProductionCompleteNotification(batchSize, duration)
notificationService.sendEmergencyNotification(message)
```

#### 2. useNotifications Hook (`hooks/useNotifications.ts`)
React hook that provides easy access to notification functionality:

```typescript
const {
  permission,
  isSupported,
  isGranted,
  requestPermission,
  sendNotification,
  sendDefectNotification,
  sendProductionCompleteNotification,
  sendEmergencyNotification
} = useNotifications();
```

#### 3. NotificationManager Component (`components/shared/NotificationManager.tsx`)
UI component that displays notification status and provides permission controls:

```typescript
<NotificationManager 
  showReminder={true}
  onPermissionChange={(permission) => console.log(permission)}
/>
```

## Usage Examples

### Basic Integration

```typescript
import { useNotifications } from '../hooks/useNotifications';

function MyComponent() {
  const { isGranted, sendNotification } = useNotifications();

  const handleDefectDetected = () => {
    if (isGranted) {
      sendNotification('🚨 Defect Detected', 'Quality issue found on production line');
    }
  };

  return (
    <button onClick={handleDefectDetected}>
      Test Defect Notification
    </button>
  );
}
```

### Production Integration

```typescript
// In ProductionPage.tsx
const { sendNotification, sendDefectNotification, sendEmergencyNotification } = useNotifications();

// Start production
const handleStartProduction = async () => {
  await startProduction();
  sendNotification('🚀 Production Started', 'Manufacturing process initiated');
};

// Emergency stop
const handleEmergencyStop = async () => {
  await emergencyStop();
  sendEmergencyNotification('Emergency stop activated!');
};

// Monitor for defects
useEffect(() => {
  if (defectCount > previousDefectCount) {
    sendDefectNotification('Quality Issue', 85.5);
  }
}, [defectCount]);
```

### Custom Notifications

```typescript
// Send custom notification
sendNotification('Custom Title', 'Custom message content');

// Send with specific options
notificationService.sendNotification({
  title: 'Custom Notification',
  message: 'This is a custom notification',
  requireInteraction: true,
  vibrate: [200, 100, 200]
});
```

## Browser Compatibility

### Supported Browsers
- ✅ Chrome (Desktop & Mobile)
- ✅ Edge (Chromium-based)
- ✅ Firefox (Desktop)
- ✅ Safari (Desktop & Mobile) - Limited support

### Feature Support
| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| Basic Notifications | ✅ | ✅ | ✅ | ✅ |
| Vibration | ✅ | ❌ | ❌ | ✅ |
| Silent Notifications | ✅ | ✅ | ❌ | ✅ |
| Require Interaction | ✅ | ✅ | ❌ | ✅ |

## Testing

### Manual Testing
1. **Permission Request**: Click "Enable" button in NotificationManager
2. **Test Notifications**: Use NotificationDemo component to test all notification types
3. **Background Testing**: Switch to another tab and trigger notifications
4. **Mobile Testing**: Test on mobile Chrome with vibration

### Automated Testing
```typescript
// Test notification permission
test('should request notification permission', async () => {
  const { requestPermission } = renderHook(() => useNotifications());
  const permission = await requestPermission();
  expect(permission).toBe('granted');
});

// Test notification sending
test('should send notification when permission granted', () => {
  const { sendNotification } = renderHook(() => useNotifications());
  sendNotification('Test', 'Test message');
  // Verify notification was sent
});
```

## Configuration

### Notification Options
```typescript
interface NotificationOptions {
  title: string;
  message: string;
  icon?: string;              // Custom icon URL
  badge?: string;             // Badge icon URL
  tag?: string;               // Group notifications
  requireInteraction?: boolean; // Prevent auto-close
  silent?: boolean;           // Disable sound
  vibrate?: number[];         // Vibration pattern
}
```

### Auto-Close Timing
```typescript
// Default: 5 seconds
// Customize in notificationService.ts
setTimeout(() => {
  notification.close();
}, 5000); // Change this value
```

## Security Considerations

### HTTPS Requirement
- Notifications require HTTPS in production
- Local development works with HTTP
- Service workers may be required for background notifications

### Permission Handling
- Always check permission before sending notifications
- Provide fallback alerts when notifications are disabled
- Respect user's choice to deny permissions

### Content Security
- Sanitize notification content to prevent XSS
- Avoid sensitive information in notifications
- Use appropriate notification tags for grouping

## Troubleshooting

### Common Issues

#### 1. Notifications Not Showing
- Check browser permission settings
- Ensure HTTPS is used in production
- Verify notification support with `isNotificationSupported()`

#### 2. Permission Denied
- Guide users to browser settings
- Provide fallback alerts
- Show helpful error messages

#### 3. Mobile Notifications
- Test on actual mobile devices
- Check vibration API support
- Verify mobile Chrome compatibility

### Debug Mode
```typescript
// Enable debug logging
const DEBUG_NOTIFICATIONS = true;

if (DEBUG_NOTIFICATIONS) {
  console.log('Notification permission:', permission);
  console.log('Notification supported:', isSupported);
}
```

## Future Enhancements

### Planned Features
- [ ] Service Worker integration for background notifications
- [ ] Notification sound customization
- [ ] Notification history and management
- [ ] Push notification support
- [ ] Notification templates and scheduling
- [ ] Multi-language notification support

### Performance Optimizations
- [ ] Notification queuing for high-frequency events
- [ ] Debounced notification sending
- [ ] Notification deduplication
- [ ] Memory usage optimization

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `npm install`
3. Start development server: `npm run dev`
4. Test notifications in browser

### Code Style
- Use TypeScript for type safety
- Follow React hooks best practices
- Maintain consistent error handling
- Add comprehensive JSDoc comments

### Testing Guidelines
- Test on multiple browsers
- Test permission scenarios
- Test mobile devices
- Test background/foreground behavior

## License

This notification system is part of the Smart Factory Control application and follows the same licensing terms. 
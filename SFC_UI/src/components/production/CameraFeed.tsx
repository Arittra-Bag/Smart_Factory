import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { Camera, Hand, AlertCircle } from 'lucide-react';
import { ProductionState, GestureType } from '../../types';
import { recognizeGestureFromImage } from '../../api';

interface CameraFeedProps {
  onGestureDetected: (gesture: GestureType) => void;
  currentGesture: GestureType;
  productionState: ProductionState;
}

export default function CameraFeed({ onGestureDetected, currentGesture, productionState }: CameraFeedProps) {
  const [isConnected, setIsConnected] = useState(true);
  const [loading, setLoading] = useState(false);
  const webcamRef = useRef<Webcam>(null);
  const [lastGesture, setLastGesture] = useState<GestureType | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isConnected) {
      interval = setInterval(async () => {
        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();
          if (imageSrc) {
            setLoading(true);
            // Convert base64 to Blob
            const res = await fetch(imageSrc);
            const blob = await res.blob();
            try {
              const result = await recognizeGestureFromImage(blob);
              if (result && result.gesture) {
                setLastGesture(result.gesture);
                onGestureDetected(result.gesture);
              }
            } catch (err) {
              // Optionally handle error
            } finally {
              setLoading(false);
            }
          }
        }
      }, 2000); // every 2 seconds
    }
    return () => { if (interval) clearInterval(interval); };
  }, [isConnected, onGestureDetected]);

  const gestureInstructions = [
    { gesture: '✊', name: 'Fist', action: 'Emergency Stop', color: 'text-red-600' },
    { gesture: '✌️', name: 'Peace Sign', action: 'Start Production', color: 'text-green-600' },
    { gesture: '✋', name: 'Palm', action: 'Quality Check', color: 'text-blue-600' },
    { gesture: '👆', name: 'Index Finger', action: 'Plot Defect Rate', color: 'text-purple-600' },
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sm:p-6">
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 space-y-2 sm:space-y-0">
        <div className="flex items-center space-x-2">
          <Camera className="h-5 w-5 sm:h-6 sm:w-6 text-blue-500" />
          <h3 className="text-lg sm:text-xl font-semibold text-gray-900">Live Camera Feed</h3>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-gray-600">{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>

      {/* Camera Display */}
      <div className="relative bg-gray-900 rounded-lg mb-4 h-48 sm:h-64 lg:h-80 xl:h-96">
        {isConnected ? (
          <>
            {/* Real webcam feed */}
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              className="absolute inset-0 w-full h-full object-cover rounded-lg"
              videoConstraints={{ facingMode: 'user' }}
            />
            {/* Gesture detection overlay */}
            {(currentGesture || lastGesture) && (
              <div className="absolute top-2 sm:top-4 left-2 sm:left-4 bg-black bg-opacity-75 text-white px-2 sm:px-3 py-1 sm:py-2 rounded-lg">
                <div className="flex items-center space-x-2">
                  <Hand className="h-3 w-3 sm:h-4 sm:w-4" />
                  <span className="text-xs sm:text-sm">
                    Detected: {currentGesture || lastGesture}
                  </span>
                </div>
              </div>
            )}
            {/* Loading spinner */}
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-30 rounded-lg">
                <div className="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-8 w-8"></div>
              </div>
            )}
            {/* Production overlay */}
            <div className="absolute top-2 sm:top-4 right-2 sm:right-4 bg-black bg-opacity-75 text-white px-2 sm:px-3 py-1 sm:py-2 rounded-lg">
              <div className="text-xs sm:text-sm">
                <div>Production: #{productionState.productionCount}</div>
                <div>Quality: {productionState.qualityScore.toFixed(1)}%</div>
              </div>
            </div>
            {/* Status overlay */}
            <div className="absolute bottom-2 sm:bottom-4 left-2 sm:left-4 bg-black bg-opacity-75 text-white px-2 sm:px-3 py-1 sm:py-2 rounded-lg">
              <div className="text-xs sm:text-sm">
                <div>Mode: {productionState.mode}</div>
                <div>Batch: {productionState.batchSize}</div>
              </div>
            </div>
          </>
        ) : (
          <div className="absolute inset-0 bg-red-900 rounded-lg flex items-center justify-center">
            <div className="text-center text-white">
              <AlertCircle className="h-8 w-8 sm:h-12 sm:w-12 lg:h-16 lg:w-16 mx-auto mb-2 sm:mb-4" />
              <p className="text-sm sm:text-base lg:text-lg mb-1 sm:mb-2">Camera Disconnected</p>
              <p className="text-xs sm:text-sm opacity-75">Check camera connection</p>
            </div>
          </div>
        )}
      </div>
      {/* Gesture Instructions */}
      <div className="bg-gray-50 rounded-lg p-3 sm:p-4">
        <h4 className="text-sm sm:text-base font-semibold text-gray-700 mb-3">Gesture Controls</h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3">
          {gestureInstructions.map((instruction, index) => (
            <div key={index} className="flex items-center space-x-2 sm:space-x-3 p-2 sm:p-3 bg-white rounded border hover:shadow-sm transition-shadow duration-200">
              <span className="text-lg sm:text-2xl lg:text-3xl flex-shrink-0">{instruction.gesture}</span>
              <div className="min-w-0 flex-1">
                <div className="text-xs sm:text-sm font-medium text-gray-900 truncate">{instruction.name}</div>
                <div className={`text-xs sm:text-sm ${instruction.color} truncate`}>{instruction.action}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
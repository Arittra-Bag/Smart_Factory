import React, { useState, useEffect, useRef } from 'react';
import { ProductionState, GestureType } from '../../types';
import { processFrame } from '../../api';

interface RealTimeVideoProcessorProps {
  productionState: ProductionState;
  machineStatus: 'STANDBY' | 'RUNNING' | 'EMERGENCY' | 'QUALITY_CHECK';
  onGestureDetected: (gesture: GestureType) => void;
  currentGesture: GestureType | null;
}

export default function RealTimeVideoProcessor({
  productionState,
  machineStatus,
  onGestureDetected,
  currentGesture
}: RealTimeVideoProcessorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [productionImageUrl, setProductionImageUrl] = useState<string | null>(null);
  const [defectResult, setDefectResult] = useState<{
    is_defective: boolean;
    defect_rate: number;
    prediction: number;
  } | null>(null);
  const [processingFrame, setProcessingFrame] = useState(false);

  // Start video stream
  useEffect(() => {
    const startVideoStream = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: 640, 
            height: 480,
            facingMode: 'user'
          } 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsStreaming(true);
          setStreamError(null);
        }
      } catch (error) {
        console.error('Error accessing camera:', error);
        setStreamError('Could not access camera. Please check permissions.');
        setIsStreaming(false);
      }
    };

    startVideoStream();

    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Process frames for gesture detection and production monitoring
  useEffect(() => {
    if (!isStreaming || !videoRef.current || !canvasRef.current) return;

    const processFrameInterval = async () => {
      if (processingFrame) return;
      setProcessingFrame(true);

      try {
        const video = videoRef.current!;
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext('2d')!;

        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to blob for API call
        canvas.toBlob(async (blob) => {
          if (!blob) return;

          try {
            // Send frame for processing to backend using the API function
            const result = await processFrame(blob);
            
            // Update gesture if detected
            if (result.gesture && result.gesture !== currentGesture) {
              onGestureDetected(result.gesture);
            }

            // Update production image if available
            if (result.production_image) {
              setProductionImageUrl(`data:image/jpeg;base64,${result.production_image}`);
            }

            // Update defect result if available
            if (result.defect_result) {
              setDefectResult(result.defect_result);
            }

          } catch (error) {
            console.error('Error processing frame:', error);
          } finally {
            setProcessingFrame(false);
          }
        }, 'image/jpeg', 0.8);

      } catch (error) {
        console.error('Error in frame processing:', error);
      }
    };

    // Process frames at 10 FPS (every 100ms)
    const interval = setInterval(processFrameInterval, 100);

    return () => clearInterval(interval);
  }, [isStreaming, currentGesture, onGestureDetected, processingFrame]);

  const getStatusColor = () => {
    switch (machineStatus) {
      case 'RUNNING': return 'text-green-600';
      case 'EMERGENCY': return 'text-red-600';
      case 'QUALITY_CHECK': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusBgColor = () => {
    switch (machineStatus) {
      case 'RUNNING': return 'bg-green-100 border-green-300';
      case 'EMERGENCY': return 'bg-red-100 border-red-300';
      case 'QUALITY_CHECK': return 'bg-yellow-100 border-yellow-300';
      default: return 'bg-gray-100 border-gray-300';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Real-Time Production Monitor</h3>
        <div className={`px-3 py-1 rounded-full text-sm font-medium border ${getStatusBgColor()} ${getStatusColor()}`}>
          {machineStatus}
        </div>
      </div>

      {streamError ? (
        <div className="text-center py-8">
          <div className="text-red-500 mb-2">⚠️ Camera Error</div>
          <p className="text-gray-600 text-sm">{streamError}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Live Camera Feed */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Live Camera Feed</h4>
            <div className="relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-64 object-cover rounded-lg border border-gray-300"
              />
              <canvas
                ref={canvasRef}
                className="hidden"
              />
              
              {/* Gesture Overlay */}
              {currentGesture && (
                <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-sm">
                  Gesture: {currentGesture.toUpperCase()}
                </div>
              )}

              {/* Processing Indicator */}
              {processingFrame && (
                <div className="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-sm animate-pulse">
                  Processing...
                </div>
              )}
            </div>
          </div>

          {/* Production Image and Defect Analysis */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Production Item Analysis</h4>
            <div className="space-y-3">
              {/* Production Image */}
              <div className="relative">
                {productionImageUrl ? (
                  <img 
                    src={productionImageUrl} 
                    alt="Production Item" 
                    className="w-full h-32 object-contain rounded border border-gray-300"
                  />
                ) : (
                  <div className="w-full h-32 flex items-center justify-center bg-gray-100 rounded border border-gray-300 text-gray-400 text-sm">
                    No Production Image
                  </div>
                )}
                
                {/* Production Count Overlay */}
                <div className="absolute top-1 left-1 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-xs">
                  Item #{productionState.productionCount}
                </div>
              </div>

              {/* Defect Analysis Result */}
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Defect Analysis</span>
                  {defectResult && (
                    <span className={`text-xs px-2 py-1 rounded ${
                      defectResult.is_defective 
                        ? 'bg-red-100 text-red-800' 
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {defectResult.is_defective ? 'DEFECTIVE' : 'OK'}
                    </span>
                  )}
                </div>
                
                {defectResult ? (
                  <div className="space-y-1">
                    <div className="text-xs text-gray-600">
                      Defect Rate: <span className="font-medium">{defectResult.defect_rate.toFixed(2)}%</span>
                    </div>
                    <div className="text-xs text-gray-600">
                      Confidence: <span className="font-medium">{(defectResult.prediction * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-xs text-gray-500">Analyzing...</div>
                )}
              </div>

              {/* Real-time Metrics */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-blue-50 p-2 rounded">
                  <div className="text-blue-800 font-medium">Production</div>
                  <div className="text-blue-600">{productionState.productionCount}</div>
                </div>
                <div className="bg-red-50 p-2 rounded">
                  <div className="text-red-800 font-medium">Defects</div>
                  <div className="text-red-600">{productionState.defectCount}</div>
                </div>
                <div className="bg-green-50 p-2 rounded">
                  <div className="text-green-800 font-medium">Quality</div>
                  <div className="text-green-600">{productionState.qualityScore.toFixed(1)}%</div>
                </div>
                <div className="bg-purple-50 p-2 rounded">
                  <div className="text-purple-800 font-medium">FPS</div>
                  <div className="text-purple-600">{productionState.fps}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <h5 className="text-sm font-medium text-gray-700 mb-2">Gesture Controls</h5>
        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
          <div>✌️ Peace Sign: Start Production</div>
          <div>✋ Palm: Quality Check</div>
          <div>✊ Fist: Emergency Stop</div>
        </div>
      </div>
    </div>
  );
} 
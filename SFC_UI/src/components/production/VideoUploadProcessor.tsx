import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Pause, RotateCcw, Eye, AlertCircle } from 'lucide-react';
import { ProductionState, GestureType } from '../../types';
import { processFrame, getProductionMetrics, getCurrentProductionImage } from '../../api';

interface VideoUploadProcessorProps {
  productionState: ProductionState;
  machineStatus: 'STANDBY' | 'RUNNING' | 'EMERGENCY' | 'QUALITY_CHECK';
  onGestureDetected: (gesture: GestureType) => void;
  currentGesture: GestureType | null;
}

export default function VideoUploadProcessor({
  productionState,
  machineStatus,
  onGestureDetected,
  currentGesture
}: VideoUploadProcessorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [productionImageUrl, setProductionImageUrl] = useState<string | null>(null);
  const [defectResult, setDefectResult] = useState<{
    is_defective: boolean;
    defect_rate: number;
    prediction: number;
  } | null>(null);

  // Add local productionState
  const [localProductionState, setLocalProductionState] = useState<ProductionState>({
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

  // Handle video upload
  const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('video/')) {
      setUploadError('Please select a valid video file.');
      return;
    }

    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
      setUploadError('Video file size must be less than 100MB.');
      return;
    }

    setUploadError(null);
    setUploadedVideo(file);
    setVideoUrl(URL.createObjectURL(file));
    setIsPlaying(false);
    setProcessingProgress(0);
  };

  // Handle video play/pause
  const togglePlayPause = () => {
    if (!videoRef.current) return;
    
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Handle video reset
  const resetVideo = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.pause();
      setIsPlaying(false);
      setProcessingProgress(0);
    }
  };

  // Process video frames for gesture detection
  useEffect(() => {
    if (!isPlaying || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;

    let frameCount = 0;
    let totalFrames = 0;
    let processingInterval: number;

    const processVideoFrame = async () => {
      if (video.paused || video.ended) {
        setIsPlaying(false);
        return;
      }

      try {
        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to blob for API call
        canvas.toBlob(async (blob) => {
          if (!blob) return;

          try {
            setIsProcessing(true);
            
            // Send frame for processing to backend
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

            // Update processing progress
            frameCount++;
            if (totalFrames > 0) {
              setProcessingProgress((frameCount / totalFrames) * 100);
            }

            // Update local production state
            setLocalProductionState((prev) => ({
              ...prev,
              productionCount: result.production_count ?? prev.productionCount,
              batchSize: result.batch_size ?? prev.batchSize,
              defectCount: result.defect_count ?? prev.defectCount,
              qualityScore: result.quality_score ?? prev.qualityScore,
              fps: result.fps ?? prev.fps,
              currentGesture: result.gesture ?? prev.currentGesture,
            }));

          } catch (error) {
            console.error('Error processing frame:', error);
          } finally {
            setIsProcessing(false);
          }
        }, 'image/jpeg', 0.8);

      } catch (error) {
        console.error('Error in frame processing:', error);
      }
    };

    // Calculate total frames for progress tracking
    const calculateTotalFrames = () => {
      if (video.duration && video.videoWidth) {
        // Estimate total frames based on duration and typical frame rate
        totalFrames = Math.floor(video.duration * 10); // Assuming 10 FPS processing
      }
    };

    // Start processing when video starts playing
    const handlePlay = () => {
      calculateTotalFrames();
      processingInterval = setInterval(processVideoFrame, 100); // Process at 10 FPS
    };

    const handlePause = () => {
      if (processingInterval) {
        clearInterval(processingInterval);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
      if (processingInterval) {
        clearInterval(processingInterval);
      }
    };

    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
      if (processingInterval) {
        clearInterval(processingInterval);
      }
    };
  }, [isPlaying, currentGesture, onGestureDetected]);

  // Cleanup video URL on unmount
  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  // Poll for metrics and production image every second
  useEffect(() => {
    let interval: number | undefined;
    if (!isPlaying) {
      interval = window.setInterval(async () => {
        try {
          const metrics = await getProductionMetrics();
          setLocalProductionState((prev) => ({
            ...prev,
            productionCount: metrics.production_count ?? prev.productionCount,
            defectCount: metrics.defect_count ?? prev.defectCount,
            batchSize: metrics.batch_size ?? prev.batchSize,
            qualityScore: metrics.quality_score ?? prev.qualityScore,
            fps: metrics.fps ?? prev.fps,
            currentGesture: metrics.current_gesture ?? prev.currentGesture,
            machineStatus: metrics.machine_status ?? prev.machineStatus,
            totalProduction: metrics.total_production ?? prev.totalProduction,
            productionMode: metrics.production_mode ?? prev.productionMode,
            simulationMode: metrics.simulation_mode ?? prev.simulationMode,
            testMode: metrics.test_mode ?? prev.testMode,
            emergencyMode: metrics.emergency_mode ?? prev.emergencyMode,
            emergencyResetProgress: metrics.emergency_reset_progress ?? prev.emergencyResetProgress,
          }));
          // Update production image
          const imgBlob = await getCurrentProductionImage();
          if (imgBlob) {
            const imgUrl = URL.createObjectURL(imgBlob);
            setProductionImageUrl(imgUrl);
          }
        } catch (err) {
          // Optionally handle error
        }
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying]);

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
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Metrics and Production Image Panel (always visible) */}
      <div>
        <div className="space-y-2 mb-6">
          <h4 className="text-sm font-medium text-gray-700">Production Image</h4>
          <div className="relative mb-2">
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
          </div>
          <h4 className="text-sm font-medium text-gray-700">Live Production Metrics</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-blue-50 p-2 rounded">
              <div className="text-blue-800 font-medium">Production Count</div>
              <div className="text-blue-600">{localProductionState.productionCount ?? 'N/A'}</div>
            </div>
            <div className="bg-red-50 p-2 rounded">
              <div className="text-red-800 font-medium">Defect Count</div>
              <div className="text-red-600">{localProductionState.defectCount ?? 'N/A'}</div>
            </div>
            <div className="bg-green-50 p-2 rounded">
              <div className="text-green-800 font-medium">Batch Size</div>
              <div className="text-green-600">{localProductionState.batchSize ?? 'N/A'}</div>
            </div>
            <div className="bg-purple-50 p-2 rounded">
              <div className="text-purple-800 font-medium">Quality Score</div>
              <div className="text-purple-600">{localProductionState.qualityScore ?? 'N/A'}%</div>
            </div>
            <div className="bg-yellow-50 p-2 rounded">
              <div className="text-yellow-800 font-medium">FPS</div>
              <div className="text-yellow-600">{localProductionState.fps ?? 'N/A'}</div>
            </div>
            <div className="bg-pink-50 p-2 rounded">
              <div className="text-pink-800 font-medium">Current Gesture</div>
              <div className="text-pink-600">{localProductionState.currentGesture ?? 'N/A'}</div>
            </div>
            <div className="bg-gray-50 p-2 rounded">
              <div className="text-gray-800 font-medium">Machine Status</div>
              <div className="text-gray-600">{localProductionState.machineStatus ?? 'N/A'}</div>
            </div>
            <div className="bg-indigo-50 p-2 rounded">
              <div className="text-indigo-800 font-medium">Total Production</div>
              <div className="text-indigo-600">{localProductionState.totalProduction ?? 'N/A'}</div>
            </div>
            {/* <div className="bg-blue-100 p-2 rounded">
              <div className="text-blue-900 font-medium">Production Mode</div>
              <div className="text-blue-700">{localProductionState.productionMode ? 'ON' : 'OFF'}</div>
            </div>
            <div className="bg-green-100 p-2 rounded">
              <div className="text-green-900 font-medium">Simulation Mode</div>
              <div className="text-green-700">{localProductionState.simulationMode ? 'ON' : 'OFF'}</div>
            </div>
            <div className="bg-purple-100 p-2 rounded">
              <div className="text-purple-900 font-medium">Test Mode</div>
              <div className="text-purple-700">{localProductionState.testMode ? 'ON' : 'OFF'}</div>
            </div>
            <div className="bg-red-100 p-2 rounded">
              <div className="text-red-900 font-medium">Emergency Mode</div>
              <div className="text-red-700">{localProductionState.emergencyMode ? 'ON' : 'OFF'}</div>
            </div> */}
            <div className="bg-orange-100 p-2 rounded">
              <div className="text-orange-900 font-medium">Emergency Reset Progress</div>
              <div className="text-orange-700">{localProductionState.emergencyResetProgress ?? 'N/A'}%</div>
            </div>
          </div>
        </div>
      </div>
      {/* Video Upload/Player Panel (conditional) */}
      <div>
        {!uploadedVideo ? (
          // Upload Section
          <div className="text-center py-8">
            <div className="max-w-md mx-auto">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 hover:border-blue-400 transition-colors">
                <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <h4 className="text-lg font-medium text-gray-900 mb-2">Upload Video for Gesture Recognition</h4>
                <p className="text-sm text-gray-600 mb-4">
                  Upload a video file containing hand gestures to control production.
                  <br />
                  Supported formats: MP4, AVI, MOV (Max 100MB)
                </p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                >
                  Choose Video File
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  className="hidden"
                />
              </div>
              {uploadError && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center">
                    <AlertCircle className="h-4 w-4 text-red-500 mr-2" />
                    <span className="text-sm text-red-700">{uploadError}</span>
                  </div>
                </div>
              )}
              <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                <h5 className="text-sm font-medium text-gray-700 mb-2">Gesture Instructions</h5>
                <div className="grid grid-cols-1 gap-2 text-xs text-gray-600">
                  <div className="flex items-center">
                    <span className="mr-2">✌️</span>
                    <span>Peace Sign: Start Production</span>
                  </div>
                  <div className="flex items-center">
                    <span className="mr-2">✋</span>
                    <span>Palm: Quality Check</span>
                  </div>
                  <div className="flex items-center">
                    <span className="mr-2">✊</span>
                    <span>Fist: Emergency Stop</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          // Video Player Section
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Uploaded Video</h4>
            <div className="relative">
              <video
                ref={videoRef}
                src={videoUrl!}
                className="w-full h-64 object-cover rounded-lg border border-gray-300"
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
              />
              <canvas
                ref={canvasRef}
                className="hidden"
              />
              {/* Video Controls Overlay */}
              <div className="absolute bottom-2 left-2 right-2 flex justify-center space-x-2">
                <button
                  onClick={togglePlayPause}
                  className="bg-black bg-opacity-75 text-white p-2 rounded-full hover:bg-opacity-90 transition-colors"
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <button
                  onClick={resetVideo}
                  className="bg-black bg-opacity-75 text-white p-2 rounded-full hover:bg-opacity-90 transition-colors"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>
              {/* Gesture Overlay */}
              {currentGesture && (
                <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-sm">
                  Gesture: {currentGesture.toUpperCase()}
                </div>
              )}
              {/* Processing Indicator */}
              {isProcessing && (
                <div className="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-sm animate-pulse">
                  Processing...
                </div>
              )}
            </div>
            {/* Processing Progress */}
            {isPlaying && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-gray-600">
                  <span>Processing Progress</span>
                  <span>{processingProgress.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${processingProgress}%` }}
                  ></div>
                </div>
              </div>
            )}
            {/* Upload New Video Button */}
            <button
              onClick={() => {
                setUploadedVideo(null);
                setVideoUrl(null);
                setIsPlaying(false);
                setProcessingProgress(0);
                setProductionImageUrl(null);
                setDefectResult(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
              className="w-full mt-2 bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              Upload New Video
            </button>
          </div>
        )}
        {/* Instructions */}
        {/* <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <h5 className="text-sm font-medium text-gray-700 mb-2">How to Use</h5>
          <div className="text-xs text-gray-600 space-y-1">
            <div>1. Upload a video file containing hand gestures</div>
            <div>2. Use the video controls to play/pause/reset the video</div>
            <div>3. As the video plays, gestures will be recognized and processed</div>
            <div>4. Production will continue based on detected gestures</div>
          </div>
        </div> */}
      </div>
    </div>
  );
} 
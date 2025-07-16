import React, { useState, useRef, useEffect } from 'react';
import { Upload, Search, Download, Settings, BarChart3, Eye, EyeOff } from 'lucide-react';
import { detectImage } from '../api';

interface DetectionResult {
  bbox: [number, number, number, number];
  confidence: number;
  model: string;
  area: number;
}

interface ModelStats {
  count: number;
  avg_confidence: number;
  min_confidence: number;
  max_confidence: number;
}

interface DetectionMetrics {
  total_detections: number;
  enabled_models: number;
  total_models: number;
  model_stats: Record<string, ModelStats>;
  processing_time: number;
  image_size: { width: number; height: number };
}

export default function DetectionPage() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
  const [metrics, setMetrics] = useState<DetectionMetrics | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showMetrics, setShowMetrics] = useState(true);
  const [detectionSettings, setDetectionSettings] = useState({
    confidence: 0.25,
    iou: 0.45,
    motion_threshold: 30.0,
    motion_detection_enabled: true
  });
  const [enabledModels, setEnabledModels] = useState<Record<string, boolean>>({
    'cylinder_detector': true,
    'cylinder_detector2': true,
    'cylinder_detector3': true,
    'cylinder_detector4': true,
    'cylinder_detector5': true,
    'cylinder_detector6': true,
    'cylinder_detector7': true
  });
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const modelColors = {
    'cylinder_detector': '#00FF00',
    'cylinder_detector2': '#FF0000',
    'cylinder_detector3': '#0000FF',
    'cylinder_detector4': '#FFFF00',
    'cylinder_detector5': '#FF00FF',
    'cylinder_detector6': '#00FFFF',
    'cylinder_detector7': '#800080'
  };

  // Draw detections on canvas when results change
  useEffect(() => {
    if (canvasRef.current && imagePreview && detectionResults.length > 0) {
      drawDetectionsOnCanvas(canvasRef.current);
    }
  }, [detectionResults, imagePreview]);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setDetectionResults([]);
      setMetrics(null);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setDetectionResults([]);
      setMetrics(null);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const runDetection = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    
    try {
      const data = await detectImage(selectedImage, detectionSettings, enabledModels);
      setDetectionResults(data.detections || []);
      setMetrics(data.metrics || null);
    } catch (error) {
      console.error('Error running detection:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleModel = (modelName: string) => {
    setEnabledModels(prev => ({
      ...prev,
      [modelName]: !prev[modelName]
    }));
  };

  const toggleAllModels = () => {
    const allEnabled = Object.values(enabledModels).every(enabled => enabled);
    const newState = Object.keys(enabledModels).reduce((acc, model) => {
      acc[model] = !allEnabled;
      return acc;
    }, {} as Record<string, boolean>);
    setEnabledModels(newState);
  };

  const downloadResults = () => {
    if (!imagePreview || detectionResults.length === 0) return;
    
    const canvas = document.createElement('canvas');
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        
        // Draw detection boxes
        detectionResults.forEach(detection => {
          const [x1, y1, x2, y2] = detection.bbox;
          const color = modelColors[detection.model as keyof typeof modelColors] || '#FF0000';
          
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          
          // Draw label
          ctx.fillStyle = color;
          ctx.font = '14px Arial';
          ctx.fillText(`${detection.model} (${(detection.confidence * 100).toFixed(1)}%)`, x1, y1 - 5);
        });
        
        // Download the image
        const link = document.createElement('a');
        link.download = 'detection_result.png';
        link.href = canvas.toDataURL();
        link.click();
      }
    };
    img.src = imagePreview;
  };

  const drawDetectionsOnCanvas = (canvas: HTMLCanvasElement) => {
    if (!imagePreview || detectionResults.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      detectionResults.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;
        const color = modelColors[detection.model as keyof typeof modelColors] || '#FF0000';
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      });
    };
    img.src = imagePreview;
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Object Detection</h1>
        <p className="text-gray-600">Upload an image to run ensemble detection using multiple YOLO models</p>
      </div>

      {/* Top Section - Image Upload and Settings Side by Side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Left Column - Image Upload and Results */}
        <div className="space-y-6">
          {/* Image Upload */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Upload className="w-5 h-5 mr-2" />
              Image Upload
            </h2>
            
            <div
              className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              {imagePreview ? (
                <div className="space-y-4">
                  <img 
                    src={imagePreview} 
                    alt="Preview" 
                    className="max-w-full h-auto max-h-48 mx-auto rounded"
                  />
                  <p className="text-sm text-gray-600">
                    Click to change image or drag a new one
                  </p>
                </div>
              ) : (
                <div>
                  <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600 mb-2">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-gray-500">
                    PNG, JPG, JPEG up to 10MB
                  </p>
                </div>
              )}
            </div>
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              aria-label="Upload image file"
            />
            
            {selectedImage && (
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-700">
                  <strong>File:</strong> {selectedImage.name}
                </p>
                <p className="text-sm text-gray-700">
                  <strong>Size:</strong> {(selectedImage.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>

          {/* Detection Results */}
          {imagePreview && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Detection Results</h2>
              
              <div className="relative">
                {detectionResults.length > 0 ? (
                  <>
                    <canvas
                      ref={canvasRef}
                      className="max-w-full h-auto border rounded-lg"
                      style={{ display: 'block' }}
                    />
                    <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                      <p className="text-sm text-blue-800">
                        <strong>{detectionResults.length}</strong> objects detected
                      </p>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <Search className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                    <p className="text-sm">Click "Run Detection" to process the image</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Settings */}
        <div className="space-y-6">
          {/* Detection Settings */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Settings className="w-5 h-5 mr-2" />
              Detection Settings
            </h2>
            
            <div className="space-y-4">
              <div>
                <label htmlFor="confidence-threshold" className="block text-sm font-medium text-gray-700 mb-1">
                  Confidence Threshold: {detectionSettings.confidence.toFixed(2)}
                </label>
                <input
                  id="confidence-threshold"
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.01"
                  value={detectionSettings.confidence}
                  onChange={(e) => setDetectionSettings(prev => ({
                    ...prev,
                    confidence: parseFloat(e.target.value)
                  }))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  aria-label="Confidence threshold slider"
                />
              </div>
              
              <div>
                <label htmlFor="iou-threshold" className="block text-sm font-medium text-gray-700 mb-1">
                  IoU Threshold: {detectionSettings.iou.toFixed(2)}
                </label>
                <input
                  id="iou-threshold"
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.01"
                  value={detectionSettings.iou}
                  onChange={(e) => setDetectionSettings(prev => ({
                    ...prev,
                    iou: parseFloat(e.target.value)
                  }))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  aria-label="IoU threshold slider"
                />
              </div>
              
              <div>
                <label htmlFor="motion-threshold" className="block text-sm font-medium text-gray-700 mb-1">
                  Motion Threshold: {detectionSettings.motion_threshold.toFixed(1)}
                </label>
                <input
                  id="motion-threshold"
                  type="range"
                  min="5"
                  max="100"
                  step="1"
                  value={detectionSettings.motion_threshold}
                  onChange={(e) => setDetectionSettings(prev => ({
                    ...prev,
                    motion_threshold: parseFloat(e.target.value)
                  }))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  aria-label="Motion threshold slider"
                />
              </div>
              
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="motion-detection"
                  checked={detectionSettings.motion_detection_enabled}
                  onChange={(e) => setDetectionSettings(prev => ({
                    ...prev,
                    motion_detection_enabled: e.target.checked
                  }))}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="motion-detection" className="ml-2 block text-sm text-gray-700">
                  Enable Motion Detection
                </label>
              </div>
            </div>
          </div>

          {/* Model Selection */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Model Selection
              </h2>
              <button
                onClick={toggleAllModels}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Toggle All
              </button>
            </div>
            
            <div className="space-y-2">
              {Object.entries(enabledModels).map(([modelName, enabled]) => (
                <div key={modelName} className="flex items-center">
                  <input
                    type="checkbox"
                    id={modelName}
                    checked={enabled}
                    onChange={() => toggleModel(modelName)}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <label htmlFor={modelName} className="ml-2 block text-sm text-gray-700 flex items-center">
                    <div 
                      className="w-3 h-3 rounded-full mr-2"
                      style={{ backgroundColor: modelColors[modelName as keyof typeof modelColors] }}
                    />
                    {modelName.replace('_', ' ')}
                  </label>
                </div>
              ))}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="space-y-3">
              <button
                onClick={runDetection}
                disabled={!selectedImage || isProcessing}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center font-medium"
              >
                {isProcessing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4 mr-2" />
                    Run Detection
                  </>
                )}
              </button>
              
              {detectionResults.length > 0 && (
                <button
                  onClick={downloadResults}
                  className="w-full bg-green-600 text-white py-3 px-4 rounded-md hover:bg-green-700 flex items-center justify-center font-medium"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download Results
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section - Metrics Dashboard */}
      <div className="space-y-6">
        {/* Metrics Toggle */}
        <div className="flex justify-end">
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className="flex items-center text-sm text-gray-600 hover:text-gray-800 bg-white px-4 py-2 rounded-lg shadow-md"
          >
            {showMetrics ? <EyeOff className="w-4 h-4 mr-1" /> : <Eye className="w-4 h-4 mr-1" />}
            {showMetrics ? 'Hide' : 'Show'} Metrics
          </button>
        </div>

        {/* Metrics Dashboard */}
        {showMetrics && metrics && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Detection Metrics Dashboard</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-blue-600 font-medium">Total Detections</p>
                <p className="text-2xl font-bold text-blue-900">{metrics.total_detections}</p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <p className="text-sm text-green-600 font-medium">Enabled Models</p>
                <p className="text-2xl font-bold text-green-900">{metrics.enabled_models}/{metrics.total_models}</p>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <p className="text-sm text-purple-600 font-medium">Processing Time</p>
                <p className="text-2xl font-bold text-purple-900">{metrics.processing_time.toFixed(2)}s</p>
              </div>
              <div className="bg-orange-50 p-4 rounded-lg">
                <p className="text-sm text-orange-600 font-medium">Image Size</p>
                <p className="text-2xl font-bold text-orange-900">{metrics.image_size.width}×{metrics.image_size.height}</p>
              </div>
            </div>

            {/* Model Performance */}
            <div>
              <h3 className="text-md font-semibold text-gray-900 mb-3">Model Performance</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {Object.entries(metrics.model_stats).map(([modelName, stats]) => (
                  <div key={modelName} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center">
                      <div 
                        className="w-3 h-3 rounded-full mr-3"
                        style={{ backgroundColor: modelColors[modelName as keyof typeof modelColors] }}
                      />
                      <span className="text-sm font-medium text-gray-700">
                        {modelName.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">
                        Detections: <span className="font-medium">{stats.count}</span>
                      </p>
                      <p className="text-sm text-gray-600">
                        Avg Confidence: <span className="font-medium">{(stats.avg_confidence * 100).toFixed(1)}%</span>
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>


    </div>
  );
} 
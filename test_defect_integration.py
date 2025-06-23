#!/usr/bin/env python3
"""
Test script for defect detection model integration
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

def test_defect_model():
    """Test the defect detection model"""
    
    # Check if model file exists
    model_path = "defect_detector_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using train_defect_model.py")
        return False
    
    try:
        # Load the model
        print("🔄 Loading defect detection model...")
        model = load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test with different types of images
        test_cases = [
            ("Random noise", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)),
            ("Solid color", np.full((480, 640, 3), 128, dtype=np.uint8)),
            ("Gradient", np.tile(np.linspace(0, 255, 640).astype(np.uint8), (480, 1, 3)))
        ]
        
        print("\n🧪 Testing defect detection...")
        for name, test_image in test_cases:
            # Resize to match model input
            img = cv2.resize(test_image, (128, 128))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            prediction = model.predict(img, verbose=0)[0][0]
            defect_rate = prediction * 100
            is_defective = prediction >= 0.5
            
            print(f"   {name}:")
            print(f"     - Defect Rate: {defect_rate:.2f}%")
            print(f"     - Is Defective: {is_defective}")
            print(f"     - Raw Prediction: {prediction:.4f}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def test_model_integration():
    """Test the model integration with the smart factory controller"""
    try:
        # Import the controller
        from smart_factory_control import SmartFactoryController
        
        print("🔄 Testing SmartFactoryController integration...")
        controller = SmartFactoryController()
        
        # Test the predict_defect method
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = controller.predict_defect(test_image)
        
        print("✅ Integration test successful!")
        print(f"   - Defect Rate: {result['defect_rate']:.2f}%")
        print(f"   - Is Defective: {result['is_defective']}")
        print(f"   - Raw Prediction: {result['prediction']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Defect Detection Model Integration")
    print("=" * 50)
    
    # Test 1: Basic model functionality
    print("\n1. Testing basic model functionality...")
    if test_defect_model():
        print("✅ Basic model test passed!")
    else:
        print("❌ Basic model test failed!")
        exit(1)
    
    # Test 2: Integration with SmartFactoryController
    print("\n2. Testing integration with SmartFactoryController...")
    if test_model_integration():
        print("✅ Integration test passed!")
    else:
        print("❌ Integration test failed!")
        exit(1)
    
    print("\n🎉 All tests passed! The defect detection model is ready for use.")
    print("\nTo use the model:")
    print("1. Run the smart factory control system: streamlit run smart_factory_control.py")
    print("2. Show a PALM gesture to trigger quality check")
    print("3. The system will analyze the current camera frame for defects")
    print("4. Results will be logged to safety_check_records.csv") 
#!/usr/bin/env python3
"""
Test script for defect detection model integration
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, cohen_kappa_score, jaccard_score
from imblearn.metrics import geometric_mean_score

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
            ("Gradient", np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1)).reshape(480, 640, 1).repeat(3, axis=2))
        ]
        
        print("\n🧪 Testing defect detection...")
        for name, test_image in test_cases:
            # Defensive check for empty or invalid images
            if test_image is None or test_image.size == 0:
                print(f"❌ Skipping {name}: empty or invalid image")
                continue
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

def evaluate_model():
    """Evaluate the defect detection model"""
    
    # Paths and parameters
    binary_dataset = "binary_defect_dataset"
    img_size = (128, 128)
    batch_size = 32

    # Load model
    model = load_model("defect_detector_model.h5")

    # Data generator for validation/test
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        binary_dataset,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    # Get all validation data and predictions
    y_true = []
    y_pred = []
    y_prob = []
    val_gen.reset()
    for i in range(len(val_gen)):
        X, y = val_gen[i]
        preds = model.predict(X).flatten()
        y_prob.extend(preds)
        y_pred.extend((preds > 0.5).astype(int))
        y_true.extend(y.astype(int))
        if (i+1)*val_gen.batch_size >= val_gen.samples:
            break
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 1. Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # 3. ROC Curve + AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()
    print(f"AUC Score: {auc_score:.4f}")

    # 4. Cohen's Kappa Score
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa Score: {kappa:.4f}")

    # 5. G-Mean Score
    gmean = geometric_mean_score(y_true, y_pred)
    print(f"G-Mean Score: {gmean:.4f}")

    # 6. Mean IoU Score
    iou = jaccard_score(y_true, y_pred, average='binary')
    print(f"Mean IoU Score: {iou:.4f}")

    # 7. Accuracy vs Loss Curve (if you have history)
    # If you saved the training history (e.g., as a .npy or .pkl file), you can plot it here.
    # Otherwise, this is only available during training.

    print("All metrics and plots have been generated and saved.")

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
    
    # Test 3: Model evaluation
    print("\n3. Evaluating the model...")
    evaluate_model()
    
    print("\n🎉 All tests passed! The defect detection model is ready for use.")
    print("\nTo use the model:")
    print("1. Run the smart factory control system: streamlit run smart_factory_control.py")
    print("2. Show a PALM gesture to trigger quality check")
    print("3. The system will analyze the current camera frame for defects")
    print("4. Results will be logged to safety_check_records.csv") 
#!/usr/bin/env python3
"""
Simple test script to verify smart factory control integration with frontend
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:5000"

def test_smart_factory_status():
    """Test smart factory status endpoint"""
    print("🔍 Testing smart factory status...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/smart-factory/status")
        if response.status_code == 200:
            status = response.json()
            print("✅ Smart factory status retrieved successfully:")
            print(f"   - Production Mode: {status.get('production_mode', False)}")
            print(f"   - Test Mode: {status.get('test_mode', False)}")
            print(f"   - Simulation Mode: {status.get('simulation_mode', False)}")
            print(f"   - Emergency Mode: {status.get('emergency_mode', False)}")
            print(f"   - Machine Status: {status.get('machine_status', 'Unknown')}")
            return True
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing status: {e}")
        return False

def test_smart_factory_metrics():
    """Test smart factory metrics endpoint"""
    print("\n🔍 Testing smart factory metrics...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/smart-factory/metrics")
        if response.status_code == 200:
            metrics = response.json()
            print("✅ Smart factory metrics retrieved successfully:")
            print(f"   - Production Count: {metrics.get('production_count', 0)}")
            print(f"   - Batch Size: {metrics.get('batch_size', 0)}")
            print(f"   - Defect Count: {metrics.get('defect_count', 0)}")
            print(f"   - Quality Score: {metrics.get('quality_score', 0):.1f}%")
            print(f"   - FPS: {metrics.get('fps', 0)}")
            print(f"   - Current Gesture: {metrics.get('current_gesture', 'None')}")
            return True
        else:
            print(f"❌ Failed to get metrics: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing metrics: {e}")
        return False

def test_production_image():
    """Test production image endpoint"""
    print("\n🔍 Testing production image endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/smart-factory/production-image")
        if response.status_code == 200:
            print("✅ Production image retrieved successfully")
            print(f"   - Image size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Failed to get production image: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing production image: {e}")
        return False

def test_defect_result():
    """Test defect result endpoint"""
    print("\n🔍 Testing defect result endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/smart-factory/defect-result")
        if response.status_code == 200:
            defect_result = response.json()
            print("✅ Defect result retrieved successfully:")
            print(f"   - Is Defective: {defect_result.get('is_defective', False)}")
            print(f"   - Defect Rate: {defect_result.get('defect_rate', 0):.2f}%")
            print(f"   - Prediction: {defect_result.get('prediction', 0):.4f}")
            return True
        else:
            print(f"❌ Failed to get defect result: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing defect result: {e}")
        return False

def test_production_control():
    """Test production control endpoints"""
    print("\n🔍 Testing production control...")
    
    # Test start production
    try:
        response = requests.post(f"{API_BASE_URL}/api/production/start")
        if response.status_code == 200:
            print("✅ Start production successful")
        else:
            print(f"❌ Failed to start production: {response.status_code}")
    except Exception as e:
        print(f"❌ Error starting production: {e}")
    
    # Wait a moment
    time.sleep(1)
    
    # Test quality check
    try:
        response = requests.post(f"{API_BASE_URL}/api/production/quality_check")
        if response.status_code == 200:
            print("✅ Quality check successful")
        else:
            print(f"❌ Failed to perform quality check: {response.status_code}")
    except Exception as e:
        print(f"❌ Error performing quality check: {e}")
    
    # Wait a moment
    time.sleep(1)
    
    # Test emergency stop
    try:
        response = requests.post(f"{API_BASE_URL}/api/production/emergency_stop")
        if response.status_code == 200:
            print("✅ Emergency stop successful")
        else:
            print(f"❌ Failed to perform emergency stop: {response.status_code}")
    except Exception as e:
        print(f"❌ Error performing emergency stop: {e}")

def test_system_metrics():
    """Test system metrics endpoint"""
    print("\n🔍 Testing system metrics...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/metrics")
        if response.status_code == 200:
            metrics = response.json()
            print("✅ System metrics retrieved successfully:")
            print(f"   - Total Production: {metrics.get('totalProduction', 0)}")
            print(f"   - Quality Score: {metrics.get('currentQualityScore', 0):.1f}%")
            print(f"   - Defect Rate: {metrics.get('activeDefectRate', 0):.2f}%")
            print(f"   - Machine Status: {metrics.get('machineStatus', 'Unknown')}")
            return True
        else:
            print(f"❌ Failed to get system metrics: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing system metrics: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Starting Smart Factory Control Integration Tests")
    print("=" * 60)
    
    # Test basic endpoints
    test_smart_factory_status()
    test_smart_factory_metrics()
    test_system_metrics()
    test_production_image()
    test_defect_result()
    test_production_control()
    
    print("\n" + "=" * 60)
    print("✅ Integration tests completed!")
    print("\n📋 Next steps:")
    print("1. Start the Flask backend: python app.py")
    print("2. Start the React frontend: cd SFC_UI && npm run dev")
    print("3. Open the frontend in your browser")
    print("4. Navigate to the Production Control page")
    print("5. Test the real-time video processing and gesture controls")

if __name__ == "__main__":
    main() 
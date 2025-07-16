# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from smart_factory_control import SmartFactoryController
import numpy as np
import cv2
import datetime
import os
import json
import matplotlib.pyplot as plt
import mediapipe as mp
import time
from collections import defaultdict
import threading
import csv
from tensorflow.keras.models import load_model
import random
import glob
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive for server environments
import io
import traceback
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import google.generativeai as genai
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import atexit
import tempfile
from PIL import Image as PILImage

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Global list to track temporary files for cleanup
temp_files = []

def cleanup_temp_files():
    """Clean up temporary files on exit"""
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"🧹 Cleaned up temporary file: {temp_file}")
        except Exception as e:
            print(f"⚠️ Could not remove temporary file {temp_file}: {e}")

# Register cleanup function to run on exit
atexit.register(cleanup_temp_files)

app = Flask(__name__)
CORS(app)

# Initialize the controller (singleton for now)
controller = SmartFactoryController()

# Initialize ensemble detector
ensemble_detector = None

def get_ensemble_detector():
    """Get or create ensemble detector instance"""
    global ensemble_detector
    if ensemble_detector is None:
        try:
            from live_ensemble_detection import LiveEnsembleDetector
            ensemble_detector = LiveEnsembleDetector()
            if not ensemble_detector.scan_and_load_models():
                print("❌ Failed to load ensemble detection models")
                return None
            print("✅ Ensemble detector initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing ensemble detector: {e}")
            return None
    return ensemble_detector

def get_system_metrics():
    """Get system metrics in the format expected by frontend"""
    return {
        'totalProduction': getattr(controller, 'total_production', getattr(controller, 'production_count', 0)),
        'currentQualityScore': getattr(controller, 'quality_score', 0),
        'activeDefectRate': (getattr(controller, 'defect_count', 0) / getattr(controller, 'batch_size', 1) * 100) if getattr(controller, 'batch_size', 0) > 0 else 0,
        'machineStatus': getattr(controller, 'machine_status', ''),
        'emergencyEvents': len(getattr(controller, 'emergency_logs', [])) if hasattr(controller, 'emergency_logs') else 0,
        'systemUptime': 142.5,  # Mock uptime for now
        'defectCount': getattr(controller, 'defect_count', 0),
        'batchSize': getattr(controller, 'batch_size', 0),
        'currentGesture': getattr(controller, 'current_gesture', None),
        'fps': getattr(controller, 'fps', 0)
    }

def get_smart_factory_metrics():
    """Get smart factory metrics in the format expected by frontend"""
    return {
        'production_count': getattr(controller, 'production_count', 0),
        'batch_size': getattr(controller, 'batch_size', 0),
        'defect_count': getattr(controller, 'defect_count', 0),
        'quality_score': getattr(controller, 'quality_score', 100.0),
        'fps': getattr(controller, 'fps', 0),
        'current_gesture': getattr(controller, 'current_gesture', None),
        'machine_status': getattr(controller, 'machine_status', 'STANDBY'),
        'emergency_reset_progress': getattr(controller, 'emergency_reset_progress', 0),
        'production_mode': getattr(controller, 'production_mode', False),
        'test_mode': getattr(controller, 'test_mode', False),
        'simulation_mode': getattr(controller, 'simulation_mode', False),
        'emergency_mode': getattr(controller, 'emergency_mode', False),
        'total_production': getattr(controller, 'total_production', 0)
    }

def get_smart_factory_status():
    """Get smart factory status"""
    return {
        'production_mode': getattr(controller, 'production_mode', False),
        'test_mode': getattr(controller, 'test_mode', False),
        'simulation_mode': getattr(controller, 'simulation_mode', False),
        'emergency_mode': getattr(controller, 'emergency_mode', False),
        'machine_status': getattr(controller, 'machine_status', 'STANDBY'),
        'current_gesture': getattr(controller, 'current_gesture', None),
        'emergency_reset_progress': getattr(controller, 'emergency_reset_progress', 0)
    }

def get_safety_records_from_db():
    """Get safety records from MySQL database in the format expected by frontend"""
    records = []
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', ''),
            user=os.getenv('DB_USER', ''),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', '')
        )
        cursor = connection.cursor()
        sql = """
            SELECT batch_date, batch_size, defect_rate, timestamp 
            FROM safety_check_records 
            ORDER BY batch_date DESC, timestamp DESC
        """
        cursor.execute(sql)
        results = cursor.fetchall()
        record_id = 1
        for row in results:
            batch_date, batch_size, defect_rate, timestamp = row
            quality_score = 100 - defect_rate
            status = 'Pass' if quality_score >= 70 else 'Fail'
            record = {
                'id': str(record_id),
                'date': batch_date.strftime('%Y-%m-%d') if hasattr(batch_date, 'strftime') else str(batch_date),
                'batchSize': int(batch_size),
                'defectRate': float(defect_rate),
                'timestamp': str(timestamp),
                'qualityScore': quality_score,
                'status': status
            }
            records.append(record)
            record_id += 1
        cursor.close()
        connection.close()
        print(f"✅ Retrieved {len(records)} safety records from database")
    except Error as e:
        print(f"❌ Database error: {e}")
        return get_safety_records_from_csv()
    except Exception as e:
        print(f"❌ Error retrieving safety records from database: {e}")
        return get_safety_records_from_csv()
    return records

def get_safety_records_from_csv():
    """Get safety records from CSV file (fallback method)"""
    records = []
    try:
        with open(controller.safety_check_records, 'r') as f:
            lines = f.readlines()
        current_date = None
        record_id = 1
        for line in lines:
            line = line.strip()
            if line.startswith('----') and line.endswith('----'):
                current_date = line.replace('----', '').strip()
            elif line and not line.startswith('Batch Size') and ',' in line and current_date:
                parts = line.split(',')
                if len(parts) >= 3:
                    batch_size = int(float(parts[0]))
                    defect_rate = float(parts[1])
                    timestamp = parts[2]
                    quality_score = 100 - defect_rate
                    status = 'Pass' if quality_score >= 70 else 'Fail'
                    record = {
                        'id': str(record_id),
                        'date': current_date,
                        'batchSize': batch_size,
                        'defectRate': defect_rate,
                        'timestamp': timestamp,
                        'qualityScore': quality_score,
                        'status': status
                    }
                    records.append(record)
                    record_id += 1
    except Exception as e:
        print(f"Error reading safety records from CSV: {e}")
    return records

def log_detection_metrics_to_db(total_detections, enabled_models, processing_time, model_stats):
    """Log detection metrics to the database"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', ''),
            user=os.getenv('DB_USER', ''),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', '')
        )
        cursor = connection.cursor()
        
        # Insert into detection_metrics_log
        insert_metrics_sql = """
            INSERT INTO detection_metrics_log (total_detections, enabled_models, processing_time)
            VALUES (%s, %s, %s)
        """
        cursor.execute(insert_metrics_sql, (total_detections, enabled_models, processing_time))
        
        # Get the ID of the inserted record
        detection_metrics_id = cursor.lastrowid
        
        # Insert model performance data
        insert_model_sql = """
            INSERT INTO model_performance_log (detection_metrics_id, model_name, detections, avg_confidence)
            VALUES (%s, %s, %s, %s)
        """
        
        for model_name, stats in model_stats.items():
            cursor.execute(insert_model_sql, (
                detection_metrics_id,
                model_name,
                stats['count'],
                stats['avg_confidence']
            ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print(f"✅ Logged detection metrics to database: {total_detections} detections, {enabled_models} models, {processing_time:.2f}s")
        print(f"📊 Database logging successful! Check your MySQL tables for new entries.")
        print(f"   - detection_metrics_log: Added record with {total_detections} detections")
        print(f"   - model_performance_log: Added {len(model_stats)} model performance records")
        return True
        
    except Error as e:
        print(f"❌ Database error logging detection metrics: {e}")
        return False
    except Exception as e:
        print(f"❌ Error logging detection metrics: {e}")
        return False

def get_detection_metrics_history():
    """Get detection metrics history from database"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', ''),
            user=os.getenv('DB_USER', ''),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', '')
        )
        cursor = connection.cursor()
        
        # Get recent detection metrics with model performance
        sql = """
            SELECT 
                dml.id,
                dml.timestamp,
                dml.total_detections,
                dml.enabled_models,
                dml.processing_time,
                mpl.model_name,
                mpl.detections as model_detections,
                mpl.avg_confidence
            FROM detection_metrics_log dml
            LEFT JOIN model_performance_log mpl ON dml.id = mpl.detection_metrics_id
            ORDER BY dml.timestamp DESC
            LIMIT 100
        """
        cursor.execute(sql)
        results = cursor.fetchall()
        
        # Group by detection session
        detection_sessions = {}
        for row in results:
            (detection_id, timestamp, total_detections, enabled_models, 
             processing_time, model_name, model_detections, avg_confidence) = row
            
            if detection_id not in detection_sessions:
                detection_sessions[detection_id] = {
                    'id': detection_id,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp),
                    'total_detections': total_detections,
                    'enabled_models': enabled_models,
                    'processing_time': processing_time,
                    'models': {}
                }
            
            if model_name:
                detection_sessions[detection_id]['models'][model_name] = {
                    'detections': model_detections,
                    'avg_confidence': avg_confidence
                }
        
        cursor.close()
        connection.close()
        
        return list(detection_sessions.values())
        
    except Error as e:
        print(f"❌ Database error getting detection metrics history: {e}")
        return []
    except Exception as e:
        print(f"❌ Error getting detection metrics history: {e}")
        return []

def get_safety_records_formatted():
    """Get safety records in the format expected by frontend - now from database"""
    return get_safety_records_from_db()

def get_emergency_events_formatted():
    """Get emergency events in the format expected by frontend"""
    events = []
    csv_file = 'emergency_reset_log.csv'
    if not os.path.exists(csv_file):
        return events
    
    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        headers = []
        event_id = 1
        for i, line in enumerate(lines):
            line = line.strip()
            if i == 0:
                headers = [h.strip() for h in line.split(',')]
            elif line:
                parts = line.split(',')
                if len(parts) == len(headers):
                    log = {headers[j]: parts[j] for j in range(len(headers))}
                    
                    # Convert to frontend format
                    event = {
                        'id': str(event_id),
                        'timestamp': log.get('timestamp', ''),
                        'eventType': log.get('event_type', 'EMERGENCY_STOP'),
                        'totalProduction': int(log.get('total_production', 0)),
                        'productionCount': int(log.get('production_count', 0)),
                        'defectCount': int(log.get('defect_count', 0)),
                        'defectRate': float(log.get('defect_rate', 0)),
                        'qualityScore': float(log.get('quality_score', 0)),
                        'batchSize': int(log.get('batch_size', 0)),
                        'machineStatus': log.get('machine_status', 'EMERGENCY')
                    }
                    events.append(event)
                    event_id += 1
    except Exception as e:
        print(f"Error reading emergency logs: {e}")
    return events

def get_alerts():
    """Get system alerts in the format expected by frontend"""
    alerts = [
        {
            'id': '1',
            'type': 'warning',
            'message': 'Defect rate approaching threshold (2.5%)',
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'id': '2',
            'type': 'info',
            'message': f'Batch {controller.total_production} completed successfully',
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'id': '3',
            'type': 'success',
            'message': f'Quality check passed - {controller.quality_score:.1f}% score',
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    ]
    return alerts

def to_py(val):
    if isinstance(val, (np.generic, np.ndarray)):
        return val.item()
    return val

# Frontend API endpoints to get metrics
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics for frontend"""
    return jsonify(get_system_metrics())

@app.route('/api/smart-factory/status', methods=['GET'])
def get_smart_factory_status_endpoint():
    """Get smart factory status for frontend"""
    return jsonify(get_smart_factory_status())

@app.route('/api/smart-factory/metrics', methods=['GET'])
def get_smart_factory_metrics_endpoint():
    """Get smart factory metrics for frontend"""
    return jsonify(get_smart_factory_metrics())

@app.route('/api/smart-factory/production-image', methods=['GET'])
def get_current_production_image():
    """Get current production image from smart factory controller"""
    try:
        # Get current production image from controller
        production_image = controller.get_current_production_image()
        if production_image is not None:
            # Convert numpy array to bytes
            _, buffer = cv2.imencode('.jpg', production_image)
            io_buf = io.BytesIO(buffer)
            io_buf.seek(0)
            return send_file(io_buf, mimetype='image/jpeg')
        else:
            # Return a default image or error
            return jsonify({'error': 'No production image available'}), 404
    except Exception as e:
        print(f"Error getting production image: {e}")
        return jsonify({'error': 'Failed to get production image'}), 500

@app.route('/api/smart-factory/defect-result', methods=['GET'])
def get_current_defect_result():
    """Get current defect result from smart factory controller"""
    try:
        # Get current production image for defect analysis
        production_image = controller.get_current_production_image()
        if production_image is not None:
            # Run defect detection
            defect_result = controller.predict_defect(production_image)
            return jsonify(defect_result)
        else:
            return jsonify({'error': 'No production image available for defect analysis'}), 404
    except Exception as e:
        print(f"Error getting defect result: {e}")
        return jsonify({'error': 'Failed to get defect result'}), 500

@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    """Process a video frame for gesture detection and production monitoring"""
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        frame_file = request.files['frame']
        frame_bytes = frame_file.read()
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        controller.process_frame(frame)
        gesture = controller.current_gesture
        metrics = get_smart_factory_metrics()
        # Only return gesture and metrics for instant recognition
        response = {
            'gesture': gesture,
            'metrics': metrics
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process frame'}), 500

@app.route('/api/safety-records', methods=['GET'])
def get_safety_records():
    """Get safety records for frontend"""
    return jsonify(get_safety_records_formatted())

@app.route('/api/emergency-logs', methods=['GET'])
def get_emergency_logs():
    """Get emergency logs for frontend"""
    return jsonify(get_emergency_events_formatted())

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for frontend from database"""
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Date parameter required'}), 400
    
    batch_sizes, defect_rates = [], []
    try:
        # Try to get data from database first
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', ''),
            user=os.getenv('DB_USER', ''),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', '')
        )
        cursor = connection.cursor()
        
        # Query to get data for specific date
        sql = """
            SELECT batch_size, defect_rate 
            FROM safety_check_records 
            WHERE DATE(batch_date) = %s 
            ORDER BY timestamp
        """
        cursor.execute(sql, (date_str,))
        results = cursor.fetchall()
        
        for row in results:
            batch_size, defect_rate = row
            batch_sizes.append(int(batch_size))
            defect_rates.append(float(defect_rate))
            
        cursor.close()
        connection.close()
        print(f"✅ Retrieved {len(batch_sizes)} records from database for {date_str}")
        
    except Error as e:
        print(f"❌ Database error for analytics: {e}")
        # Fallback to CSV
        batch_sizes, defect_rates = get_analytics_from_csv(date_str)
    except Exception as e:
        print(f"❌ Error retrieving analytics from database: {e}")
        # Fallback to CSV
        batch_sizes, defect_rates = get_analytics_from_csv(date_str)
    
    return jsonify({
        'batch_sizes': batch_sizes,
        'defect_rates': defect_rates,
        'mean_defect_rate': sum(defect_rates) / len(defect_rates) if defect_rates else 0,
        'avg_batch_size': sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
    })

def get_analytics_from_csv(date_str):
    """Get analytics data from CSV file (fallback method)"""
    batch_sizes, defect_rates = [], []
    try:
        with open(controller.safety_check_records, 'r') as f:
            lines = f.readlines()
        section_start = None
        section_end = None
        for i, line in enumerate(lines):
            if line.strip() == f"---- {date_str} ----":
                section_start = i
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("----"):
                        section_end = j
                        break
                if section_end is None:
                    section_end = len(lines)
                break
        section_lines = lines[section_start+1:section_end] if section_start is not None else []
        for line in section_lines:
            line = line.strip()
            if line == "" or line.startswith("Batch Size"):
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    batch_size = int(float(parts[0]))
                    defect_rate = float(parts[1])
                    batch_sizes.append(batch_size)
                    defect_rates.append(defect_rate)
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading analytics from CSV: {e}")
    return batch_sizes, defect_rates

@app.route('/api/ai-summary', methods=['GET'])
def get_ai_summary():
    """Get AI summary for frontend using Gemini"""
    try:
        date_str = request.args.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
        print(f"🔍 AI Summary requested for date: {date_str}")
        
        batch_sizes, defect_rates = [], []
        print("🔍 Generating plot...")
        fig = controller.plot_defect_rate(date_str)
        print("✅ Plot generated successfully")
        
        # Try to get data from database first
        try:
            connection = mysql.connector.connect(
                host=os.getenv('DB_HOST', ''),
                user=os.getenv('DB_USER', ''),
                password=os.getenv('DB_PASSWORD', ''),
                database=os.getenv('DB_NAME', '')
            )
            cursor = connection.cursor()
            
            # Query to get data for specific date
            sql = """
                SELECT batch_size, defect_rate 
                FROM safety_check_records 
                WHERE DATE(batch_date) = %s 
                ORDER BY timestamp
            """
            cursor.execute(sql, (date_str,))
            results = cursor.fetchall()
            
            for row in results:
                batch_size, defect_rate = row
                batch_sizes.append(int(batch_size))
                defect_rates.append(float(defect_rate))
                
            cursor.close()
            connection.close()
            print(f"✅ Retrieved {len(batch_sizes)} records from database for AI summary")
            
        except Error as e:
            print(f"❌ Database error for AI summary: {e}")
            # Fallback to CSV
            batch_sizes, defect_rates = get_analytics_from_csv(date_str)
        except Exception as e:
            print(f"❌ Error retrieving data for AI summary: {e}")
            # Fallback to CSV
            batch_sizes, defect_rates = get_analytics_from_csv(date_str)
        
        print(f"🔍 Data loaded: {len(batch_sizes)} batches, {len(defect_rates)} defect rates")
        
        # Check if Gemini API is available
        if not controller.gemini_model:
            print("⚠️ Gemini API not available, generating mock analysis")
            # Generate mock AI analysis
            avg_defect_rate = sum(defect_rates) / len(defect_rates) if defect_rates else 2.5
            max_defect_rate = max(defect_rates) if defect_rates else 3.0
            min_defect_rate = min(defect_rates) if defect_rates else 2.0
            total_batches = len(batch_sizes) if batch_sizes else 10
            
            mock_summary = {
                "executiveSummary": f"Production analysis for {date_str} shows {'excellent' if avg_defect_rate < 2.0 else 'good' if avg_defect_rate < 3.0 else 'acceptable' if avg_defect_rate < 5.0 else 'concerning'} quality performance with an average defect rate of {avg_defect_rate:.2f}%. The manufacturing process demonstrates {'consistent' if max_defect_rate - min_defect_rate < 1.0 else 'variable'} quality control measures across {total_batches} batches.",
                "trendAnalysis": [
                    f"Defect rates show {'stable' if max_defect_rate - min_defect_rate < 0.5 else 'fluctuating'} patterns across production batches",
                    f"Quality performance is {'within acceptable limits' if avg_defect_rate < 3.0 else 'approaching threshold levels' if avg_defect_rate < 5.0 else 'below target standards'}",
                    f"Production consistency is {'excellent' if max_defect_rate - min_defect_rate < 0.5 else 'good' if max_defect_rate - min_defect_rate < 1.0 else 'needs improvement'}"
                ],
                "qualityAssessment": {
                    "score": max(0, 100 - avg_defect_rate * 10),
                    "trend": "stable" if max_defect_rate - min_defect_rate < 0.5 else "variable",
                    "riskLevel": "low" if avg_defect_rate < 2.0 else "medium" if avg_defect_rate < 4.0 else "high"
                },
                "recommendations": [
                    "Continue monitoring quality metrics closely",
                    "Maintain current quality control procedures",
                    "Review production parameters if defect rates exceed 3%"
                ],
                "alerts": [
                    {
                        "type": "info",
                        "message": f"Quality analysis completed for {date_str}"
                    }
                ]
            }
            print("✅ Mock analysis generated successfully")
            return jsonify(mock_summary)
        
        # Set a timeout for the Gemini call (cross-platform)
        ai_summary_text = None
        ai_error = None
        def ai_task():
            nonlocal ai_summary_text, ai_error
            try:
                ai_summary_text = controller.analyze_graph_with_gemini(fig, date_str, batch_sizes, defect_rates)
            except Exception as e:
                ai_error = str(e)
        thread = threading.Thread(target=ai_task)
        thread.start()
        thread.join(timeout=45)
        if thread.is_alive() or ai_error:
            print("⚠️ Gemini API timed out or errored, falling back to mock analysis")
            # Fall back to mock analysis
            avg_defect_rate = sum(defect_rates) / len(defect_rates) if defect_rates else 2.5
            max_defect_rate = max(defect_rates) if defect_rates else 3.0
            min_defect_rate = min(defect_rates) if defect_rates else 2.0
            total_batches = len(batch_sizes) if batch_sizes else 10
            mock_summary = {
                "executiveSummary": f"Production analysis for {date_str} shows {'excellent' if avg_defect_rate < 2.0 else 'good' if avg_defect_rate < 3.0 else 'acceptable' if avg_defect_rate < 5.0 else 'concerning'} quality performance with an average defect rate of {avg_defect_rate:.2f}%. The manufacturing process demonstrates {'consistent' if max_defect_rate - min_defect_rate < 1.0 else 'variable'} quality control measures across {total_batches} batches.",
                "trendAnalysis": [
                    f"Defect rates show {'stable' if max_defect_rate - min_defect_rate < 0.5 else 'fluctuating'} patterns across production batches",
                    f"Quality performance is {'within acceptable limits' if avg_defect_rate < 3.0 else 'approaching threshold levels' if avg_defect_rate < 5.0 else 'below target standards'}",
                    f"Production consistency is {'excellent' if max_defect_rate - min_defect_rate < 0.5 else 'good' if max_defect_rate - min_defect_rate < 1.0 else 'needs improvement'}"
                ],
                "qualityAssessment": {
                    "score": max(0, 100 - avg_defect_rate * 10),
                    "trend": "stable" if max_defect_rate - min_defect_rate < 0.5 else "variable",
                    "riskLevel": "low" if avg_defect_rate < 2.0 else "medium" if avg_defect_rate < 4.0 else "high"
                },
                "recommendations": [
                    "Continue monitoring quality metrics closely",
                    "Maintain current quality control procedures",
                    "Review production parameters if defect rates exceed 3%"
                ],
                "alerts": [
                    {
                        "type": "warning",
                        "message": f"AI analysis timed out, using fallback analysis for {date_str}"
                    }
                ]
            }
            return jsonify(mock_summary)
        else:
            print(f"🔍 Gemini response length: {len(ai_summary_text) if ai_summary_text else 0}")
            # Parse the Gemini response into a structured object
            print("🔍 Parsing Gemini response...")
            from smart_factory_control import parse_gemini_report
            parsed_summary = parse_gemini_report(ai_summary_text)
            print(f"🔍 Parsed summary keys: {list(parsed_summary.keys())}")
            print(f"🔍 Executive summary length: {len(parsed_summary.get('executiveSummary', ''))}")
            print(f"🔍 Trend analysis items: {len(parsed_summary.get('trendAnalysis', []))}")
            print("✅ AI analysis completed successfully")
            return jsonify(parsed_summary)
            
    except Exception as e:
        print(f"❌ Error in AI summary endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts_endpoint():
    """Get system alerts for frontend"""
    return jsonify(get_alerts())

# Production control endpoints
@app.route('/api/production/start', methods=['POST'])
def start_production():
    controller.production_mode = True
    controller.machine_status = "RUNNING"
    return jsonify({'status': 'success', 'message': 'Production started', 'metrics': get_system_metrics()})

@app.route('/api/production/pause', methods=['POST'])
def pause_production():
    controller.production_mode = False
    controller.machine_status = "STANDBY"
    return jsonify({'status': 'success', 'message': 'Production paused', 'metrics': get_system_metrics()})

@app.route('/api/production/emergency_stop', methods=['POST'])
def emergency_stop():
    controller.production_mode = False
    controller.machine_status = "EMERGENCY"
    controller.emergency_mode = True
    # Log production details
    batch_size = getattr(controller, 'batch_size', 0)
    defect_count = getattr(controller, 'defect_count', 0)
    production_count = getattr(controller, 'production_count', 1)
    defect_rate = (defect_count / max(production_count, 1)) * 100
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    controller.log_safety_check(batch_size, defect_rate, timestamp)
    controller.reset_production()  # Reset all metrics after logging
    return jsonify({'status': 'success', 'message': 'Emergency stop triggered', 'metrics': get_system_metrics()})

@app.route('/api/production/quality_check', methods=['POST'])
def quality_check():
    controller.machine_status = "QUALITY_CHECK"
    return jsonify({'status': 'success', 'message': 'Quality check performed', 'metrics': get_system_metrics()})

@app.route('/api/production/image', methods=['GET'])
def get_production_image():
    """Return the current production image as a JPEG for the frontend display."""
    frame = controller.get_current_production_image()
    if frame is None:
        return {'error': 'No production image available'}, 404
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')

@app.route('/api/production/defect', methods=['GET'])
def get_production_defect():
    """Return the defect detection result for the current production image."""
    frame = controller.get_current_production_image()
    if frame is None:
        return {'error': 'No production image available'}, 404
    # Use the existing perform_production_quality_check logic
    result = controller.perform_production_quality_check(frame)
    # The function may return None or a result dict; adapt as needed
    if result is None:
        # Try to get defect info from controller state if available
        is_defective = getattr(controller, 'last_defect_result', False)
        defect_score = getattr(controller, 'last_defect_score', 0)
    else:
        is_defective = result.get('is_defective', False)
        defect_score = result.get('defect_score', 0)
    return {'is_defective': is_defective, 'defect_score': defect_score}

# Safety records endpoint (legacy)
@app.route('/api/safety/records', methods=['GET'])
def get_safety_records_legacy():
    date_filter = request.args.get('date')
    records = []
    try:
        with open(controller.safety_check_records, 'r') as f:
            lines = f.readlines()
        current_date = None
        for line in lines:
            line = line.strip()
            if line.startswith('----') and line.endswith('----'):
                current_date = line.replace('----', '').strip()
            elif line and not line.startswith('Batch Size') and ',' in line and current_date:
                parts = line.split(',')
                if len(parts) >= 3:
                    record = {
                        'date': current_date,
                        'batch_size': int(float(parts[0])),
                        'defect_rate': float(parts[1]),
                        'timestamp': parts[2]
                    }
                    if not date_filter or date_filter == current_date:
                        records.append(record)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'success', 'data': records})

@app.route('/api/safety/log', methods=['POST'])
def log_safety_check():
    data = request.get_json()
    batch_size = data.get('batch_size')
    defect_rate = data.get('defect_rate')
    timestamp = data.get('timestamp') or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if batch_size is None or defect_rate is None:
        return jsonify({'status': 'error', 'message': 'batch_size and defect_rate are required'}), 400
    try:
        controller.log_safety_check(batch_size, defect_rate, timestamp)
        return jsonify({'status': 'success', 'message': 'Safety check logged'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Test mode endpoints
@app.route('/api/test/start', methods=['POST'])
def start_test_mode():
    try:
        controller.start_test_mode()
        return jsonify({'status': 'success', 'message': 'Test mode started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/test/stop', methods=['POST'])
def stop_test_mode():
    try:
        controller.stop_test_mode()
        return jsonify({'status': 'success', 'message': 'Test mode stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/test/results', methods=['GET'])
def get_test_results():
    try:
        results = controller.test_results if hasattr(controller, 'test_results') else []
        return jsonify({'status': 'success', 'data': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ai-summary-pdf', methods=['POST'])
def export_ai_summary_pdf():
    try:
        print("🔍 PDF export endpoint called")
        data = request.get_json()
        print(f"🔍 Received data: {data}")
        
        date_str = data.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
        batch_sizes = data.get('batch_sizes', [])
        defect_rates = data.get('defect_rates', [])
        
        print(f"🔍 Date: {date_str}")
        print(f"🔍 Batch sizes: {len(batch_sizes)}")
        print(f"🔍 Defect rates: {len(defect_rates)}")
        
        print("🔍 Generating plot...")
        fig = controller.plot_defect_rate(date_str)
        if fig is None:
            print("🔍 No data for plot, creating placeholder...")
            fig = plt.figure()
            plt.text(0.5, 0.5, 'No data available for this date', ha='center', va='center', fontsize=16)
            plt.axis('off')
        
        # Use Gemini for AI summary
        print("🔍 Generating AI summary with Gemini...")
        ai_analysis = controller.analyze_graph_with_gemini(fig, date_str, batch_sizes, defect_rates)
        
        print("🔍 Generating PDF report...")
        pdf_path = controller.generate_pdf_report(ai_analysis, date_str, batch_sizes, defect_rates, fig)
        
        print(f"🔍 PDF path returned: {pdf_path}")
        
        if pdf_path and os.path.exists(pdf_path):
            print(f"✅ PDF file exists, sending file: {pdf_path}")
            print(f"✅ File size: {os.path.getsize(pdf_path)} bytes")
            return send_file(pdf_path, as_attachment=True)
        else:
            print(f"❌ PDF file not found or not created: {pdf_path}")
            return jsonify({'error': 'Failed to generate PDF'}), 500
            
    except Exception as e:
        print(f"❌ Error in PDF export endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

@app.route('/api/export/graph/<date_str>', methods=['GET'])
def export_graph_image(date_str):
    """Export graph image for a specific date"""
    try:
        print(f"🔍 Graph export endpoint called for date: {date_str}")
        format_type = request.args.get('format', 'png').lower()
        
        if format_type not in ['png', 'jpg', 'jpeg']:
            format_type = 'png'
        
        # Generate the plot
        fig = controller.plot_defect_rate(date_str)
        if fig is None:
            print("🔍 No data for plot, creating placeholder...")
            fig = plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, f'No data available for {date_str}', ha='center', va='center', fontsize=16)
            plt.axis('off')
        
        # Save the figure to a temporary file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"defect_analysis_graph_{date_str}_{timestamp}.{format_type}"
        
        fig.savefig(filename, format=format_type, bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory
        
        # Add to cleanup list
        temp_files.append(filename)
        
        print(f"✅ Graph saved: {filename}")
        
        if os.path.exists(filename):
            # Schedule cleanup after 5 minutes
            def delayed_cleanup():
                time.sleep(300)  # 5 minutes
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                        temp_files.remove(filename)
                        print(f"🧹 Cleaned up temporary graph file: {filename}")
                except Exception as e:
                    print(f"⚠️ Could not remove temporary file {filename}: {e}")
            
            cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
            cleanup_thread.start()
            
            return send_file(filename, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'Failed to generate graph'}), 500
            
    except Exception as e:
        print(f"❌ Error in graph export endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate graph: {str(e)}'}), 500

@app.route('/api/export/csv/<date_str>', methods=['GET'])
def export_csv_data(date_str):
    """Export CSV data for a specific date"""
    try:
        print(f"🔍 CSV export endpoint called for date: {date_str}")
        
        # Get data for the specific date
        batch_sizes, defect_rates, timestamps = [], [], []
        
        try:
            with open(controller.safety_check_records, 'r') as f:
                lines = f.readlines()
            
            section_start = None
            section_end = None
            for i, line in enumerate(lines):
                if line.strip() == f"---- {date_str} ----":
                    section_start = i
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith("----"):
                            section_end = j
                            break
                    if section_end is None:
                        section_end = len(lines)
                    break
            
            if section_start is not None:
                section_lines = lines[section_start+1:section_end]
                for line in section_lines:
                    line = line.strip()
                    if line == "" or line.startswith("Batch Size"):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            batch_size = int(float(parts[0]))
                            defect_rate = float(parts[1])
                            timestamp = parts[2]
                            batch_sizes.append(batch_size)
                            defect_rates.append(defect_rate)
                            timestamps.append(timestamp)
                        except Exception:
                            continue
        except Exception as e:
            print(f"Warning: Could not read data for {date_str}: {e}")
        
        # If no data found, generate sample data
        if not batch_sizes:
            print("🔍 No data found, generating sample data for CSV")
            batch_sizes = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
            defect_rates = [2.1, 1.8, 2.3, 1.9, 2.0, 1.7, 2.2, 1.6, 1.9, 2.1]
            timestamps = [f"{date_str} 09:00:00", f"{date_str} 10:00:00", f"{date_str} 11:00:00", 
                        f"{date_str} 12:00:00", f"{date_str} 13:00:00", f"{date_str} 14:00:00",
                        f"{date_str} 15:00:00", f"{date_str} 16:00:00", f"{date_str} 17:00:00", f"{date_str} 18:00:00"]
        
        # Create CSV content
        csv_buffer = io.StringIO()
        csv_buffer.write("Date,Batch_Number,Batch_Size,Defect_Rate,Quality_Score,Status,Timestamp\n")
        
        for i, (batch_size, defect_rate, timestamp) in enumerate(zip(batch_sizes, defect_rates, timestamps)):
            quality_score = 100 - defect_rate
            status = 'Excellent' if defect_rate < 2.0 else 'Good' if defect_rate < 3.0 else 'Acceptable' if defect_rate < 5.0 else 'Needs Attention'
            csv_buffer.write(f"{date_str},{i+1},{batch_size},{defect_rate:.2f},{quality_score:.1f},{status},{timestamp}\n")
        
        # Add summary statistics
        avg_defect_rate = sum(defect_rates) / len(defect_rates) if defect_rates else 0
        csv_buffer.write(f"\nSummary Statistics\n")
        csv_buffer.write(f"Date,{date_str}\n")
        csv_buffer.write(f"Total_Batches,{len(batch_sizes)}\n")
        csv_buffer.write(f"Average_Defect_Rate,{avg_defect_rate:.2f}\n")
        csv_buffer.write(f"Max_Defect_Rate,{max(defect_rates):.2f}\n")
        csv_buffer.write(f"Min_Defect_Rate,{min(defect_rates):.2f}\n")
        csv_buffer.write(f"Quality_Score,{100 - avg_defect_rate:.1f}\n")
        
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        
        # Create file response
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"defect_analysis_data_{date_str}_{timestamp}.csv"
        
        # Convert string to bytes for file download
        csv_bytes = csv_content.encode('utf-8')
        
        from flask import Response
        response = Response(csv_bytes, mimetype='text/csv')
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        
        print(f"✅ CSV generated: {filename}")
        return response
        
    except Exception as e:
        print(f"❌ Error in CSV export endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate CSV: {str(e)}'}), 500

@app.route('/api/gesture/recognize', methods=['POST'])
def recognize_gesture():
    """Recognize hand gesture from an uploaded image and return the gesture and action."""
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return {'error': 'Invalid image file'}, 400
    try:
        # Call the main process_frame logic to handle production and gesture state
        controller.process_frame(image)
        # After processing, get the current gesture and machine status
        gesture = getattr(controller, 'current_gesture', None)
        action = None
        if gesture == 'start_production':
            action = 'start_production'
        elif gesture == 'emergency_stop':
            action = 'emergency_stop'
        elif gesture == 'quality_check':
            action = 'quality_check'
        elif gesture == 'plot_defect_rate':
            action = 'plot_defect_rate'
        return {'gesture': gesture, 'action': action, 'machine_status': controller.machine_status}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/detect-image', methods=['POST'])
def detect_image():
    """Process an image for ensemble detection using multiple YOLO models."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    settings_str = request.form.get('settings', '{}')
    enabled_models_str = request.form.get('enabled_models', '{}')
    
    try:
        # Parse settings and enabled models
        settings = json.loads(settings_str)
        enabled_models = json.loads(enabled_models_str)
        
        # Read the image file
        image_bytes = file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Get the ensemble detector
        detector = get_ensemble_detector()
        if detector is None:
            return jsonify({'error': 'Ensemble detector not initialized'}), 500
        
        # Update enabled models in detector
        detector.enabled_models = enabled_models
        
        # Get detection parameters
        conf_threshold = settings.get('confidence', 0.25)
        iou_threshold = settings.get('iou', 0.45)
        
        # Start timing
        start_time = time.time()
        
        # Run ensemble detection
        detections = detector.run_ensemble_detection(image, conf_threshold, iou_threshold)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate model statistics
        model_stats = {}
        for model_name in detector.models.keys():
            model_detections = [d for d in detections if d['model'] == model_name]
            if model_detections:
                confidences = [d['confidence'] for d in model_detections]
                model_stats[model_name] = {
                    'count': len(model_detections),
                    'avg_confidence': sum(confidences) / len(confidences),
                    'min_confidence': min(confidences),
                    'max_confidence': max(confidences)
                }
            else:
                model_stats[model_name] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'min_confidence': 0,
                    'max_confidence': 0
                }
        
        # Count enabled models
        enabled_count = sum(1 for enabled in enabled_models.values() if enabled)
        
        # Log detection metrics to database
        log_detection_metrics_to_db(len(detections), enabled_count, processing_time, model_stats)
        
        # Format the result for the frontend
        result = {
            'detections': detections,
            'metrics': {
                'total_detections': len(detections),
                'enabled_models': enabled_count,
                'total_models': len(detector.models),
                'model_stats': model_stats,
                'processing_time': processing_time,
                'image_size': {
                    'width': width,
                    'height': height
                }
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in detect-image endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection-history', methods=['GET'])
def get_detection_history():
    """Get detection metrics history for frontend"""
    return jsonify(get_detection_metrics_history())

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)

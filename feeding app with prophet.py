import pickle
import sys
import os
import time
import csv
import datetime
import numpy as np
import cv2
from prophet_feeding_model import ProphetFeedingModel
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QWidget, QLabel, QGridLayout, QGroupBox,
                             QMessageBox, QFileDialog, QHeaderView, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QDateTime
import matplotlib
import pyqtgraph as pg

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

plt.ioff()  # Turn off interactive mode
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import traceback  # Add explicit import for traceback module

# Constants
MAX_FEEDING_mode_DURATION = 600  # 10 minutes maximum for any feeding mode
MAX_DURING_mode_NO_DOSAGE = 300  # 5 minutes maximum in "feeding" mode with no dosage
MAX_FEEDING_DURATION = 600  # 10 minutes
MAX_TRAJECTORY_LEN = 30  # Number of frames to keep for trajectory lines
SPEED_HISTORY_LEN = 600  # Number of readings to keep for graphs
MONITORING_WINDOW = 300  # 5 minutes (300 seconds) monitoring window
FEED_DECISION_INTERVAL = 300  # Check for feeding every 60s
MIN_FEED_INTERVAL = 7200  # Minimum seconds between feedings (2 hours)
INITIAL_FEED_DELAY = 300  # Seconds before first feeding analysis (15 minutes)
PRE_FEEDING_DURATION = 300  # 5 minutes pre-feeding analysis
POST_FEEDING_DURATION = 300  # 5 minutes post-feeding analysis
DOSAGE_ASSESSMENT_PERIOD = 30  # 30 seconds to assess each dosage effect
DAILY_OPERATION_START = "07:30"  # Daily start time
DAILY_OPERATION_END = "23:59"  # Daily end time
DATA_FOLDER = "fish_data"  # Folder to store CSV and model data
# YOLOv8 model path with fallback checks
MODEL_PATH = "runs/detect/train8/weights/best.pt"
DEBUG_MODE = False  # Set to False to disable debug output

# Create data folder if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)

# Check for CUDA availability
CUDA_AVAILABLE = False
try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_count > 0:
        CUDA_AVAILABLE = True
        print(f"CUDA is available for OpenCV with {cuda_count} device(s)!")
    else:
        print("CUDA is available in OpenCV API but no CUDA devices found.")
except Exception as e:
    print(f"OpenCV CUDA support not available: {e}")

# No need to try exporting to TensorRT if CUDA isn't available for PyTorch
if os.path.exists(MODEL_PATH) and CUDA_AVAILABLE:
    try:
        # Don't rely on PyTorch CUDA for Jetson - use TensorRT directly
        # The YOLO model will use TensorRT optimizations automatically
        print("Model found. Will use for inference without TensorRT export.")
    except Exception as e:
        print(f"Error with model: {e}")
        print("Will use standard model instead.")

def datetime_to_epoch(dt_obj):
    """Convert datetime object to epoch seconds reliably for PyQtGraph"""
    if isinstance(dt_obj, datetime.datetime):
        return (dt_obj - datetime.datetime(1970, 1, 1)).total_seconds()
    elif isinstance(dt_obj, (int, float)) and dt_obj > 1e9:  # Already looks like an epoch timestamp
        return dt_obj
    else:
        # Return current time as fallback
        print(f"WARNING: Could not convert {dt_obj} to epoch time, using current time")
        return time.time()

def ensure_datetime(timestamp):
    """Ensure a timestamp is converted to a datetime object"""
    if isinstance(timestamp, datetime.datetime):
        return timestamp
    elif isinstance(timestamp, (int, float)) and timestamp > 1e9:  # Epoch timestamp
        return datetime.datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        # Try to parse string to datetime
        try:
            return datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                return datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                # Last resort, return current time
                print(f"WARNING: Could not parse timestamp string: {timestamp}")
                return datetime.datetime.now()
    else:
        # Default fallback
        print(f"WARNING: Unknown timestamp format: {timestamp}")
        return datetime.datetime.now()

class KalmanTracker:
    """Tracker for fish speed estimation with speed Kalman filter only"""

    def __init__(self, box_wh):
        # Fish size - just use width
        self.fish_size = box_wh[0]  # Width of bounding box

        # Speed-specific Kalman filter (1D)
        self.speed_kf = KalmanFilter(dim_x=2, dim_z=1)

        dt = 1.0 / 30.0  # Assume 30fps by default

        # State transition matrix for speed [speed, accel]
        self.speed_kf.F = np.array([
            [1, dt],
            [0, 1]
        ])

        # Measurement function (only measure speed)
        self.speed_kf.H = np.array([[1, 0]])

        # Measurement noise (lower for smoother output)
        self.speed_kf.R = np.array([[0.5]])

        # Process noise
        self.speed_kf.Q = np.array([
            [0.05, 0.02],
            [0.02, 0.1]
        ])

        # Initial state uncertainty
        self.speed_kf.P = np.array([
            [1, 0],
            [0, 1]
        ])

        # Initial state [speed, accel]
        self.speed_kf.x = np.array([0, 0]).reshape(2, 1)

        # Storage for positions and speeds
        self.positions = deque(maxlen=MAX_TRAJECTORY_LEN)
        self.raw_speeds = deque(maxlen=10)
        self.raw_speed = 0
        self.filtered_speed = 0
        self.active = True
        self.last_update = time.time()
        self.color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )

    def update(self, bbox):
        # Extract center point from bbox (x, y, w, h)
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2

        # Update fish size (just the width)
        self.fish_size = bbox[2]

        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time

        # Update speed KF transition matrix with actual dt
        if dt > 0:
            self.speed_kf.F[0, 1] = dt

        # Store position (directly from detection, no position KF)
        current_position = (int(cx), int(cy))
        self.positions.append(current_position)

        # Calculate raw speed (body lengths per second)
        if len(self.positions) >= 2:
            p1 = np.array(self.positions[-2])
            p2 = np.array(self.positions[-1])
            distance = np.linalg.norm(p2 - p1)  # Euclidean distance

            # Convert to body lengths per second
            if self.fish_size > 0 and dt > 0:
                # Calculate raw speed
                self.raw_speed = distance / self.fish_size / dt
                self.raw_speeds.append(self.raw_speed)

                # Predict and update speed KF
                self.speed_kf.predict()
                self.speed_kf.update(np.array([self.raw_speed]))

                # Get filtered speed
                self.filtered_speed = float(self.speed_kf.x[0])

                if DEBUG_MODE:  # Debug output to compare raw vs filtered
                    print(f"Fish speed - Raw: {self.raw_speed:.2f}, Filtered: {self.filtered_speed:.2f} BL/s")

        return np.array([cx, cy])

    def get_speed(self):
        """Return the Kalman-filtered speed value"""
        return self.filtered_speed

    def get_trajectory(self):
        return list(self.positions)


class VideoThread(QThread):
    """Thread for processing video frames with YOLO detection - optimized version"""
    frame_ready = pyqtSignal(np.ndarray, list, float, float)

    def __init__(self, camera_source=0):
        super().__init__()
        self.camera_source = camera_source
        self.running = False
        self.trackers = {}
        self.next_id = 0
        self.model = None

        self.speed_history = deque(maxlen=600)
        self.variance_history = deque(maxlen=600)
        self.timestamps = deque(maxlen=600)

        self.current_window_speeds = deque(maxlen=300)
        self.current_window_variances = deque(maxlen=300)
        self.current_window_timestamps = deque(maxlen=300)

        # Initialize with some data for graphs
        self.ensure_data_collection()

    # When adding data from your video thread
    def add_data_point(self, timestamp, speed, variance, event_type=None):
        """Add a data point with verified timestamp format"""
        # Convert timestamp to datetime if it's not already
        if not isinstance(timestamp, datetime.datetime):
            try:
                if isinstance(timestamp, (int, float)) and timestamp > 1e9:
                    timestamp = datetime.datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                print(f"Error converting timestamp: {e}")
                timestamp = datetime.datetime.now()  # Use current time as fallback

        # Add to rolling window with verified datetime object
        self.current_window_speeds.append(speed)
        self.current_window_variances.append(variance)
        self.current_window_timestamps.append(timestamp)

    def run(self):
        """Thread for processing video frames with YOLO detection - optimized version"""
        print("VideoThread started. Initialising data collection...")
        try:
            # Target resolution - force 640 x 640
            target_width, target_height = 640, 480
            # Check if model path exists
            if not os.path.exists(MODEL_PATH):
                print(f"ERROR: Model file not found at {MODEL_PATH}")
                print("Using YOLO default model for testing purposes.")
                # Use a default YOLO model for testing
                self.model = YOLO("yolov8n.pt")  # Use smaller model for better performance
            else:
                # Try to load engine file first
                engine_path = os.path.splitext(MODEL_PATH)[0] + '.engine'
                if os.path.exists(engine_path):
                    try:
                        print(f"Loading TensorRT engine from {engine_path}")
                        self.model = YOLO(engine_path)
                        print("TensorRT engine loaded successfully")
                    except Exception as e:
                        print(f"Error loading engine: {e}")
                        print("Falling back to original model")
                        self.model = YOLO(MODEL_PATH)
                else:
                    # Load regular model
                    print(f"Loading standard model from {MODEL_PATH}")
                    self.model = YOLO(MODEL_PATH)

            # Check for CUDA - DO NOT force CPU mode
            import torch
            if torch.cuda.is_available():
                try:
                    print("CUDA is available, using for inference")
                    # Let the model use CUDA naturally
                except Exception as e:
                    print(f"Error using CUDA: {e}")
            else:
                print("CUDA not available, using CPU")


            try:
                # Initialize video capture with multiple fallback options
                cap = None

                # Try method 1: V4L2 with MJPG format
                try:
                    print("Trying V4L2 with MJPG format...")
                    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    cap.set(cv2.CAP_PROP_FPS, 60)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

                    if not cap.isOpened():
                        print("Failed to open with V4L2+MJPG")
                        cap = None
                    else:
                        print("Successfully opened camera with V4L2+MJPG")
                except Exception as e:
                    print(f"Error with V4L2+MJPG: {e}")
                    cap = None

                # Try method 2: Standard capture
                if cap is None or not cap.isOpened():
                    try:
                        print("Trying standard capture...")
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

                        if not cap.isOpened():
                            print("Failed to open with standard capture")
                            cap = None
                        else:
                            print("Successfully opened camera with standard capture")
                    except Exception as e:
                        print(f"Error with standard capture: {e}")
                        cap = None

                # Final fallback
                if cap is None or not cap.isOpened():
                    print("All camera methods failed. Using fallback.")
                    self.handle_camera_failure()
                    return

                # Get actual camera properties
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Camera reports resolution: {actual_width}x{actual_height}")

            except Exception as e:
                print(f"Camera initialisation error: {e}")
                self.handle_camera_failure()
                return

            self.running = True
            last_data_time = time.time()  # Track last time we added to data collections
            frame_count = 0
            last_fps_check = time.time()

            # Initialize buffer for moving average
            speed_buffer = []
            variance_buffer = []
            buffer_size = 5  # Average over 5 frames for smoother readings

            while self.running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from camera. Retrying...")
                        # Wait and retry a few times before giving up
                        time.sleep(0.5)
                        continue


                    if frame.shape[1] != target_width or frame.shape[0] != target_height:
                        frame = cv2.resize(frame, (target_width, target_height))

                    # Skip processing if graph update is in progress
                    graph_update_in_progress = False
                    if hasattr(self, 'parent'):
                        graph_update_in_progress = getattr(self.parent, 'updating_graphs', False)

                    if graph_update_in_progress:
                        # Skip this frame
                        continue

                    # Run YOLOv8 detection and tracking
                    try:
                        results = self.model.track(
                            frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=0.7,  # Lower confidence threshold
                            iou=0.5,
                            agnostic_nms=True,
                            verbose=True
                        )
                    except Exception as e:
                        print(f"YOLO tracking error: {e}")
                        # Continue with empty results
                        tracks = []
                        avg_speed = 0.0 if not self.speed_history else self.speed_history[-1]
                        speed_variance = 0.0 if not self.variance_history else self.variance_history[-1]
                        self.frame_ready.emit(frame, tracks, avg_speed, speed_variance)
                        continue
                        # Use empty results
                        tracks = []
                        avg_speed = 0.0 if not self.speed_history else self.speed_history[-1]
                        speed_variance = 0.0 if not self.variance_history else self.variance_history[-1]

                        # Only record data points at 1-second intervals to reduce data volume
                        if time.time() - last_data_time >= 1.0:
                            current_time = datetime.datetime.now()
                            midnight = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                            seconds_since_midnight = (current_time - midnight).total_seconds()
                            self.speed_history.append(avg_speed)
                            self.variance_history.append(speed_variance)
                            self.timestamps.append(seconds_since_midnight)
                            last_data_time = time.time()

                        # Emit frame
                        self.frame_ready.emit(frame, tracks, avg_speed, speed_variance)
                        continue

                    # Process detection results
                    if results and len(results) > 0:
                        tracks = []
                        speeds = []

                        # Get the first result (only one frame)
                        result = results[0]

                        if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                            try:
                                boxes = result.boxes.xyxy.cpu().numpy()
                                track_ids = result.boxes.id.int().cpu().numpy()

                                # Update trackers
                                current_ids = set()
                                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                                    x1, y1, x2, y2 = box
                                    w, h = x2 - x1, y2 - y1
                                    box_center = [x1, y1, w, h]

                                    # In the run method of VideoThread, when creating a new tracker:
                                    if track_id not in self.trackers:
                                        self.trackers[track_id] = KalmanTracker((w, h))

                                    # Update tracker
                                    self.trackers[track_id].update(box_center)
                                    tracks.append((track_id, box, self.trackers[track_id].get_trajectory(),
                                                   self.trackers[track_id].color))
                                    speeds.append(self.trackers[track_id].get_speed())
                                    current_ids.add(track_id)

                                # Remove inactive trackers
                                for track_id in list(self.trackers.keys()):
                                    if track_id not in current_ids:
                                        del self.trackers[track_id]
                            except Exception as e:
                                print(f"Tracker update error: {e}")
                                tracks = []
                                speeds = []

                        # Calculate average speed and variance for this frame
                        current_time = datetime.datetime.now()
                        if speeds:
                            avg_speed = np.mean(speeds)
                            speed_variance = np.var(speeds) if len(speeds) > 1 else 0
                        else:
                            # Use previous values or defaults
                            avg_speed = 0.0 if not self.speed_history else self.speed_history[-1]
                            speed_variance = 0.0 if not self.variance_history else self.variance_history[-1]

                        # Add to moving average buffer
                        speed_buffer.append(avg_speed)
                        variance_buffer.append(speed_variance)
                        if len(speed_buffer) > buffer_size:
                            speed_buffer.pop(0)
                            variance_buffer.pop(0)

                        # Use moving average for smoother readings
                        smoothed_speed = np.mean(speed_buffer)
                        smoothed_variance = np.mean(variance_buffer)

                        # Only record data points at 1-second intervals to reduce data volume
                        if time.time() - last_data_time >= 1.0:
                            self.speed_history.append(smoothed_speed)
                            self.variance_history.append(smoothed_variance)
                            self.timestamps.append(current_time)
                            last_data_time = time.time()

                            if DEBUG_MODE:
                                print(
                                    f"Added data point: Speed={smoothed_speed:.2f}, Variance={smoothed_variance:.2f}, Total points={len(self.speed_history)}")

                        # Emit frame with tracking data and metrics - using smoothed values
                        self.frame_ready.emit(
                            frame,
                            tracks,
                            smoothed_speed,
                            smoothed_variance
                        )
                    else:
                        # No results, emit frame with empty tracking
                        current_time = datetime.datetime.now()
                        avg_speed = 0.0 if not self.speed_history else self.speed_history[-1]
                        speed_variance = 0.0 if not self.variance_history else self.variance_history[-1]

                        # Only record data points at 1-second intervals
                        if time.time() - last_data_time >= 1.0:
                            self.speed_history.append(avg_speed)
                            self.variance_history.append(speed_variance)
                            self.timestamps.append(current_time)
                            last_data_time = time.time()

                        self.frame_ready.emit(frame, [], avg_speed, speed_variance)

                    # Small delay to reduce CPU usage - we don't need to process every frame
                    # This helps stabilize performance
                    time.sleep(0.01)

                except Exception as e:
                    print(f"Frame processing error: {e}")
                    traceback.print_exc()
                    time.sleep(0.5)  # Delay before trying again

            cap.release()

        except Exception as e:
            print(f"Video thread critical error: {e}")
            traceback.print_exc()

    def handle_camera_failure(self):
        """Handle failure to open camera by creating dummy frames"""
        while self.running:
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Camera Unavailable", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Use dummy data
            tracks = []
            avg_speed = 0.5 + 0.2 * np.sin(time.time())
            speed_variance = 0.1 + 0.05 * np.cos(time.time())

            # Record dummy data at 1 second intervals
            current_time = datetime.datetime.now()
            if not hasattr(self, 'last_data_timestamp') or (
                    current_time - self.last_data_timestamp).total_seconds() >= 1.0:
                self.speed_history.append(avg_speed)
                self.variance_history.append(speed_variance)
                self.timestamps.append(current_time)
                self.last_data_timestamp = current_time

            # Emit frame
            self.frame_ready.emit(dummy_frame, tracks, avg_speed, speed_variance)
            time.sleep(0.1)

    def stop(self):
        self.running = False
        self.wait()

    def get_speed_data(self, window_seconds=None):
        """Get speed data, optionally limited to the recent time window

        Args:
            window_seconds: If provided, only return data from the last window_seconds.
                        If None, return all data.
        """
        if DEBUG_MODE:
            print(f"get_speed_data called. Data points: {len(self.timestamps)}")
            if len(self.timestamps) > 0:
                print(f"First timestamp: {self.timestamps[0]}, Last timestamp: {self.timestamps[-1]}")
                print(
                    f"Speed range: {min(self.speed_history) if self.speed_history else 0}-{max(self.speed_history) if self.speed_history else 0}")

        # If no window specified or not enough data, return all data
        if window_seconds is None or len(self.timestamps) < 2:
            return list(self.timestamps), list(self.speed_history), list(self.variance_history)

        # Calculate window based on most recent timestamp
        if self.timestamps:
            current_time = self.timestamps[-1]
            cutoff_time = current_time - datetime.timedelta(seconds=window_seconds)

            # Find index of first timestamp in window
            start_idx = 0
            for i, ts in enumerate(self.timestamps):
                if ts >= cutoff_time:
                    start_idx = i
                    break

            # Return sliced data
            return (list(self.timestamps)[start_idx:],
                    list(self.speed_history)[start_idx:],
                    list(self.variance_history)[start_idx:])

        # Fallback if no timestamps
        return [], [], []

    def get_current_window_data(self):
        """Return the current 5-minute window for analysis"""
        return list(self.current_window_timestamps), list(self.current_window_speeds), list(
            self.current_window_variances)

    def ensure_data_collection(self):
        """Add minimal dummy data if necessary for testing graphs"""
        if len(self.timestamps) < 2:
            if DEBUG_MODE:
                print("Adding minimal dummy data points for graph testing")

            current_time = datetime.datetime.now()
            earlier_time = current_time - datetime.timedelta(seconds=10)

            # Add just two points 10 seconds apart to establish timeline
            self.timestamps.append(earlier_time)
            self.speed_history.append(0.5)
            self.variance_history.append(0.1)

            self.timestamps.append(current_time)
            self.speed_history.append(0.7)
            self.variance_history.append(0.2)

            if DEBUG_MODE:
                print(f"Added minimal dummy points. Now have {len(self.timestamps)} timestamps")


class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        """Convert timestamp values to readable time strings for PyQtGraph axis"""
        try:
            # Format as actual clock time (HH:MM:SS)
            time_strings = []

            for value in values:
                # Skip invalid values
                if not np.isfinite(value):
                    time_strings.append('')
                    continue

                # PyQtGraph works with float values - convert to datetime for formatting
                if value > 1e9:  # Unix timestamps are typically large numbers
                    # Convert epoch seconds to datetime
                    try:
                        # Apply timezone adjustment for display purposes
                        # From UTC time to local time (UTC-8)
                        dt = datetime.datetime.fromtimestamp(value)
                        # If needed, adjust the time to match your local timezone
                        dt = dt - datetime.timedelta(hours=8)  # Uncomment and adjust if necessary
                        time_strings.append(dt.strftime('%H:%M:%S'))
                    except (ValueError, OverflowError):
                        # Fallback if conversion fails
                        time_strings.append(f"{value:.1f}")
                else:
                    # Fallback for non-timestamp values
                    time_strings.append(f"{value:.1f}")

            return time_strings
        except Exception as e:
            print(f"Error formatting time axis: {e}")
            # Fallback to default formatting
            return [str(v) for v in values]

class SpeedGraph(QWidget):
    """PyQtGraph implementation for displaying speed metrics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create time axis for x-axis
        time_axis = TimeAxisItem(orientation='bottom')

        # Create plot widget with custom time axis
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': time_axis})
        layout.addWidget(self.plot_widget)

        # Configure plot
        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.setLabel('left', 'Speed (BL/s)')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 2)  # Initial y range

        self.setMinimumHeight(200)

        # Create plot data item for speed line
        self.speed_line = self.plot_widget.plot(pen=pg.mkPen(color='b', width=2))

        # Placeholders for vertical markers (e.g., missed feedings)
        self.markers = []

        # Store references to data
        self.dates = []
        self.speeds = []

        # For satiated region and tracking if data has been plotted
        self.satiated_region = None
        self.has_data = False

        print("PyQtGraph SpeedGraph initialized")

    def update_plot(self, timestamps, speeds, satiated_range=None):
        """Updated method for updating PyQtGraph plots with proper timestamp handling"""
        # Determine if this is speed or variance graph
        graph_type = 'Speed' if hasattr(self, 'speed_line') else 'Variance'


        if not timestamps or not speeds or len(timestamps) < 2:
            print("Not enough data points, skipping update")
            return

        try:
            # Make sure timestamps and values have the same length
            min_length = min(len(timestamps), len(speeds))
            timestamps = timestamps[-min_length:]
            values = speeds[-min_length:]

            # First ensure we have datetime objects
            dt_timestamps = []
            for ts in timestamps:
                if isinstance(ts, datetime.datetime):
                    dt_timestamps.append(ts)
                else:
                    # Use our helper to convert to datetime
                    dt_timestamps.append(self.parent().ensure_datetime(ts))

            # Find the current time and the cutoff time (5 minutes ago)
            current_time = dt_timestamps[-1]  # Most recent timestamp
            cutoff_time = current_time - datetime.timedelta(seconds=300)  # 5 minutes before latest timestamp

            # Find the index where we should start (first point after cutoff)
            start_idx = 0
            for i, ts in enumerate(dt_timestamps):
                if ts >= cutoff_time:
                    start_idx = i
                    break

            # Create window of data - only the last 5 minutes
            window_timestamps = dt_timestamps[start_idx:]
            window_values = values[start_idx:]

            # Convert datetime timestamps to epoch time for PyQtGraph
            epoch_times = []
            for ts in window_timestamps:
                epoch_times.append(datetime_to_epoch(ts))

            # Update data in plot
            if hasattr(self, 'speed_line'):
                self.speed_line.setData(epoch_times, window_values)
            elif hasattr(self, 'variance_line'):
                self.variance_line.setData(epoch_times, window_values)

            # Update x range to show all data
            if len(epoch_times) >= 2:
                self.plot_widget.setXRange(min(epoch_times), max(epoch_times))

                # Update y range based on data with some padding
                max_value = max(window_values) if window_values else (2 if hasattr(self, 'speed_line') else 1)
                min_y = 0
                max_y = max(2 if hasattr(self, 'speed_line') else 1, max_value * 1.2)
                self.plot_widget.setYRange(min_y, max_y)

            # Handle satiated region
            if satiated_range and satiated_range != getattr(self, 'last_satiated_range', None):
                self.last_satiated_range = satiated_range

                # Remove old region if it exists
                if hasattr(self, 'satiated_region') and self.satiated_region:
                    self.plot_widget.removeItem(self.satiated_region)

                # Create new region
                y_min, y_max = satiated_range
                self.satiated_region = pg.LinearRegionItem(
                    values=[y_min, y_max],
                    orientation=pg.LinearRegionItem.Horizontal,
                    brush=pg.mkBrush(0, 255, 0, 50),  # Semi-transparent green
                    movable=False
                )
                self.plot_widget.addItem(self.satiated_region)

            # Mark that we have data
            self.has_data = True

        except Exception as e:
            print(f"Error updating {graph_type.lower()} graph: {e}")
            import traceback
            traceback.print_exc()

    def add_missed_feeding_marker(self, timestamp):
        """Add a vertical line to indicate a missed feeding event"""
        if isinstance(timestamp, datetime.datetime):
            # Convert to epoch time
            epoch_time = (timestamp - datetime.datetime(1970, 1, 1)).total_seconds()

            # Create vertical line
            line = pg.InfiniteLine(
                pos=epoch_time,
                angle=90,
                pen=pg.mkPen(color='b', width=1, style=Qt.DashLine),
                label=f"Missed {timestamp.strftime('%H:%M')}"
            )

            # Add to plot and store reference
            self.plot_widget.addItem(line)
            self.markers.append(line)
            return line
        return None

    def clear_markers(self):
        """Remove all markers from the plot"""
        for marker in self.markers:
            self.plot_widget.removeItem(marker)
        self.markers = []


class VarianceGraph(QWidget):
    """PyQtGraph implementation for displaying variance metrics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create time axis for x-axis
        time_axis = TimeAxisItem(orientation='bottom')

        # Create plot widget with custom time axis
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': time_axis})
        layout.addWidget(self.plot_widget)

        # Configure plot
        self.plot_widget.setBackground('w')  # White background
        self.plot_widget.setLabel('left', 'Variance')
        self.plot_widget.setLabel('bottom', 'Time')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 1)  # Initial y range

        self.setMinimumHeight(200)

        # Create plot data item for variance line
        self.variance_line = self.plot_widget.plot(pen=pg.mkPen(color='r', width=2))

        # Placeholders for vertical markers (e.g., missed feedings)
        self.markers = []

        # Store references to data
        self.dates = []
        self.variances = []

        # For satiated region and tracking if data has been plotted
        self.satiated_region = None
        self.has_data = False

        print("PyQtGraph VarianceGraph initialized")

    def update_plot(self, timestamps, variances, satiated_range=None):
        """Updated method for updating PyQtGraph plots with proper timestamp handling"""
        # Determine if this is speed or variance graph
        graph_type = 'Speed' if hasattr(self, 'speed_line') else 'Variance'

        if not timestamps or not variances or len(timestamps) < 2:
            print("Not enough data points, skipping update")
            return

        try:
            # Make sure timestamps and values have the same length
            min_length = min(len(timestamps), len(variances))
            timestamps = timestamps[-min_length:]
            values = variances[-min_length:]

            # First ensure we have datetime objects
            dt_timestamps = []
            for ts in timestamps:
                if isinstance(ts, datetime.datetime):
                    dt_timestamps.append(ts)
                else:
                    # Use our helper to convert to datetime
                    dt_timestamps.append(self.parent().ensure_datetime(ts))

            # Find the current time and the cutoff time (5 minutes ago)
            current_time = dt_timestamps[-1]  # Most recent timestamp
            cutoff_time = current_time - datetime.timedelta(seconds=300)  # 5 minutes before latest timestamp

            # Find the index where we should start (first point after cutoff)
            start_idx = 0
            for i, ts in enumerate(dt_timestamps):
                if ts >= cutoff_time:
                    start_idx = i
                    break

            # Create window of data - only the last 5 minutes
            window_timestamps = dt_timestamps[start_idx:]
            window_values = values[start_idx:]

            # Convert datetime timestamps to epoch time for PyQtGraph
            epoch_times = []
            for ts in window_timestamps:
                epoch_times.append(datetime_to_epoch(ts))

            # Update data in plot
            if hasattr(self, 'speed_line'):
                self.speed_line.setData(epoch_times, window_values)
            elif hasattr(self, 'variance_line'):
                self.variance_line.setData(epoch_times, window_values)

            # Update x range to show all data
            if len(epoch_times) >= 2:
                self.plot_widget.setXRange(min(epoch_times), max(epoch_times))

                # Update y range based on data with some padding
                max_value = max(window_values) if window_values else (2 if hasattr(self, 'speed_line') else 1)
                min_y = 0
                max_y = max(2 if hasattr(self, 'speed_line') else 1, max_value * 1.2)
                self.plot_widget.setYRange(min_y, max_y)

            # Handle satiated region
            if satiated_range and satiated_range != getattr(self, 'last_satiated_range', None):
                self.last_satiated_range = satiated_range

                # Remove old region if it exists
                if hasattr(self, 'satiated_region') and self.satiated_region:
                    self.plot_widget.removeItem(self.satiated_region)

                # Create new region
                y_min, y_max = satiated_range
                self.satiated_region = pg.LinearRegionItem(
                    values=[y_min, y_max],
                    orientation=pg.LinearRegionItem.Horizontal,
                    brush=pg.mkBrush(0, 255, 0, 50),  # Semi-transparent green
                    movable=False
                )
                self.plot_widget.addItem(self.satiated_region)

            # Mark that we have data
            self.has_data = True

        except Exception as e:
            print(f"Error updating {graph_type.lower()} graph: {e}")
            import traceback
            traceback.print_exc()

    def add_missed_feeding_marker(self, timestamp):
        """Add a vertical line to indicate a missed feeding event"""
        if isinstance(timestamp, datetime.datetime):
            # Convert to epoch time
            epoch_time = (timestamp - datetime.datetime(1970, 1, 1)).total_seconds()

            # Create vertical line
            line = pg.InfiniteLine(
                pos=epoch_time,
                angle=90,
                pen=pg.mkPen(color='b', width=1, style=Qt.DashLine),
                label=f"Missed {timestamp.strftime('%H:%M')}"
            )

            # Add to plot and store reference
            self.plot_widget.addItem(line)
            self.markers.append(line)
            return line
        return None

    def clear_markers(self):
        """Remove all markers from the plot"""
        for marker in self.markers:
            self.plot_widget.removeItem(marker)
        self.markers = []

class FeedingHistoryTable(QWidget):
    """Widget for displaying feeding history using a proper table widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Initialize max visible rows
        self.max_visible_rows = 8  # Show 8 rows instead of just 5
        self.full_history = []  # Store the full history

        # Create table widget with grid lines
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Time", "Dosages"])

        # Set gridlines
        self.table.setShowGrid(True)
        self.table.setGridStyle(Qt.SolidLine)

        # Set header appearance
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft)
        header.setStretchLastSection(False)

        # Set column sizing
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Time column stretches
        header.setSectionResizeMode(1, QHeaderView.Fixed)    # Dosages column fixed width
        self.table.setColumnWidth(1, 70)  # Fixed width for dosages

        # Set alternating row colors
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f0f0f0;
                gridline-color: #c0c0c0;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 4px;
                border: 1px solid #c0c0c0;
                font-weight: bold;
            }
        """)

        # Configure row height - REDUCED to fit more rows
        self.table.verticalHeader().setVisible(False)  # Hide row numbers
        self.table.verticalHeader().setDefaultSectionSize(20)  # Make rows more compact

        self.layout.addWidget(self.table)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def update_history(self, feeding_history):
        """Update the feeding history table with proper timestamp handling"""
        print(f"update_history called with {len(feeding_history)} records")

        # Store the full history
        self.full_history = feeding_history

        # Clear table - IMPORTANT: Always reset row count to zero
        self.table.setRowCount(0)

        # Calculate how many rows to show
        rows_to_show = min(len(feeding_history), self.max_visible_rows if hasattr(self, 'max_visible_rows') else 10)

        # Add rows for each feeding event (most recent first)
        for i, feed in enumerate(feeding_history[:rows_to_show]):
            # Insert a new row
            self.table.insertRow(i)

            # Format time with robust error handling
            try:
                timestamp = feed['timestamp']
                # Handle different timestamp formats
                if isinstance(timestamp, datetime.datetime):
                    time_str = timestamp.strftime("%m/%d %H:%M")
                elif isinstance(timestamp, str):
                    # Try to parse string to datetime
                    try:
                        time_obj = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        time_str = time_obj.strftime("%m/%d %H:%M")
                    except ValueError:
                        # If parsing fails, try another common format
                        try:
                            time_obj = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                            time_str = time_obj.strftime("%m/%d %H:%M")
                        except ValueError:
                            # Just use the string as is
                            time_str = timestamp
                elif isinstance(timestamp, (int, float)) and timestamp > 1e9:
                    # Looks like an epoch timestamp
                    time_obj = datetime.datetime.fromtimestamp(timestamp)
                    time_str = time_obj.strftime("%m/%d %H:%M")
                else:
                    time_str = str(timestamp)
            except Exception as e:
                print(f"Error formatting timestamp: {e}")
                time_str = "Unknown"

            time_item = QTableWidgetItem(time_str)
            self.table.setItem(i, 0, time_item)

            # Display dosage count with error handling
            try:
                # Check if this was a missed feeding
                is_missed = feed.get('missed', False)

                if is_missed:
                    dosage_item = QTableWidgetItem("Missed")
                    dosage_item.setForeground(QColor(100, 100, 255))  # Blue for missed
                else:
                    # Get dosage count with fallback to 1
                    dosage_count = feed.get('dosage_count', 1)
                    dosage_item = QTableWidgetItem(str(dosage_count))

                dosage_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, 1, dosage_item)
            except Exception as e:
                print(f"Error setting dosage item: {e}")
                self.table.setItem(i, 1, QTableWidgetItem("?"))

        print(f"Table updated with {self.table.rowCount()} rows")

class DataLogger:
    """Class for logging fish metrics to CSV with dosage count"""

    def __init__(self):
        self.filename = os.path.join(DATA_FOLDER,
                                     f"fish_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        # Create file with headers - now with DosageCount column
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'AvgSpeed', 'SpeedVariance', 'FeedingEvent', 'DosageCount'])

    def log_data(self, timestamp, avg_speed, speed_variance, feeding_event=False, dosage_count=0):
        """Log a data point to CSV with dosage count"""
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                avg_speed,
                speed_variance,
                1 if feeding_event else 0,
                dosage_count
            ])

class SmartFishFeederApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Add after other initialisations
        self.direct_post_feeding_check_timer = QTimer()
        self.direct_post_feeding_check_timer.timeout.connect(self.direct_post_feeding_check)
        self.direct_post_feeding_check_timer.start(1000)  # Check every second

        # Enable exception handling for slots
        sys.excepthook = self.excepthook

        # Set window properties
        self.setWindowTitle("Smart Fish Feeder")
        self.setGeometry(100, 100, 1200, 800)

        try:
            # Check for model path
            if not os.path.exists(MODEL_PATH):
                print(f"WARNING: Model not found at {MODEL_PATH}")
                print("The application will attempt to use a default YOLO model for demonstration.")

            # Initialize components
            self.video_thread = VideoThread()
            self.feeding_model = ProphetFeedingModel()
            self.data_logger = DataLogger()
            self.video_thread.parent = self
            self.initialize_mode_tracking()
            self.feeding_active = False
            self.feed_start_time = None
            self.pre_feeding_speeds = []
            self.pre_feeding_variances = []
            self.during_feeding_speeds = []
            self.during_feeding_variances = []
            self.post_feeding_speeds = []
            self.post_feeding_variances = []
            self.updating_graphs = False
            self.system_mode = "initialising"  # initialising, monitoring, pre_feeding, feeding, post_feeding, cooldown
            self.last_feed_check = datetime.datetime.now()
            self.last_dosage_time = None
            self.dosage_count = 0
            self.cooldown_active = False
            self.cooldown_end_time = None

            self.setup_resource_monitoring()
            self.last_memory_log = datetime.datetime.now()

            self.setup_resource_monitoring()
            self.last_memory_log = datetime.datetime.now()

            # Check if we should be in cooldown based on last feeding time
            if self.feeding_model.get_feeding_history():
                last_feed = self.feeding_model.get_feeding_history()[-1]['timestamp']
                time_since_last_feed = (datetime.datetime.now() - last_feed).total_seconds()

                if time_since_last_feed < MIN_FEED_INTERVAL:
                    # Restore cooldown state
                    self.cooldown_active = True
                    self.cooldown_end_time = last_feed + datetime.timedelta(seconds=MIN_FEED_INTERVAL)
                    print(f"Restored cooldown period until {self.cooldown_end_time}")

            # Add a dosage response check timer
            self.dosage_timer = QTimer()
            self.dosage_timer.timeout.connect(self.check_dosage_response)

            # Track today's feedings
            self.today_date = datetime.datetime.now().date()
            self.today_feeding_count = 0

            # UI setup
            self.setup_ui()

            # Load today's feeding count from disk
            self.load_today_feeding_count()

            # Initialize display buffers (add this line)
            self.initialize_display_buffers()

            # Connect signals
            self.video_thread.frame_ready.connect(self.update_frame)

            # Start video processing
            self.video_thread.start()

            # Start decision timer
            self.decision_timer = QTimer()
            self.decision_timer.timeout.connect(self.check_feeding_decision)
            self.decision_timer.start(FEED_DECISION_INTERVAL * 1000)  # Convert to milliseconds

            # Start data display timer (updates graphs)
            self.display_timer = QTimer()
            self.display_timer.timeout.connect(self.update_display)
            self.display_timer.start(1000)  # Update every second

            # Schedule first feeding analysis
            QTimer.singleShot(INITIAL_FEED_DELAY * 1000, self.first_feeding_check)

            # For example, if you're using a QTimer.singleShot for initialisation:
            QTimer.singleShot(MONITORING_WINDOW * 1000, self.mark_initialisation_complete)

            # Daily timer to reset feeding count
            self.daily_timer = QTimer()
            self.daily_timer.timeout.connect(self.check_day_change)
            self.daily_timer.start(2000)  # Check every minute

            # Add these new variables for time-based comparison
            self.feeding_response_history = deque(maxlen=20)  # Stores last 20 speed/variance readings during feeding
            self.feeding_baseline_established = False
            self.baseline_speeds = []
            self.baseline_variances = []

            # Initialize feeding data collection arrays
            self.pre_feeding_data = []
            self.during_feeding_data = []
            self.post_feeding_data = []

            self.feeding_safety_timer = QTimer()
            self.feeding_safety_timer.timeout.connect(self.check_feeding_safety)
            self.feeding_safety_timer.start(30000)  # Check every 30 seconds

            # Add these new variables to the SmartFishFeederApp class in the __init__ method:
            self.was_outside_operating_hours = not self.is_within_operating_hours(datetime.datetime.now())
            self.last_operating_hours_state_change = datetime.datetime.now()

            self.initialisation_start_time = datetime.datetime.now()
            self.initialisation_complete = False
            QTimer.singleShot(4 * 60 * 60 * 1000, self.restart_critical_components)
            print("[MAINTENANCE] First component restart scheduled in 4 hours")

            # Initialize Prophet UI components
            self.setup_prophet_ui()

            # Add mode start timestamp tracking
            self.mode_start_times = {
                "initialising": datetime.datetime.now(),
                "monitoring": None,
                "pre_feeding": None,
                "feeding": None,
                "post_feeding": None,
                "cooldown": None
            }

            # In SmartFishFeederApp.__init__ after creating both components:
            self.feeding_model.video_thread = self.video_thread

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR in app initialisation: {e}")
            # Show error in UI when possible
            QMessageBox.critical(self, "initialisation Error",
                                 f"Failed to initialize the application: {str(e)}\n\n"
                                 "The application will continue with limited functionality.")

    def update_frame(self, frame, tracks, avg_speed, speed_variance):
        """Update the displayed video frame with tracking info - with reduced data storage"""
        # Log data
        current_time = datetime.datetime.now()
        self.data_logger.log_data(current_time, avg_speed, speed_variance,
                                  feeding_event=(self.system_mode == "feeding"))

        # Add data to 1-second display buffers
        self.speed_buffer.append(avg_speed)
        self.variance_buffer.append(speed_variance)

        # Only store in appropriate arrays if we're not just monitoring
        # This avoids creating separate redundant arrays in monitoring mode
        if self.system_mode == "pre_feeding":
            self.pre_feeding_speeds.append(avg_speed)
            self.pre_feeding_variances.append(speed_variance)
            # Track in feeding model with event type
            self.feeding_model.add_data_point(current_time, avg_speed, speed_variance, event_type="pre_feeding")
        elif self.system_mode == "feeding":
            self.during_feeding_speeds.append(avg_speed)
            self.during_feeding_variances.append(speed_variance)
            # Track in feeding model with event type
            self.feeding_model.add_data_point(current_time, avg_speed, speed_variance, event_type="during_feeding")
        elif self.system_mode == "post_feeding":
            self.post_feeding_speeds.append(avg_speed)
            self.post_feeding_variances.append(speed_variance)
            # Track in feeding model with event type
            self.feeding_model.add_data_point(current_time, avg_speed, speed_variance, event_type="post_feeding")
        elif self.system_mode == "monitoring" or self.system_mode == "initialising":
            # For monitoring, just add to the feeding model without creating separate arrays
            self.feeding_model.add_data_point(current_time, avg_speed, speed_variance)

        # Draw bounding boxes and trajectories
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        # Create a copy of the frame for drawing
        draw_frame = frame.copy()

        # Draw tracks
        for track_id, box, trajectory, color in tracks:
            x1, y1, x2, y2 = box
            # Draw bounding box
            cv2.rectangle(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

            # Draw trajectory
            points = trajectory[-30:]
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(draw_frame, points[i], points[i + 1], color, 1)
            # Print first trajectory point coordinates
            if DEBUG_MODE:
                if len(points) > 0:
                    print(f"First trajectory point: {points[0]}, Bounding box: {box}")

            # Convert to QImage
        q_img = QImage(draw_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Create and set pixmap
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(),
                                                 Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Add cooldown indicator if active
        if self.cooldown_active and self.cooldown_end_time:
            remaining_seconds = max(0, (self.cooldown_end_time - current_time).total_seconds())
            cooldown_percentage = remaining_seconds / MIN_FEED_INTERVAL
            cooldown_width = int(width * cooldown_percentage)

            # Draw cooldown progress bar
            cv2.rectangle(draw_frame, (0, height - 10), (cooldown_width, height), (0, 0, 255), -1)
            cv2.putText(draw_frame, f"Cooldown: {remaining_seconds:.0f}s / {MIN_FEED_INTERVAL}s",
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Update metrics display with 1-second average every second
        now = time.time()
        if now - self.last_display_update >= self.display_update_interval and len(self.speed_buffer) > 0:
            avg_speed_1s = np.mean(self.speed_buffer)
            avg_variance_1s = np.mean(self.variance_buffer)

            self.speed_value.setText(f"{avg_speed_1s:.2f}")
            self.variance_value.setText(f"{avg_variance_1s:.2f}")

            # Add color coding based on values
            if avg_speed_1s > 1.0:  # High activity
                self.speed_value.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF4500;")
            elif avg_speed_1s > 0.5:  # Medium activity
                self.speed_value.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFA500;")
            else:  # Low activity
                self.speed_value.setStyleSheet("font-size: 24px; font-weight: bold; color: #008000;")

            self.last_display_update = now

        # Update status based on feeding mode
        if self.system_mode == "monitoring":
            # Only update if not showing a feeding recommendation
            if not "FEEDING RECOMMENDED" in self.status_label.text():
                # For rolling window, show data points instead of time
                if hasattr(self.feeding_model, 'current_window_timestamps'):
                    data_points = len(self.feeding_model.current_window_timestamps)
                    fullness = min(100, int((data_points / MONITORING_WINDOW) * 100))

                    if fullness < 100:
                        self.status_label.setText(f"Monitoring fish behavior... Building data: {fullness}% complete")
                    else:
                        self.status_label.setText("Monitoring fish behavior using 5-minute rolling window")
                else:
                    self.status_label.setText("initialising monitoring system...")
        elif self.system_mode == "pre_feeding":
            self.status_label.setText("Pre-feeding analysis...")
        elif self.system_mode == "feeding":
            self.status_label.setText("Feeding in progress...")
        elif self.system_mode == "post_feeding":
            self.status_label.setText("Post-feeding analysis...")

    def check_day_change(self):
        """Check if day has changed to reset feeding count"""
        current_date = datetime.datetime.now().date()
        if current_date != self.today_date:
            self.today_date = current_date
            self.today_feeding_count = 0
            self.update_feeding_count_display()
            # Save the reset count
            self.save_today_feeding_count()

    def update_feeding_count_display(self):
        """Update the feeding count display"""
        if not hasattr(self, 'feeding_count_label'):
            # Label doesn't exist yet, skip update
            return

        self.feeding_count_label.setText(f"Today's feedings: {self.today_feeding_count}")

        # Add warning if approaching max daily feedings
        if self.today_feeding_count >= 4:
            self.feeding_count_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.feeding_count_label.setStyleSheet("")

    @staticmethod
    def excepthook(self, exc_type, exc_value, exc_tb):
        """Global exception handler for PyQt slots"""
        try:
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"UNHANDLED EXCEPTION:\n{tb}")
            # Continue normal exception processing
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception as e:
            print(f"Error in excepthook: {e}")
            # Basic fallback if traceback formatting fails
            print(f"Original exception: {exc_type.__name__}: {exc_value}")

    def setup_ui(self):
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QGridLayout()
        main_widget.setLayout(main_layout)

        # Video display area
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid #cccccc; background-color: #000000;")

        # Control panel
        control_panel = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)

        # Prophet Dashboard
        self.prophet_dashboard_button = QPushButton("Feeding Forecast")
        self.prophet_dashboard_button.clicked.connect(self.create_prophet_dashboard)
        control_layout.addWidget(self.prophet_dashboard_button)

        # First create the feed button (before trying to reference it)
        self.feed_button = QPushButton("CONFIRM FEED")
        self.feed_button.setMinimumHeight(80)
        self.feed_button.setStyleSheet("background-color: #808080; color: white; font-size: 18px; font-weight: bold;")
        self.feed_button.clicked.connect(self.confirm_dosage)
        self.feed_button.setEnabled(False)  # Disabled by default

        # Simplified status box
        status_box = QGroupBox("System Status")
        status_layout = QVBoxLayout()

        control_layout.setSpacing(5)  # Reduce padding between elements
        control_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # Dosage counter (only visible during feeding)
        self.dosage_display = QLabel("")
        self.dosage_display.setStyleSheet("font-size: 16px;")

        status_layout.addWidget(self.dosage_display)
        status_box.setLayout(status_layout)

        # Status label
        self.status_label = QLabel("System starting...")
        self.status_label.setStyleSheet("font-size: 14px;")
        self.status_label.setWordWrap(True)

        # Feeding count label
        self.feeding_count_label = QLabel("Today's feedings: 0")

        # Now add the widgets to the layout in the correct order
        control_layout.addWidget(status_box)
        control_layout.addWidget(self.feed_button)
        control_layout.addWidget(self.feeding_count_label)

        # Metrics display
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(2)  # Minimal spacing

        speed_box = QWidget()
        speed_layout = QHBoxLayout(speed_box)
        speed_layout.setContentsMargins(2, 2, 2, 2)
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_value = QLabel("0.00")
        self.speed_value.setStyleSheet("font-size: 16px; font-weight: bold;")
        speed_layout.addWidget(self.speed_value)

        # Create a compact variance display
        variance_box = QWidget()
        variance_layout = QHBoxLayout(variance_box)
        variance_layout.setContentsMargins(2, 2, 2, 2)
        variance_layout.addWidget(QLabel("Var:"))
        self.variance_value = QLabel("0.00")
        self.variance_value.setStyleSheet("font-size: 16px; font-weight: bold;")
        variance_layout.addWidget(self.variance_value)

        metrics_layout.addWidget(speed_box)
        metrics_layout.addWidget(variance_box)

        # Add to control layout
        control_layout.addWidget(self.feed_button)
        control_layout.addWidget(self.feeding_count_label)
        control_layout.addLayout(metrics_layout)
        control_layout.addStretch()

        # Graphs
        self.speed_graph = SpeedGraph(self)
        self.variance_graph = VarianceGraph(self)

        # Feeding history
        history_box = QGroupBox("Feeding History")
        history_layout = QVBoxLayout()
        self.feeding_history_table = FeedingHistoryTable()
        history_layout.addWidget(self.feeding_history_table)
        history_box.setLayout(history_layout)

        # Add widgets to main layout
        main_layout.addWidget(self.video_label, 0, 0, 1, 2)
        main_layout.addWidget(control_panel, 0, 2, 1, 1)
        main_layout.addWidget(self.speed_graph, 1, 0, 1, 1)
        main_layout.addWidget(self.variance_graph, 1, 1, 1, 1)
        main_layout.addWidget(history_box, 1, 2, 1, 1)

        main_layout.setRowStretch(0, 2)  # Give more space to video
        main_layout.setRowStretch(1, 1)  # More space for graphs

        # Set column stretching
        main_layout.setColumnStretch(0, 2)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 1)

        # Status indicator labels
        self.mode_label = QLabel("Mode: INITIALISING")
        self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.dosage_label = QLabel("Dosage: 0")
        self.dosage_label.setStyleSheet("font-size: 16px;")

        self.time_label = QLabel("Time in mode: 00:00")
        self.time_label.setStyleSheet("font-size: 14px;")

        status_layout.addWidget(self.mode_label)
        status_layout.addWidget(self.dosage_label)
        status_layout.addWidget(self.time_label)
        status_box.setLayout(status_layout)

        # Add status update timer
        self.status_update_timer = QTimer()
        self.status_update_timer.timeout.connect(self.update_status_display)
        self.status_update_timer.start(1000)  # Update every second

        control_layout.addWidget(status_box)
        control_layout.addWidget(self.feed_button)
        control_layout.addWidget(self.status_label)

    def update_display(self):
        """Update the UI displays with optimized data handling"""
        # Add post-feeding completion check
        current_time = datetime.datetime.now()

        # IMPORTANT: Add this to prevent too frequent updates
        if hasattr(self, 'last_display_update_time'):
            if time.time() - self.last_display_update_time < 3.0:  # Only update every 3 seconds
                self.updating_graphs = False  # Clear flag immediately
                return
        self.last_display_update_time = time.time()

        self.updating_graphs = True  # Set flag to indicate graph updates are happening

        # Reduce update frequency during initialization
        if self.system_mode == "initialising":
            if not hasattr(self, 'last_init_graph_update'):
                self.last_init_graph_update = 0

            # During initialization, only update graphs every 5 seconds
            if time.time() - self.last_init_graph_update < 5.0:  # 5 second minimum between updates
                self.updating_graphs = False  # Clear flag immediately
                return  # Skip this update entirely

            self.last_init_graph_update = time.time()

        try:
            # Check if post-feeding mode should be completed
            if (self.system_mode == "post_feeding" and
                    hasattr(self, 'post_feeding_end_time') and
                    current_time >= self.post_feeding_end_time and
                    not getattr(self, 'post_feeding_completion_triggered', False)):
                print(f"[DISPLAY-CHECK] Post-feeding end time reached: {current_time} >= {self.post_feeding_end_time}")
                self.post_feeding_completion_triggered = True  # Prevent multiple calls

                # Use a very short timer to complete after this update finishes
                QTimer.singleShot(100, self.force_complete_feeding_cycle)
                print("[DISPLAY-CHECK] Scheduled immediate completion")

            # For live graphs, get data from video thread with time window limitation
            timestamps, speeds, variances = self.video_thread.get_speed_data(window_seconds=300)  # Last 5 minutes

            # Get satiated ranges from model
            speed_range, variance_range = self.feeding_model.get_satiated_ranges()

            # Only update graphs if we have enough data
            if len(timestamps) >= 2 and len(speeds) >= 2 and len(variances) >= 2:
                # Update graphs with deep copies to avoid modifying original data
                self.speed_graph.update_plot(
                    timestamps.copy() if isinstance(timestamps, list) else list(timestamps),
                    speeds.copy() if isinstance(speeds, list) else list(speeds),
                    speed_range
                )

                self.variance_graph.update_plot(
                    timestamps.copy() if isinstance(timestamps, list) else list(timestamps),
                    variances.copy() if isinstance(variances, list) else list(variances),
                    variance_range
                )

                # Synchronize graph time windows
                self.synchronize_graph_windows()

                # Mark missed feeding events on graphs with blue vertical lines
                missed_feedings = self.feeding_model.get_missed_feedings()
                if missed_feedings:
                    for missed in missed_feedings:
                        # Only show missed feedings that are within our current view
                        if timestamps and missed['timestamp'] >= timestamps[0]:
                            # Convert timestamp to matplotlib format
                            import matplotlib.dates
                            event_time = matplotlib.dates.date2num(missed['timestamp'])

                            # Add blue dotted line on both graphs
                            if hasattr(self, 'speed_graph') and self.speed_graph.plot_widget:
                                self.speed_graph.plot_widget.axvline(x=event_time, color='blue', linestyle='--', alpha=0.5)
                            if hasattr(self, 'variance_graph') and self.variance_graph.axes:
                                self.variance_graph.axes.axvline(x=event_time, color='blue', linestyle='--', alpha=0.5)

                    # Force redraw of graphs
                    if hasattr(self, 'speed_graph'):
                        self.speed_graph.fig.canvas.draw()
                    if hasattr(self, 'variance_graph'):
                        self.variance_graph.fig.canvas.draw()

            # Update feeding history - include both regular and missed feedings
            all_feedings = self.feeding_model.get_feeding_history().copy()
            missed_feedings = self.feeding_model.get_missed_feedings()
            if missed_feedings:
                # Add missed feedings with appropriate formatting
                for missed in missed_feedings:
                    feed_record = {
                        'timestamp': missed['timestamp'],
                        'features': missed['features'],
                        'dosage_count': 0,
                        'missed': True
                    }
                    all_feedings.append(feed_record)

                # Sort by timestamp (most recent first)
                all_feedings.sort(key=lambda x: x['timestamp'], reverse=True)

                # Update the feeding history table with all feedings
                self.feeding_history_table.update_history(all_feedings)
            else:
                # Just update with regular feeding history
                self.feeding_history_table.update_history(self.feeding_model.get_feeding_history())

            # Check if new day started
            self.check_day_change()

        except Exception as e:
            print(f"Error in update_display: {e}")
            traceback.print_exc()

        finally:
            # Clear flag when done - VERY IMPORTANT
            self.updating_graphs = False

    def direct_post_feeding_check(self):
        """Direct method to check and enforce post-feeding completion"""
        if self.system_mode != "post_feeding":
            return  # Only relevant for post-feeding mode

        current_time = datetime.datetime.now()
        if not hasattr(self, 'mode_start_times') or "post_feeding" not in self.mode_start_times:
            return  # No valid start time

        post_start_time = self.mode_start_times["post_feeding"]
        elapsed_seconds = (current_time - post_start_time).total_seconds()

        # Print status every 5 seconds for debugging
        if int(elapsed_seconds) % 5 == 0:
            print(
                f"[DIRECT-CHECK] Post-feeding active for {elapsed_seconds:.1f} seconds (limit: {POST_FEEDING_DURATION})")

        # If we've exceeded the post-feeding duration, force completion
        if elapsed_seconds >= POST_FEEDING_DURATION:
            print(f"[DIRECT-CHECK] Post-feeding duration exceeded: {elapsed_seconds:.1f} >= {POST_FEEDING_DURATION}")
            print(f"[DIRECT-CHECK] FORCING completion NOW at {current_time}")

            # Call complete_feeding_cycle directly
            self.complete_feeding_cycle()

            return True  # Indicates we took action

        return False  # No action needed

    def update_status_display(self):
        """Update the status display with current mode, time, and dosage information"""
        current_time = datetime.datetime.now()

        # First, ensure cooldown state and mode are consistent
        if self.cooldown_active and self.cooldown_end_time and current_time < self.cooldown_end_time:
            if self.system_mode != "cooldown":
                print(f"[DISPLAY] Correcting mode from {self.system_mode} to cooldown due to active cooldown")
                self.system_mode = "cooldown"
                if "monitoring" in self.mode_start_times and self.mode_start_times["monitoring"]:
                    self.mode_start_times["cooldown"] = self.mode_start_times["monitoring"]

            # Always update the status during cooldown (not just when it shows "FEEDING RECOMMENDED")
            remaining_minutes = int((self.cooldown_end_time - current_time).total_seconds() / 60)
            self.status_label.setText(f"Cooldown active. Next feeding available in {remaining_minutes} minutes")
            self.status_label.setStyleSheet("")  # Reset style

        # Make sure any "FEEDING RECOMMENDED" message is cleared during cooldown
        if self.cooldown_active and self.cooldown_end_time:
            if "FEEDING RECOMMENDED" in self.status_label.text():
                remaining_minutes = int((self.cooldown_end_time - current_time).total_seconds() / 60)
                self.status_label.setText(f"Cooldown active. Next feeding available in {remaining_minutes} minutes")
                self.status_label.setStyleSheet("")  # Reset style

        """Update the status display with current mode, time, and dosage information"""
        current_time = datetime.datetime.now()

        # Update mode label with color coding
        if self.system_mode == "initialising":
            self.mode_label.setText("Mode: INITIALISING")
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: purple;")

            # Calculate elapsed initialisation time
            elapsed_time = (current_time - self.mode_start_times["initialising"]).total_seconds()
            remaining = max(0, MONITORING_WINDOW - elapsed_time)

            if remaining > 0:
                # Show countdown
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                self.time_label.setText(f"Initialising: {minutes:02d}:{seconds:02d} remaining")
                self.status_label.setText("System initialisation in progress. Please wait...")
            else:
                # initialisation complete - transition to monitoring
                self.system_mode = "monitoring"
                self.mode_start_times["monitoring"] = current_time
                self.time_label.setText("initialisation complete. Using rolling data window.")
                self.status_label.setText("Monitoring fish behavior. Using 5-minute rolling window.")
                # Now call the method again to update the UI for monitoring mode
                self.update_status_display()
                return

        elif self.system_mode == "monitoring":
            self.mode_label.setText("Mode: MONITORING")
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")

            # Show time until next feed check
            if hasattr(self, 'last_feed_check'):
                time_since_check = (current_time - self.last_feed_check).total_seconds()
                next_check_in = max(0, FEED_DECISION_INTERVAL - time_since_check)
                next_min = int(next_check_in // 60)
                next_sec = int(next_check_in % 60)
                self.time_label.setText(f"Next feeding check: {next_min:02d}:{next_sec:02d}")
            else:
                self.time_label.setText("Using 5-minute rolling data window")

        elif self.system_mode == "cooldown":
            self.mode_label.setText("Mode: COOLDOWN")
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")

            # Show cooldown remaining time
            if self.cooldown_end_time:
                remaining_time = max(0, (self.cooldown_end_time - current_time).total_seconds())
                minutes = int(remaining_time // 60)
                seconds = int(remaining_time % 60)
                self.time_label.setText(f"Cooldown remaining: {minutes:02d}:{seconds:02d}")

        elif self.system_mode == "pre_feeding":
            self.mode_label.setText("Mode: PRE-FEEDING ANALYSIS")
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: orange;")

            # Show time in pre-feeding mode
            elapsed_time = (current_time - self.mode_start_times["pre_feeding"]).total_seconds()
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            remaining = max(0, PRE_FEEDING_DURATION - elapsed_time)
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)
            self.time_label.setText(
                f"Pre-feeding: {rem_min:02d}:{rem_sec:02d}) remaining")

        elif self.system_mode == "feeding":
            self.mode_label.setText("Mode: ACTIVE FEEDING")
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")

            # Show time in feeding mode
            elapsed_time = (current_time - self.mode_start_times["feeding"]).total_seconds()

            # If no dosage yet, show dosage timeout countdown
            if self.dosage_count == 0:
                dosage_remaining = max(0, MAX_DURING_mode_NO_DOSAGE - elapsed_time)
                d_minutes = int(dosage_remaining // 60)
                d_seconds = int(dosage_remaining % 60)
                self.time_label.setText(f"Awaiting dosage: {d_minutes:02d}:{d_seconds:02d} (Feed now!)")
            else:
                # Show countdown to automatic completion
                remaining = max(0, MAX_FEEDING_DURATION - elapsed_time)
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                self.time_label.setText(f"Feeding timeout: {minutes:02d}:{seconds:02d} remaining")

        elif self.system_mode == "post_feeding":
            self.mode_label.setText("Mode: POST-FEEDING ANALYSIS")
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: purple;")

            # Show time in post-feeding mode with countdown
            elapsed_time = (current_time - self.mode_start_times["post_feeding"]).total_seconds()

            # Calculate remaining time (countdown)
            remaining = max(0, POST_FEEDING_DURATION - elapsed_time)
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)

            # Display as countdown
            self.time_label.setText(f"Post-feeding: {rem_min:02d}:{rem_sec:02d} remaining")

        # Update dosage counter
        self.dosage_label.setText(f"Dosage: {self.dosage_count}")

    def confirm_dosage(self):
        """Confirm that a dosage of food has been dispensed"""
        current_time = datetime.datetime.now()

        # Only proceed if we're in the "feeding" feeding mode
        if self.system_mode != "feeding":
            return

        # Record the dosage time
        self.last_dosage_time = current_time
        self.dosage_count += 1

        # Update dosage display
        self.dosage_label.setText(f"Dosage: {self.dosage_count}")

        # Disable the button until next dosage recommendation
        self.feed_button.setEnabled(False)
        self.feed_button.setStyleSheet("background-color: #808080; color: white; font-size: 18px; font-weight: bold;")

        # Set a timer to check fish response after the assessment period
        self.dosage_timer.start(DOSAGE_ASSESSMENT_PERIOD * 1000)

        # Update UI
        self.status_label.setText(f"Dosage #{self.dosage_count} administered. Monitoring response...")

        # Log this specific dosage event with current count
        self.data_logger.log_data(
            current_time,
            self.speed_buffer[-1] if self.speed_buffer else 0,
            self.variance_buffer[-1] if self.variance_buffer else 0,
            feeding_event=True,
            dosage_count=self.dosage_count
        )

        # In a real system, you might log each dosage
        print(f"Dosage #{self.dosage_count} confirmed at {current_time}")

    def check_dosage_response(self):
        """Check if fish have responded to the last dosage using time-based comparisons"""
        self.dosage_timer.stop()

        # Safety measure: maximum dosages
        MAX_DOSAGES_PER_FEEDING = 10  # Adjust based on your fish needs
        if self.dosage_count >= MAX_DOSAGES_PER_FEEDING:
            self.status_label.setText(f"Maximum dosage limit reached ({MAX_DOSAGES_PER_FEEDING}).")
            self.end_active_feeding()
            return

        # Safety measure: maximum feeding duration
        current_time = datetime.datetime.now()
        feeding_duration = (current_time - self.feed_start_time).total_seconds()
        if feeding_duration > MAX_FEEDING_DURATION:
            self.status_label.setText(f"Maximum feeding duration reached ({MAX_FEEDING_DURATION / 60:.0f} minutes).")
            self.end_active_feeding()
            return

        # Get recent activity data (last 30 seconds)
        recent_speeds = self.during_feeding_speeds[-30:] if len(
            self.during_feeding_speeds) > 30 else self.during_feeding_speeds
        recent_variances = self.during_feeding_variances[-30:] if len(
            self.during_feeding_variances) > 30 else self.during_feeding_variances

        if not recent_speeds:
            # Not enough data, continue feeding mode
            return

        # Calculate current metrics
        avg_speed = np.mean(recent_speeds)
        avg_variance = np.mean(recent_variances)

        # Add to response history
        self.feeding_response_history.append((avg_speed, avg_variance))

        # Establish baseline if needed (after first dosage)
        if not self.feeding_baseline_established and len(self.feeding_response_history) >= 1:
            self.baseline_speeds = [speed for speed, _ in list(self.feeding_response_history)]
            self.baseline_variances = [var for _, var in list(self.feeding_response_history)]
            self.feeding_baseline_established = True

            # Always recommend a second dosage to establish response pattern
            if self.dosage_count == 1:
                self.recommend_another_dosage()
                return

        # Only perform satiation analysis if we have enough history and baseline
        if len(self.feeding_response_history) >= 3 and self.feeding_baseline_established:
            # Time-based comparison - look at rate of change in activity
            # Get activity changes over time periods
            current_speed = avg_speed
            current_variance = avg_variance

            # Get baseline metrics (from beginning of feeding)
            baseline_speed = np.mean(self.baseline_speeds)
            baseline_variance = np.mean(self.baseline_variances)

            # Get metrics from middle of the response history
            mid_point = len(self.feeding_response_history) // 2
            mid_speeds = [speed for speed, _ in list(self.feeding_response_history)[mid_point:mid_point + 3]]
            mid_variances = [var for _, var in list(self.feeding_response_history)[mid_point:mid_point + 3]]
            mid_speed = np.mean(mid_speeds) if mid_speeds else baseline_speed
            mid_variance = np.mean(mid_variances) if mid_variances else baseline_variance

            # Calculate rates of change
            # Rate 1: Baseline to middle
            if mid_point > 0:
                rate1_speed = (mid_speed - baseline_speed) / mid_point
                rate1_variance = (mid_variance - baseline_variance) / mid_point
            else:
                rate1_speed = 0
                rate1_variance = 0

            # Rate 2: Middle to current
            recent_count = len(self.feeding_response_history) - mid_point
            if recent_count > 0:
                rate2_speed = (current_speed - mid_speed) / recent_count
                rate2_variance = (current_variance - mid_variance) / recent_count
            else:
                rate2_speed = 0
                rate2_variance = 0

            # Rate of change is slowing if rate2 < rate1
            speed_slowing = rate2_speed < rate1_speed
            variance_slowing = rate2_variance < rate1_variance

            # Additional metrics - overall activity trend
            recent_activity_trend = (current_speed / baseline_speed) if baseline_speed > 0 else 1.0

            # Log the analysis
            if DEBUG_MODE:
                print(f"Feeding response analysis:")
                print(f"  Current speed: {current_speed:.2f}, variance: {current_variance:.2f}")
                print(f"  Baseline speed: {baseline_speed:.2f}, variance: {baseline_variance:.2f}")
                print(f"  Speed rates: R1={rate1_speed:.4f}, R2={rate2_speed:.4f}, Slowing={speed_slowing}")
                print(f"  Variance rates: R1={rate1_variance:.4f}, R2={rate2_variance:.4f}, Slowing={variance_slowing}")
                print(f"  Activity trend: {recent_activity_trend:.2f}x baseline")

            # Decision making based on activity patterns
            satiated = False

            # Condition 1: Rate of change is slowing down in both metrics
            if speed_slowing and variance_slowing:
                satiated = True
                reason = "Activity response is slowing down"

            # Condition 2: Activity has decreased below a threshold compared to peak
            # Find maximum activity level observed during feeding
            max_speeds = max(speed for speed, _ in self.feeding_response_history)
            current_vs_max = current_speed / max_speeds if max_speeds > 0 else 1.0
            if current_vs_max < 0.7:  # Activity decreased by 30% from peak
                satiated = True
                reason = f"Activity decreased to {current_vs_max:.0%} of peak"

            # Condition 3: Activity has remained flat for multiple checks
            last_speeds = [speed for speed, _ in list(self.feeding_response_history)[-3:]]
            if len(last_speeds) >= 3:
                speed_changes = [abs(last_speeds[i] - last_speeds[i - 1]) / last_speeds[i - 1]
                                 if last_speeds[i - 1] > 0 else 0
                                 for i in range(1, len(last_speeds))]
                if all(change < 0.1 for change in speed_changes):  # Less than 10% change
                    satiated = True
                    reason = "Activity has stabilized"

            # If satiated, end feeding
            if satiated:
                self.status_label.setText(f"Fish appear satiated: {reason}. Ending feeding cycle.")
                self.end_active_feeding()
            else:
                # Not yet satiated, recommend another dosage
                self.recommend_another_dosage()
        else:
            # Not enough data yet, recommend another dosage by default
            self.recommend_another_dosage()

    def recommend_another_dosage(self):
        """Recommend another dosage of food"""
        self.status_label.setText("ADDITIONAL DOSAGE RECOMMENDED - Press CONFIRM button")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.feed_button.setStyleSheet("background-color: #FF4500; color: white; font-size: 18px; font-weight: bold;")
        self.feed_button.setEnabled(True)

        # Play a sound alert
        QApplication.beep()

    def start_feeding_cycle(self, current_time):
        """Begin a feeding cycle - pre, during, post modes"""
        print(f"Starting feeding cycle at {current_time} - beginning with pre-feeding analysis")

        # Set feeding state
        self.system_mode = "pre_feeding"
        self.feed_start_time = current_time
        self.mode_start_times["pre_feeding"] = current_time

        # Reset counters and history
        self.dosage_count = 0
        self.pre_feeding_speeds = []
        self.pre_feeding_variances = []
        self.during_feeding_speeds = []
        self.during_feeding_variances = []
        self.post_feeding_speeds = []
        self.post_feeding_variances = []

        # Important: Make sure feed button is disabled during pre-feeding
        self.feed_button.setEnabled(False)
        self.feed_button.setText("WAIT: PRE-FEEDING ANALYSIS")
        self.feed_button.setStyleSheet("background-color: #808080; color: white; font-size: 18px; font-weight: bold;")

        # Update UI
        self.mode_label.setText("Mode: PRE-FEEDING ANALYSIS")
        self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: orange;")
        self.status_label.setText("Starting 5-minute pre-feeding analysis...")
        self.status_label.setStyleSheet("")

        # Make sure we're collecting pre-feeding data
        self.pre_feeding_timer = QTimer()
        self.pre_feeding_timer.timeout.connect(self.collect_pre_feeding_data)
        self.pre_feeding_timer.start(1000)  # Collect once per second

        # IMPORTANT: Schedule the transition to active feeding mode
        QTimer.singleShot(PRE_FEEDING_DURATION * 1000, self.start_active_feeding)

        print(f"Pre-feeding analysis will run for {PRE_FEEDING_DURATION} seconds")

    def collect_pre_feeding_data(self):
        """Collect data during pre-feeding mode"""
        timestamp = datetime.datetime.now()
        # Get current speed and variance from the most recent data
        speed = self.speed_buffer[-1] if self.speed_buffer else 0
        variance = self.variance_buffer[-1] if self.variance_buffer else 0

        data_point = {
            "timestamp": timestamp,
            "speed": speed,
            "variance": variance
        }

        self.pre_feeding_data.append(data_point)
        # Tag this as important data to keep
        self.video_thread.add_data_point(timestamp, speed, variance, "pre_feeding")

    def start_active_feeding(self):
        """Start the active feeding mode after pre-feeding analysis"""
        print("Transitioning from pre-feeding to active feeding mode")

        # Stop the pre-feeding data collection timer
        if hasattr(self, 'pre_feeding_timer') and self.pre_feeding_timer.isActive():
            self.pre_feeding_timer.stop()
            print("Stopped pre-feeding timer")

        # Start the during-feeding mode
        self.system_mode = "feeding"
        self.mode_start_times["feeding"] = datetime.datetime.now()

        # Ensure we reset the dosage counter when starting a new feeding
        self.dosage_count = 0
        self.dosage_label.setText(f"Dosage: {self.dosage_count}")

        # IMPORTANT: Make sure feed button is properly connected before enabling
        try:
            # Disconnect any existing connections
            self.feed_button.clicked.disconnect()
        except:
            pass

        # Connect to the confirm_dosage method
        self.feed_button.clicked.connect(self.confirm_dosage)
        print("Feed button connected to confirm_dosage method")

        # Now enable the button and update its appearance
        self.feed_button.setText("CONFIRM FEED")
        self.feed_button.setStyleSheet("background-color: #FF4500; color: white; font-size: 18px; font-weight: bold;")
        self.feed_button.setEnabled(True)
        print("Feed button enabled and styled for confirmation")

        # Set up collection timer for during-feeding data
        self.during_feeding_timer = QTimer()
        self.during_feeding_timer.timeout.connect(self.collect_during_feeding_data)
        self.during_feeding_timer.start(1000)  # Collect once per second

        # Update UI
        self.status_label.setText("FEED NOW - Press CONFIRM FEED button when food is dispensed")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")

        # IMPORTANT: Log the time when we CONFIRM FEED mode
        print(
            f"[ACTIVE] Starting active feeding mode at {datetime.datetime.now()} with MAX_DURING_mode_NO_DOSAGE={MAX_DURING_mode_NO_DOSAGE}s")

        # CRITICAL: Create a reliable safety timeout for missed feedings
        # Use an actual QTimer instead of singleShot for more reliability
        self.feeding_timeout_timer = QTimer()
        self.feeding_timeout_timer.setSingleShot(True)
        self.feeding_timeout_timer.timeout.connect(self.check_feeding_timeout)
        self.feeding_timeout_timer.start(MAX_DURING_mode_NO_DOSAGE * 1000)

        print(f"[TIMER] Started feeding timeout timer for {MAX_DURING_mode_NO_DOSAGE}s")

        # Add a safety timer to ensure we eventually complete the feeding cycle
        # Also use a regular QTimer for reliability
        self.max_feeding_timer = QTimer()
        self.max_feeding_timer.setSingleShot(True)
        self.max_feeding_timer.timeout.connect(lambda: self.check_feeding_safety())
        self.max_feeding_timer.start(MAX_FEEDING_DURATION * 1000)

        print(f"[TIMER] Started max feeding duration timer for {MAX_FEEDING_DURATION}s")

        # Play a sound alert
        QApplication.beep()

    def check_incomplete_feeding(self):
        """Check if feeding cycle is incomplete and fix if needed"""
        current_time = datetime.datetime.now()
        if self.system_mode == "feeding" and self.dosage_count > 0:
            print(f"[SAFETY] Feeding mode running too long - transitioning to post-feeding")
            self.end_active_feeding()
        elif self.system_mode == "post_feeding":
            # Post mode running too long, ensure it completes
            elapsed_time = (current_time - self.mode_start_times.get("post_feeding", current_time)).total_seconds()
            if elapsed_time > POST_FEEDING_DURATION:
                print(f"[SAFETY] Post-feeding mode running too long ({elapsed_time}s) - completing cycle")
                self.complete_feeding_cycle()

    def collect_during_feeding_data(self):
        """Collect data during active feeding mode"""
        timestamp = datetime.datetime.now()
        # Get current speed and variance from the most recent data
        speed = self.speed_buffer[-1] if self.speed_buffer else 0
        variance = self.variance_buffer[-1] if self.variance_buffer else 0

        data_point = {
            "timestamp": timestamp,
            "speed": speed,
            "variance": variance
        }

        self.during_feeding_data.append(data_point)
        # Tag this as important data to keep
        self.video_thread.add_data_point(timestamp, speed, variance, "during_feeding")

    def check_feeding_timeout(self):
        """Timeout handler that properly records missed feedings as valid hunger events and ensures persistence"""
        current_time = datetime.datetime.now()

        if self.system_mode == "feeding" and self.dosage_count == 0:
            print(
                f"[TIMEOUT] Feeding timeout triggered at {current_time} - no dosage confirmed in {MAX_DURING_mode_NO_DOSAGE} seconds")

            # This was a missed feeding opportunity (user unavailable)
            self.status_label.setText("Feeding recommendation expired - recording as missed feeding")

            # Record as a valid missed feeding (not a false positive)
            self.feeding_model.add_missed_feeding(
                self.feed_start_time,
                self.pre_feeding_speeds,
                self.pre_feeding_variances
            )

            # Use this data to enhance the ML model
            self.feeding_model.analyze_hunger_from_missed_feedings()

            # CRITICAL: Force save to ensure persistence
            if hasattr(self.feeding_model, 'save_model'):
                self.feeding_model.save_model()
                print("[PERSISTENCE] Forced save of missed feeding data")

            # Start a shorter cooldown period
            self.cooldown_active = True
            # 30 minutes cooldown for missed feedings
            self.cooldown_end_time = current_time + datetime.timedelta(seconds=1800)
            print(f"[COOLDOWN] Activated shorter cooldown until {self.cooldown_end_time}")

            # CRITICAL: Completely reset the feeding state
            self.system_mode = "monitoring"
            self.mode_start_times["monitoring"] = current_time

            # Clear ALL data buffers to prevent memory issues
            self.pre_feeding_speeds = []
            self.pre_feeding_variances = []
            self.during_feeding_speeds = []
            self.during_feeding_variances = []
            self.post_feeding_speeds = []
            self.post_feeding_variances = []

            # Reset any cached data in the feeding model
            if hasattr(self.feeding_model, '_cache'):
                for key in self.feeding_model._cache:
                    self.feeding_model._cache[key] = None

            # Perform targeted garbage collection
            import gc
            gc.collect()

            # Ensure matplotlib resources are released
            if hasattr(self, 'speed_graph') and hasattr(self.speed_graph, 'fig'):
                plt.close(self.speed_graph.plot_widget)

            # Reset feeding response data
            if hasattr(self, 'feeding_response_history'):
                self.feeding_response_history.clear()
            self.feeding_baseline_established = False

            # Stop ALL timers that might be running
            if hasattr(self, 'pre_feeding_timer') and self.pre_feeding_timer.isActive():
                self.pre_feeding_timer.stop()
                print("[TIMER] Stopped pre-feeding timer")
            if hasattr(self, 'during_feeding_timer') and self.during_feeding_timer.isActive():
                self.during_feeding_timer.stop()
                print("[TIMER] Stopped during-feeding timer")
            if hasattr(self, 'post_feeding_timer') and self.post_feeding_timer.isActive():
                self.post_feeding_timer.stop()
                print("[TIMER] Stopped post-feeding timer")
            if hasattr(self, 'dosage_timer') and self.dosage_timer.isActive():
                self.dosage_timer.stop()
                print("[TIMER] Stopped dosage timer")

            # VERY IMPORTANT: Remove ALL timer flags that might prevent future timers
            if hasattr(self, 'post_feeding_timer_set'):
                delattr(self, 'post_feeding_timer_set')

            # Update button and UI state
            self.feed_button.setEnabled(False)
            self.feed_button.setStyleSheet(
                "background-color: #808080; color: white; font-size: 18px; font-weight: bold;")
            self.status_label.setText("Missed feeding recorded. Continuing monitoring with shorter cooldown (30min).")
            self.mode_label.setText("mode: MONITORING")
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")

            # Force garbage collection to help prevent memory issues
            import gc
            gc.collect()
            print("[MEMORY] Forced garbage collection")

            print(f"[COMPLETE] Feeding cycle reset after missed feeding at {current_time}")

        elif self.system_mode == "feeding" and self.dosage_count > 0:
            # Normal timeout with some dosages already confirmed
            print(f"[TIMEOUT] Feeding timeout with {self.dosage_count} dosages - transitioning to post-feeding")
            self.status_label.setText("Maximum feeding duration reached. Ending feeding cycle.")
            self.end_active_feeding()

    def end_active_feeding(self):
        """End the active feeding mode and start post-feeding analysis"""
        current_time = datetime.datetime.now()
        print(f"[DEBUG] Ending active feeding at {current_time}, transitioning to post-feeding mode")

        # Stop any active timers from the feeding mode
        if hasattr(self, 'feeding_timeout_timer') and self.feeding_timeout_timer.isActive():
            self.feeding_timeout_timer.stop()
            print("[TIMER] Stopped feeding timeout timer")

        if hasattr(self, 'max_feeding_timer') and self.max_feeding_timer.isActive():
            self.max_feeding_timer.stop()
            print("[TIMER] Stopped max feeding timer")

        if hasattr(self, 'during_feeding_timer') and self.during_feeding_timer.isActive():
            self.during_feeding_timer.stop()
            print("[TIMER] Stopped during feeding timer")

        # Update mode
        self.mode_start_times["post_feeding"] = current_time
        self.system_mode = "post_feeding"

        # Update UI
        self.status_label.setText(f"Starting {POST_FEEDING_DURATION // 60}-minute post-feeding analysis...")
        self.status_label.setStyleSheet("")
        self.feed_button.setEnabled(False)
        self.mode_label.setText("mode: POST-FEEDING ANALYSIS")
        self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: purple;")
        self.feed_button.setStyleSheet("background-color: #808080; color: white; font-size: 18px; font-weight: bold;")

        # Increment today's feeding count
        self.today_feeding_count += 1
        self.update_feeding_count_display()

        # Add this line to save the count:
        self.save_today_feeding_count()

        # Clear any existing post-feeding data to ensure we start fresh
        self.post_feeding_speeds = []
        self.post_feeding_variances = []
        self.post_feeding_data = []

        # IMPORTANT: Start a dedicated timer for post-feeding data collection
        self.post_feeding_timer = QTimer()
        self.post_feeding_timer.timeout.connect(self.collect_post_feeding_data)
        self.post_feeding_timer.start(1000)  # Collect data every second
        print("[TIMER] Started post-feeding data collection timer")

        # CRITICAL: Multi-layered approach to ensure completion
        # 1. Use QTimer.singleShot for the first approach
        print(f"[TIMER-1] Setting up primary completion timer for {POST_FEEDING_DURATION} seconds")
        QTimer.singleShot(POST_FEEDING_DURATION * 1000, self.force_complete_feeding_cycle)

        # 2. Set up a QTimer object as backup
        if hasattr(self, 'post_completion_timer') and self.post_completion_timer.isActive():
            self.post_completion_timer.stop()

        self.post_completion_timer = QTimer()
        self.post_completion_timer.setSingleShot(True)
        self.post_completion_timer.timeout.connect(self.force_complete_feeding_cycle)
        print(f"[TIMER-2] Setting up backup completion timer for {POST_FEEDING_DURATION + 5} seconds")
        self.post_completion_timer.start((POST_FEEDING_DURATION + 5) * 1000)  # 5 second buffer

        # 3. Add a safety method to the regular update_display method
        # This flag will trigger checking in update_display
        self.post_feeding_end_time = current_time + datetime.timedelta(seconds=POST_FEEDING_DURATION)
        self.post_feeding_completion_triggered = False

        expected_end_time = current_time + datetime.timedelta(seconds=POST_FEEDING_DURATION)
        print(f"[TIMER] Expected completion at {expected_end_time}")

        # Log detailed information for debugging
        print(f"[POST] Post-feeding mode started at {current_time}")
        print(f"[POST] Current feeding mode: {self.system_mode}")

    def force_complete_feeding_cycle(self):
        """Forcefully complete the feeding cycle regardless of current state"""
        current_time = datetime.datetime.now()
        print(f"[FORCE-COMPLETE] force_complete_feeding_cycle called at {current_time}")

        # Only take action if we're still in post-feeding mode
        if self.system_mode == "post_feeding":
            print("[FORCE-COMPLETE] Still in post-feeding mode, forcing completion")

            # Check how long we've been in post-feeding
            if self.mode_start_times.get("post_feeding"):
                elapsed = (current_time - self.mode_start_times["post_feeding"]).total_seconds()
                print(f"[FORCE-COMPLETE] Post-feeding duration: {elapsed:.1f} seconds")

            # Call the regular completion method
            self.complete_feeding_cycle()
        else:
            print(f"[FORCE-COMPLETE] Not in post-feeding mode (current: {self.system_mode}), no action needed")

    def check_post_feeding_timeout(self):
        """Safety check to ensure post-feeding mode doesn't run too long"""
        if self.system_mode == "post_feeding":
            print("[SAFETY] Post-feeding safety timeout triggered. Forcing completion.")
            mode_start = self.mode_start_times.get("post_feeding")
            if mode_start:
                current_time = datetime.datetime.now()
                duration = (current_time - mode_start).total_seconds()
                print(f"[SAFETY] Post-feeding ran for {duration:.1f} seconds before safety timeout")

            # Force completion
            self.complete_feeding_cycle()

    def complete_feeding_cycle(self):
        """Complete feeding cycle and update Prophet model with new data"""
        current_time = datetime.datetime.now()

        if self.system_mode != "post_feeding":
            print(f"[WARNING] complete_feeding_cycle called while in {self.system_mode} mode - ignoring")
            return

        print(f"[DEBUG] Completing feeding cycle at {current_time}")

        try:
            # Stop all active timers
            timers_to_check = [
                'pre_feeding_timer', 'during_feeding_timer', 'post_feeding_timer',
                'dosage_timer', 'post_completion_timer', 'feeding_timeout_timer',
                'max_feeding_timer'
            ]
            for timer_name in timers_to_check:
                if hasattr(self, timer_name):
                    timer = getattr(self, timer_name)
                    if hasattr(timer, 'isActive') and timer.isActive():
                        timer.stop()

            # Add feeding data to Prophet model
            if self.feed_start_time:
                print(f"Completing feeding cycle with {self.dosage_count} dosages")
                print(f"Pre-feeding data: {len(self.pre_feeding_speeds)} points")
                print(f"During-feeding data: {len(self.during_feeding_speeds)} points")
                print(f"Post-feeding data: {len(self.post_feeding_speeds)} points")

                # Ensure we have post-feeding data
                if len(self.post_feeding_speeds) < 10:
                    print("[WARNING] Very little post-feeding data, adding buffer data")
                    for _ in range(10 - len(self.post_feeding_speeds)):
                        if self.speed_buffer and self.variance_buffer:
                            self.post_feeding_speeds.append(self.speed_buffer[-1])
                            self.post_feeding_variances.append(self.variance_buffer[-1])

                # Update Prophet model with new feeding data
                self.feeding_model.add_feeding_data(
                    self.feed_start_time,
                    self.pre_feeding_speeds,
                    self.pre_feeding_variances,
                    self.during_feeding_speeds,
                    self.during_feeding_variances,
                    self.post_feeding_speeds,
                    self.post_feeding_variances,
                    dosage_count=self.dosage_count,
                    manual=False
                )

            self.system_mode = "cooldown"
            self.mode_start_times["cooldown"] = current_time

            self.pre_feeding_speeds = []
            self.pre_feeding_variances = []
            self.during_feeding_speeds = []
            self.during_feeding_variances = []
            self.post_feeding_speeds = []
            self.post_feeding_variances = []

            # Reset attribute to allow future timers
            if hasattr(self, 'post_feeding_timer_set'):
                delattr(self, 'post_feeding_timer_set')

            # Activate cooldown
            self.cooldown_active = True
            self.cooldown_end_time = current_time + datetime.timedelta(seconds=MIN_FEED_INTERVAL)

            # Reset UI state
            self.feed_button.setEnabled(False)
            self.status_label.setText(
                f"Feeding complete with {self.dosage_count} dosages. Cooldown active for 2 hours.")
            self.mode_label.setText("mode: COOLDOWN")
            self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")

            # Update Prophet forecast display with new data
            QTimer.singleShot(2000, self.update_prophet_forecast)

        except Exception as e:
            print(f"[ERROR] Error in complete_feeding_cycle: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to safe state
            self.reset_feeding_state()

    def collect_post_feeding_data(self):
        """Collect data during post-feeding mode"""
        timestamp = datetime.datetime.now()
        # Get current speed and variance from the most recent data
        speed = self.speed_buffer[-1] if self.speed_buffer else 0
        variance = self.variance_buffer[-1] if self.variance_buffer else 0

        # Store the data
        data_point = {
            "timestamp": timestamp,
            "speed": speed,
            "variance": variance
        }

        # Explicitly add to post-feeding data array
        self.post_feeding_data.append(data_point)

        # Add to the main arrays
        if speed not in self.post_feeding_speeds:
            self.post_feeding_speeds.append(speed)
        if variance not in self.post_feeding_variances:
            self.post_feeding_variances.append(variance)

        # Tag this as important data to keep
        self.video_thread.add_data_point(timestamp, speed, variance, "post_feeding")

        # Log data collection periodically
        if len(self.post_feeding_speeds) % 10 == 0:
            print(f"[POST] Collected {len(self.post_feeding_speeds)} post-feeding data points")

    def check_feeding_decision(self):
        """Periodically check if feeding should be suggested based on Prophet model"""
        current_time = datetime.datetime.now()

        # Force cooldown mode if cooldown is active
        if self.cooldown_active and self.cooldown_end_time and current_time < self.cooldown_end_time:
            if self.system_mode != "cooldown":
                print(f"[DECISION] Enforcing cooldown mode - was {self.system_mode}")
                self.system_mode = "cooldown"
                self.mode_label.setText("Mode: COOLDOWN")
                self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
                remaining_minutes = int((self.cooldown_end_time - current_time).total_seconds() / 60)
                self.status_label.setText(f"Cooldown active. Next feeding available in {remaining_minutes} minutes")

        # Check operating hours transitions
        currently_in_hours = self.is_within_operating_hours(current_time)

        # Update last_feed_check time
        self.last_feed_check = current_time

        # Handle operating hours transitions
        if self.was_outside_operating_hours and currently_in_hours:
            print(f"Entered operating hours at {current_time.strftime('%H:%M')}. Using rolling window data.")
            self.status_label.setText(f"Starting daily monitoring at {current_time.strftime('%H:%M')}")
            self.last_operating_hours_state_change = current_time
        elif not self.was_outside_operating_hours and not currently_in_hours:
            print(f"Exited operating hours at {current_time.strftime('%H:%M')}. Switching to passive monitoring.")
            if self.system_mode != "monitoring":
                print("Completing current feeding cycle before switching to passive monitoring")
                if self.system_mode == "feeding" and self.dosage_count > 0:
                    self.end_active_feeding()
                else:
                    self.reset_feeding_state()
            self.status_label.setText(f"Outside operating hours. Passive monitoring until {DAILY_OPERATION_START}.")
            self.last_operating_hours_state_change = current_time

        # Update the tracking variable
        self.was_outside_operating_hours = not currently_in_hours

        # Skip if not within operating hours
        if not currently_in_hours:
            if not hasattr(self, 'last_outside_hours_message') or time.time() - self.last_outside_hours_message > 300:
                self.status_label.setText(
                    f"Outside operating hours ({DAILY_OPERATION_START}-{DAILY_OPERATION_END}). Monitoring only.")
                self.last_outside_hours_message = time.time()
            return

        # Skip if feeding is already active or in cooldown mode
        if self.system_mode != "monitoring" and self.system_mode != "cooldown":
            return

        # Handle cooldown period explicitly
        if self.cooldown_active:
            if current_time < self.cooldown_end_time:
                remaining_minutes = (self.cooldown_end_time - current_time).total_seconds() / 60
                self.status_label.setText(
                    f"Cooldown active. Next feeding available in {int(remaining_minutes)} minutes")
                # Important: Make sure UI shows cooldown mode
                if self.system_mode != "cooldown":
                    self.system_mode = "cooldown"
                    self.mode_label.setText("Mode: COOLDOWN")
                    self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
                return
            else:
                # Only here do we transition from cooldown to monitoring
                print(f"[COOLDOWN] Cooldown period has ended at {current_time}")
                self.cooldown_active = False
                self.system_mode = "monitoring"
                self.mode_start_times["monitoring"] = current_time
                self.mode_label.setText("Mode: MONITORING")
                self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
                self.status_label.setText("Cooldown period complete. Resuming normal monitoring.")

        # Now, only if we're actually in monitoring mode (not cooldown) and cooldown isn't active,
        # check if feeding should be suggested
        if self.system_mode == "monitoring" and not self.cooldown_active:
            # Check if feeding should be recommended using Prophet model
            if self.feeding_model.should_feed(current_time):
                # If Prophet recommends feeding, show indicator in UI
                self.status_label.setText("FEEDING RECOMMENDED - Prophet model predicts fish hunger")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")

                # Begin the feeding cycle
                self.start_feeding_cycle(current_time)
            else:
                # Update next feeding prediction if no immediate feeding
                self.update_next_feeding_prediction()

            # If a feeding wasn't started, make sure the next check time is displayed correctly
            next_check_in = FEED_DECISION_INTERVAL  # Reset to full interval
            next_min = int(next_check_in // 60)
            next_sec = int(next_check_in % 60)
            self.time_label.setText(f"Next feeding check: {next_min:02d}:{next_sec:02d}")

    def first_feeding_check(self):
        """Initial feeding check after system startup"""
        self.status_label.setText("Initial feeding check in progress...")
        current_time = datetime.datetime.now()

        if self.is_within_operating_hours(current_time):
            # Instead of immediate feeding, start 5-minute analysis window
            self.status_label.setText("Starting initial behavior analysis using rolling window...")
            self.status_label.setStyleSheet("")

            # NOTE: No need to reset window with rolling approach
            # Just continue collecting data in the rolling window

            # Schedule actual feeding recommendation after enough data is collected
            QTimer.singleShot(MONITORING_WINDOW * 1000, self.initial_window_completed)

    def initial_window_completed(self):
        """Called when the initial 5-minute analysis window is complete"""
        current_time = datetime.datetime.now()
        self.initialisation_complete = True

        # Check for cooldown first
        if self.cooldown_active or self.system_mode == "cooldown":
            remaining_minutes = int((self.cooldown_end_time - current_time).total_seconds() / 60)
            self.status_label.setText(f"Cooldown active. Next feeding available in {remaining_minutes} minutes")
            return

        # Log the automated decision
        print("Automatically starting initial feeding cycle after data collection")

        # Status update for UI
        self.status_label.setText("Automated feeding cycle starting...")

        # Directly start the feeding cycle instead of waiting for button press
        self.start_feeding_cycle(current_time)

    def is_within_operating_hours(self, current_time):
        """Check if the current time is within operating hours"""
        start_time = datetime.datetime.strptime(DAILY_OPERATION_START, "%H:%M").time()
        end_time = datetime.datetime.strptime(DAILY_OPERATION_END, "%H:%M").time()

        return start_time <= current_time.time() <= end_time

    def closeEvent(self, event):
        """Handle application close event"""
        # Check if video_thread exists before stopping it
        if hasattr(self, 'video_thread') and self.video_thread is not None:
            try:
                self.video_thread.stop()
                print("Video thread stopped successfully")
            except Exception as e:
                print(f"Error stopping video thread: {e}")
        else:
            print("No video thread to stop")

        event.accept()

    def initialize_display_buffers(self):
        """Initialize buffers for 1-second averaging of display values"""
        self.speed_buffer = deque(maxlen=30)  # Assuming 30fps
        self.variance_buffer = deque(maxlen=30)
        self.last_display_update = time.time()
        self.display_update_interval = 1.0  # Update every second

    def check_feeding_safety(self):
        """Persistent safety check that runs periodically to detect and fix stuck feeding modes"""
        current_time = datetime.datetime.now()

        # Skip if we're in monitoring mode and not in cooldown
        if self.system_mode == "monitoring" and not self.cooldown_active:
            return

        if self.cooldown_active:
            # If cooldown is active but the mode isn't cooldown, fix it
            if self.system_mode != "cooldown":
                print(f"[SAFETY] Cooldown active but mode is {self.system_mode} - correcting to cooldown mode")
                self.system_mode = "cooldown"
                self.mode_start_times["cooldown"] = self.mode_start_times.get("monitoring", current_time)
                self.mode_label.setText("Mode: COOLDOWN")
                self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")

            # Check if cooldown period has ended
            if hasattr(self, 'cooldown_end_time') and current_time >= self.cooldown_end_time:
                print(f"[SAFETY] Cooldown period complete - transitioning to monitoring")
                self.system_mode = "monitoring"
                self.mode_start_times["monitoring"] = current_time
                self.cooldown_active = False
                self.status_label.setText("Cooldown period complete. Resuming normal monitoring.")
                return

        # Get the start time for the current mode
        mode_start = self.mode_start_times.get(self.system_mode)
        if not mode_start:
            # No start time recorded, set it now
            self.mode_start_times[self.system_mode] = current_time
            return

        # Calculate how long we've been in this mode
        mode_duration = (current_time - mode_start).total_seconds()

        # IMPORTANT: Log all mode durations for debugging
        print(
            f"[SAFETY] Current mode: {self.system_mode}, Duration: {mode_duration:.1f}s, Dosage count: {self.dosage_count}")

        # Check for stuck modes based on specific conditions
        if self.system_mode == "feeding" and self.dosage_count == 0 and mode_duration > MAX_DURING_mode_NO_DOSAGE:
            # Double enforcement - In "feeding" mode with no dosage for too long - record as missed feeding
            print(f"[SAFETY] 'feeding' mode with no dosage for {mode_duration:.1f}s - EMERGENCY FIXING MISSED FEEDING")

            # Force a call to check_feeding_timeout which will handle everything
            self.check_feeding_timeout()

        elif self.system_mode == "feeding" and mode_duration > MAX_FEEDING_DURATION:
            # Enforce maximum feeding duration
            print(
                f"[SAFETY] 'feeding' mode active for {mode_duration:.1f}s - exceeds MAX_FEEDING_DURATION ({MAX_FEEDING_DURATION}s)")

            if self.dosage_count > 0:
                print("[SAFETY] Ending active feeding with existing dosages")
                self.end_active_feeding()
            else:
                print("[SAFETY] No dosages - treating as missed feeding")
                self.check_feeding_timeout()

        elif self.system_mode == "post_feeding" and mode_duration > POST_FEEDING_DURATION + 60:
            # Post-feeding mode is stuck for a minute beyond expected duration
            print(f"[SAFETY] Post-feeding mode stuck for {mode_duration:.1f}s - forcing completion")
            self.complete_feeding_cycle()

        elif mode_duration > MAX_FEEDING_mode_DURATION:
            # Any mode lasting too long gets reset
            print(f"[SAFETY] '{self.system_mode}' mode stuck for {mode_duration:.1f}s - emergency reset")

            if self.system_mode == "feeding" and self.dosage_count > 0:
                # If we're in "feeding" mode with some dosages, complete the feeding cycle
                self.end_active_feeding()
            elif self.system_mode == "post_feeding":
                # For post-feeding mode that's running too long, complete the cycle properly
                print("[SAFETY] Post-feeding mode running too long - completing cycle properly")
                self.complete_feeding_cycle()
            else:
                # For any other stuck mode, reset completely with no data recording
                self.reset_feeding_state()
                self.status_label.setText(f"System reset - '{self.system_mode}' mode was active too long.")

    def reset_feeding_state(self):
        """Reset all feeding state variables and clear data buffers"""
        # Reset mode
        self.system_mode = "monitoring"
        self.mode_start_times["monitoring"] = datetime.datetime.now()

        # Reset feeding data to prevent memory buildup
        self.pre_feeding_speeds = []
        self.pre_feeding_variances = []
        self.during_feeding_speeds = []
        self.during_feeding_variances = []
        self.post_feeding_speeds = []
        self.post_feeding_variances = []

        # Reset feeding response data
        self.feeding_response_history.clear()
        self.feeding_baseline_established = False
        self.baseline_speeds = []
        self.baseline_variances = []

        # Stop any active timers
        if hasattr(self, 'pre_feeding_timer') and self.pre_feeding_timer.isActive():
            self.pre_feeding_timer.stop()
        if hasattr(self, 'during_feeding_timer') and self.during_feeding_timer.isActive():
            self.during_feeding_timer.stop()
        if hasattr(self, 'post_feeding_timer') and self.post_feeding_timer.isActive():
            self.post_feeding_timer.stop()
        if hasattr(self, 'dosage_timer') and self.dosage_timer.isActive():
            self.dosage_timer.stop()

        # Remove the flag for post-feeding timer if it exists
        if hasattr(self, 'post_feeding_timer_set'):
            delattr(self, 'post_feeding_timer_set')

        # Reset UI
        self.feed_button.setEnabled(False)
        self.feed_button.setStyleSheet("background-color: #808080; color: white; font-size: 18px; font-weight: bold;")
        self.mode_label.setText("mode: MONITORING")
        self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        self.dosage_count = 0
        self.dosage_label.setText("Dosage: 0")

    def mark_initialisation_complete(self):
        """Mark initialisation complete and check if initial feeding is needed"""
        self.initialisation_complete = True
        current_time = datetime.datetime.now()

        # Standardize on lowercase for consistency
        self.system_mode = "monitoring"
        self.mode_start_times["monitoring"] = current_time  # CHANGED FROM phase_start_times

        print("initialisation complete - checking feeding history...")

        # Check if there's a recent feeding that should trigger cooldown
        recent_feeding = False
        if hasattr(self.feeding_model, 'feeding_history') and self.feeding_model.get_feeding_history():
            last_feed = self.feeding_model.get_feeding_history()[-1]['timestamp']
            time_since_last_feed = (current_time - last_feed).total_seconds()

            if time_since_last_feed < MIN_FEED_INTERVAL:
                # Recent feeding detected - should be in cooldown
                print(f"Recent feeding detected {time_since_last_feed / 60:.1f} minutes ago - activating cooldown")
                self.system_mode = "cooldown"
                self.mode_start_times["cooldown"] = current_time
                self.cooldown_end_time = last_feed + datetime.timedelta(seconds=MIN_FEED_INTERVAL)

                # Update display
                self.update_status_display()
                return

        # Single check for feeding history
        has_history = False
        if hasattr(self.feeding_model, 'feeding_history'):
            has_history = len(self.feeding_model.get_feeding_history()) > 0
            print(f"Feeding history found: {has_history} ({len(self.feeding_model.get_feeding_history())} records)")
        else:
            print("No feeding_history attribute found in model")

        # Single decision point for what to do next
        if not has_history:
            print("NO FEEDING HISTORY DETECTED - FORCING FIRST FEEDING CYCLE")
            # Use a short delay to allow UI to update first
            QTimer.singleShot(2000, lambda: self.start_feeding_cycle(current_time))
        else:
            # Continue with normal operation using Prophet
            print(f"Using existing feeding history with {len(self.feeding_model.get_feeding_history())} records")

            if self.feeding_model.should_feed(current_time):
                self.status_label.setText("FEEDING RECOMMENDED")
                QTimer.singleShot(1000, lambda: self.start_feeding_cycle(current_time))
            else:
                self.status_label.setText("Monitoring fish behavior. Using Prophet prediction model.")

    def fix_button_connections(self):
        """Fix button connections based on current mode"""
        try:
            # Disconnect any existing connections to prevent duplicates
            self.feed_button.clicked.disconnect()
        except TypeError:
            # No connections to disconnect
            pass

        # Set up proper connection based on mode
        if self.system_mode == "monitoring":
            # In monitoring mode, button should CONFIRM FEED cycle
            self.feed_button.clicked.connect(lambda: self.start_feeding_cycle(datetime.datetime.now()))
            print("Feed button connected to start_feeding_cycle")
        elif self.system_mode == "feeding":
            # In during mode, button should confirm dosage
            self.feed_button.clicked.connect(self.confirm_dosage)
            print("Feed button connected to confirm_dosage")
        else:
            # For other modes, button should be disabled
            self.feed_button.setEnabled(False)

    def start_feeding_cycle(self, current_time):
        """Begin a feeding cycle - pre_feeding, feeding, post_feeding modes"""

        # Guard against invalid state transitions
        if self.system_mode != "monitoring" and not hasattr(self, 'force_override'):
            print(f"WARNING: Attempted to CONFIRM FEED cycle while in {self.system_mode} mode - ignoring")
            return

        print(f"Starting feeding cycle at {current_time} - beginning with pre-feeding mode")

        # Set feeding state
        self.system_mode = "pre_feeding"
        self.feed_start_time = current_time
        self.mode_start_times["pre_feeding"] = current_time  # Track mode start time

        # Reset counters and history
        self.dosage_count = 0
        self.pre_feeding_speeds = []
        self.pre_feeding_variances = []
        self.during_feeding_speeds = []
        self.during_feeding_variances = []
        self.post_feeding_speeds = []
        self.post_feeding_variances = []

        # Important: Make sure feed button is disabled during pre-feeding
        self.feed_button.setEnabled(False)
        self.feed_button.setText("WAIT: PRE-FEEDING ANALYSIS")
        self.feed_button.setStyleSheet("background-color: #808080; color: white; font-size: 18px; font-weight: bold;")

        # Update UI
        self.mode_label.setText("mode: PRE-FEEDING ANALYSIS")
        self.mode_label.setStyleSheet("font-size: 16px; font-weight: bold; color: orange;")
        self.status_label.setText("Starting 5-minute pre-feeding analysis...")
        self.status_label.setStyleSheet("")

        # Make sure we're collecting pre-feeding data
        if hasattr(self, 'pre_feeding_timer') and self.pre_feeding_timer.isActive():
            self.pre_feeding_timer.stop()

        self.pre_feeding_timer = QTimer()
        self.pre_feeding_timer.timeout.connect(self.collect_pre_feeding_data)
        self.pre_feeding_timer.start(1000)  # Collect once per second

        # IMPORTANT: Schedule the transition to active feeding mode
        QTimer.singleShot(PRE_FEEDING_DURATION * 1000, self.start_active_feeding)

        print(f"Pre-feeding analysis will run for {PRE_FEEDING_DURATION} seconds")

    def setup_prophet_ui(self):
        """Add Prophet-specific UI components to the available space in the control panel"""
        # Create a compact Prophet forecast section
        self.forecast_box = QGroupBox("Feeding Forecast")
        forecast_layout = QVBoxLayout()
        forecast_layout.setContentsMargins(5, 5, 5, 5)  # Make margins smaller to save space
        forecast_layout.setSpacing(5)  # Reduce spacing between elements
        self.forecast_box.setLayout(forecast_layout)

        # Add next feeding prediction display - keep it very compact
        self.next_feed_label = QLabel("Next feeding: --:--")
        self.next_feed_label.setStyleSheet("font-weight: bold;")
        forecast_layout.addWidget(self.next_feed_label)

        # Add a small forecast graph
        self.forecast_canvas_widget = QWidget()
        self.forecast_canvas_widget.setMinimumHeight(150)  # Small but visible
        self.forecast_canvas_widget.setMaximumHeight(180)  # Limit maximum height
        forecast_canvas_layout = QVBoxLayout()
        forecast_canvas_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        self.forecast_canvas_widget.setLayout(forecast_canvas_layout)
        self.forecast_canvas = None  # Will be set later
        forecast_layout.addWidget(self.forecast_canvas_widget)

        # Find the control panel to insert our forecast widget at the right position
        control_panel = None
        main_layout = self.centralWidget().layout()

        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.widget() and isinstance(item.widget(), QGroupBox) and item.widget().title() == "Controls":
                control_panel = item.widget()
                break

        if control_panel and control_panel.layout():
            # Get the control panel's layout
            control_layout = control_panel.layout()

            # We need to insert our forecast between the feed button and the status label
            # First, find the feed button and status label positions
            feed_button_pos = -1
            status_label_pos = -1

            for i in range(control_layout.count()):
                widget = control_layout.itemAt(i).widget()
                if widget == self.feed_button:
                    feed_button_pos = i
                elif widget == self.status_label:
                    status_label_pos = i

            # If we found both positions, insert the forecast box between them
            if feed_button_pos >= 0 and status_label_pos > feed_button_pos:
                # Create a temporary layout to preserve the existing widgets
                temp_widgets = []
                for i in range(control_layout.count()):
                    item = control_layout.itemAt(0)  # Always remove the first item
                    widget = item.widget()
                    if widget:
                        control_layout.removeItem(item)
                        temp_widgets.append(widget)
                        widget.setParent(None)  # Remove parent relationship

                # Now reinsert widgets with our forecast box in the right position
                for i, widget in enumerate(temp_widgets):
                    if i == feed_button_pos + 1:  # Insert after feed button
                        control_layout.addWidget(self.forecast_box)
                    control_layout.addWidget(widget)
            else:
                # Fallback: insert after the feed button
                control_layout.insertWidget(feed_button_pos + 1 if feed_button_pos >= 0 else 0,
                                            self.forecast_box)
        else:
            # If we can't find the control panel, just add before the feeding history
            # Find the feeding history widget
            history_box = None
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item.widget() and isinstance(item.widget(), QGroupBox) and "History" in item.widget().title():
                    history_box = item.widget()
                    break

            if history_box:
                # Get the position of the history box
                history_pos = main_layout.indexOf(history_box)
                row, column, rowspan, colspan = main_layout.getItemPosition(history_pos)

                # Add our forecast box above the history box
                main_layout.addWidget(self.forecast_box, row, column, 1, colspan)

                # Move the history box down
                main_layout.removeWidget(history_box)
                main_layout.addWidget(history_box, row + 1, column, rowspan, colspan)
            else:
                # Last resort: add to bottom right
                main_layout.addWidget(self.forecast_box, 1, 2, 1, 1)

        # Schedule first forecast update
        QTimer.singleShot(2000, self.update_prophet_forecast)

    def update_prophet_forecast(self):
        """Update the Prophet forecast display with a compact visualization"""
        if not hasattr(self.feeding_model, 'get_forecast_plot'):
            return

        try:
            # Get forecast data
            forecast_data = self.feeding_model.get_daily_forecast()
            if not forecast_data:
                self.next_feed_label.setText("Forecast unavailable - need more data")
                return

            # Create a compact figure
            fig = Figure(figsize=(4, 2.5), dpi=80)  # Smaller figure
            ax = fig.add_subplot(111)

            # Extract data for next 12 hours
            now = datetime.datetime.now()
            times = []
            hunger_scores = []
            is_feeding = []

            for pred in forecast_data:
                if pred['time'] >= now and (pred['time'] - now).total_seconds() <= 12 * 3600:
                    times.append(pred['time'])
                    hunger_scores.append(pred['hunger_score'])
                    is_feeding.append(pred['recommended'])

            # Plot hunger score line
            ax.plot(times, hunger_scores, 'b-', linewidth=1.5)

            # Add feeding threshold line
            ax.axhline(y=1.3, color='r', linestyle='--', alpha=0.7, linewidth=1)

            # Highlight recommended feeding times
            for i, (time, score, feed) in enumerate(zip(times, hunger_scores, is_feeding)):
                if feed:
                    ax.plot([time], [score], 'ro', markersize=5)  # Red dot for feeding times

            # Format x-axis to show hours
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            # Rotate labels for better readability
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(8)
                label.set_ha('right')

            # Y-axis labels
            ax.set_ylabel('Hunger', fontsize=9)
            ax.tick_params(axis='y', labelsize=8)

            # Remove title to save space
            ax.set_title('12-Hour Forecast', fontsize=10)

            # Tight layout
            fig.tight_layout()

            # Clear existing canvas
            if self.forecast_canvas is not None:
                self.forecast_canvas_widget.layout().removeWidget(self.forecast_canvas)
                self.forecast_canvas.deleteLater()

            # Create new canvas with the figure
            self.forecast_canvas = FigureCanvasQTAgg(fig)
            self.forecast_canvas_widget.layout().addWidget(self.forecast_canvas)

            # Update next feeding prediction
            next_feeding = self.get_next_recommended_feeding()
            if next_feeding:
                time_str = next_feeding['time'].strftime("%H:%M")
                confidence = int((1.0 - next_feeding['uncertainty'] / 3.0) * 100)  # Scale to percentage
                self.next_feed_label.setText(f"Next feeding: {time_str} ({confidence}% confidence)")
            else:
                self.next_feed_label.setText("No feeding needed in next 12 hours")

        except Exception as e:
            print(f"Error updating Prophet forecast: {e}")
            import traceback
            traceback.print_exc()

    def get_next_recommended_feeding(self):
        """Get the next recommended feeding time from Prophet model"""
        if not hasattr(self.feeding_model, 'get_daily_forecast') or not callable(
                getattr(self.feeding_model, 'get_daily_forecast')):
            return None

        # Get 24-hour forecast
        forecast = self.feeding_model.get_daily_forecast()
        if not forecast:
            return None

        # Get current time
        now = datetime.datetime.now()

        # Find the next recommended feeding time
        next_feedings = []
        for hour_data in forecast:
            # Only consider future times
            if hour_data['time'] <= now:
                continue

            # Check if it's a recommended feeding time
            if hour_data['hunger_score'] > 1.3:  # Threshold for feeding
                uncertainty = hour_data['upper_bound'] - hour_data['lower_bound']
                next_feedings.append({
                    'time': hour_data['time'],
                    'score': hour_data['hunger_score'],
                    'uncertainty': uncertainty
                })

        # Return the soonest recommended feeding with lowest uncertainty
        if next_feedings:
            # Sort by time
            next_feedings.sort(key=lambda x: x['time'])
            return next_feedings[0]

        return None

    def update_next_feeding_prediction(self):
        """Update the display of next predicted feeding time"""
        next_feeding = self.get_next_recommended_feeding()
        if next_feeding:
            # Get time until next feeding
            now = datetime.datetime.now()
            time_diff = (next_feeding['time'] - now).total_seconds()
            hours = int(time_diff // 3600)
            minutes = int((time_diff % 3600) // 60)

            # Update status display
            if time_diff < 3600:  # Less than 1 hour
                self.status_label.setText(
                    f"Prophet predicts feeding needed in {minutes} minutes. Continuing monitoring.")
            else:
                self.status_label.setText(
                    f"Prophet predicts feeding needed in {hours}h {minutes}m. Continuing monitoring.")
        else:
            self.status_label.setText("Monitoring fish behavior. No immediate feeding needed.")

    def create_prophet_dashboard(self):
        """Create a standalone dashboard window for Prophet model visualizations with improved layout"""
        try:
            # Create a new window
            self.prophet_dashboard = QMainWindow()
            self.prophet_dashboard.setWindowTitle("Fish Feeding Prophet Dashboard")
            self.prophet_dashboard.setGeometry(200, 200, 900, 650)  # Larger window size

            # Create central widget and layout
            central_widget = QWidget()
            self.prophet_dashboard.setCentralWidget(central_widget)
            main_layout = QVBoxLayout()
            central_widget.setLayout(main_layout)

            # Add header
            header_label = QLabel("Prophet Model Predictions")
            header_label.setStyleSheet("font-size: 18px; font-weight: bold;")
            main_layout.addWidget(header_label)

            # Add forecast graph - MAKE LARGER
            forecast_box = QGroupBox("24-Hour Hunger Forecast")
            forecast_layout = QVBoxLayout()

            # Create matplotlib canvas with larger figure
            fig = self.feeding_model.get_forecast_plot()
            if fig:
                # Adjust figure size
                fig.set_size_inches(10, 5)  # Larger figure
                canvas = FigureCanvasQTAgg(fig)
                canvas.setMinimumHeight(250)  # Force minimum height
                forecast_layout.addWidget(canvas)
            else:
                canvas_placeholder = QLabel("Forecast not available - need more feeding data")
                forecast_layout.addWidget(canvas_placeholder)

            forecast_box.setLayout(forecast_layout)
            main_layout.addWidget(forecast_box)

            # Add hourly predictions table with better spacing
            hourly_box = QGroupBox("Hourly Feeding Recommendations")
            hourly_layout = QVBoxLayout()

            hourly_table = QTableWidget()
            hourly_table.setColumnCount(4)
            hourly_table.setHorizontalHeaderLabels(["Time", "Hunger Score", "Uncertainty", "Recommendation"])

            # Set column widths explicitly
            hourly_table.setColumnWidth(0, 100)  # Time
            hourly_table.setColumnWidth(1, 120)  # Hunger Score
            hourly_table.setColumnWidth(2, 120)  # Uncertainty
            hourly_table.setColumnWidth(3, 150)  # Recommendation

            # Set minimum height to show more rows
            hourly_table.setMinimumHeight(200)

            # Get predictions for next 24 hours
            forecast_data = self.feeding_model.get_daily_forecast()
            if forecast_data:
                hourly_table.setRowCount(len(forecast_data))

                for i, hour_data in enumerate(forecast_data):
                    # Time
                    time_item = QTableWidgetItem(hour_data['time'].strftime("%H:%M"))
                    hourly_table.setItem(i, 0, time_item)

                    # Hunger score
                    score_item = QTableWidgetItem(f"{hour_data['hunger_score']:.2f}")
                    hourly_table.setItem(i, 1, score_item)

                    # Add debug print to see raw values
                    uncertainty = hour_data['upper_bound'] - hour_data['lower_bound']
                    print(
                        f"Hour: {hour_data['time'].hour}:00, Upper: {hour_data['upper_bound']:.6f}, Lower: {hour_data['lower_bound']:.6f}, Diff: {uncertainty:.6f}")
                    uncertainty_item = QTableWidgetItem(f"{uncertainty:.2f}")  # Show 4 decimal places instead of 2
                    hourly_table.setItem(i, 2, uncertainty_item)

                    # Recommendation
                    if hour_data['recommended']:
                        rec_item = QTableWidgetItem("FEED")
                        rec_item.setBackground(QColor(255, 200, 200))  # Light red
                    else:
                        rec_item = QTableWidgetItem("Wait")
                    hourly_table.setItem(i, 3, rec_item)
            else:
                hourly_table.setRowCount(1)
                hourly_table.setSpan(0, 0, 1, 4)
                hourly_table.setItem(0, 0, QTableWidgetItem("No forecast data available yet"))

            hourly_layout.addWidget(hourly_table)
            hourly_box.setLayout(hourly_layout)
            main_layout.addWidget(hourly_box)

            # Add feeding history impact - Improved formatting
            history_box = QGroupBox("Feeding History Impact")
            history_layout = QVBoxLayout()

            # Create text summary with better formatting
            if self.feeding_model.trained and self.feeding_model.prophet_model:
                history_text = QLabel(
                    f"<p><b>Prophet model trained with {len(self.feeding_model.get_feeding_history())} feeding events "
                    f"and {len(self.feeding_model.get_missed_feedings())} missed feedings.</b></p>"
                    f"<p>The model has identified daily patterns in fish hunger, with peak hunger typically "
                    f"occurring at specific times of day based on your feeding history.</p>"
                )
                history_text.setWordWrap(True)
            else:
                history_text = QLabel(
                    "<p>Not enough feeding history to analyze patterns yet.</p>"
                    "<p>Continue using the system to accumulate more data.</p>"
                )
                history_text.setWordWrap(True)

            history_layout.addWidget(history_text)
            history_box.setLayout(history_layout)
            main_layout.addWidget(history_box)

            # Add explanation with better formatting
            explanation = QLabel(
                "<p>The Prophet model analyzes your feeding history to predict when fish are likely to be hungry.<br>"
                "Green areas show predicted times of satiation, while red areas indicate likely hunger periods.<br>"
                "The system will automatically recommend feedings based on these predictions and current behavior.</p>"
            )
            explanation.setWordWrap(True)
            explanation.setStyleSheet("font-style: italic;")
            main_layout.addWidget(explanation)

            # Add refresh button
            refresh_button = QPushButton("Refresh Data")
            refresh_button.clicked.connect(lambda: self.update_prophet_dashboard())
            refresh_button.setMinimumHeight(30)  # Taller button
            main_layout.addWidget(refresh_button)

            # Show the dashboard
            self.prophet_dashboard.show()

        except Exception as e:
            print(f"Error creating Prophet dashboard: {e}")
            import traceback
            traceback.print_exc()

    def update_prophet_dashboard(self):
        """Update the Prophet dashboard with latest data"""
        if not hasattr(self, 'prophet_dashboard'):
            return

        try:
            # For demonstration, just recreate the dashboard
            self.prophet_dashboard.close()
            self.create_prophet_dashboard()
        except Exception as e:
            print(f"Error updating Prophet dashboard: {e}")

    def monitor_resources(self):
        """Monitor system resources and take action if necessary"""
        try:
            import psutil
            import gc

            # Get memory usage
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / (1024 * 1024)

            # Log memory usage every 10 minutes
            current_time = datetime.datetime.now()
            if not hasattr(self, 'last_memory_log') or (current_time - self.last_memory_log).total_seconds() > 600:
                self.last_memory_log = current_time
                print(f"[MEMORY] Current usage: {memory_usage_mb:.1f} MB at {current_time}")

            # Take action if memory usage is too high
            if memory_usage_mb > 500:  # 500MB threshold
                print(f"[WARNING] High memory usage detected: {memory_usage_mb:.1f} MB")
                # Force garbage collection
                gc.collect()

                # Clear matplotlib cache
                plt.close('all')

            return memory_usage_mb
        except ImportError:
            print("psutil package not available for memory monitoring")
            return 0
        except Exception as e:
            print(f"Error in resource monitoring: {e}")
            return 0

    def setup_resource_monitoring(self):
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self.monitor_resources)
        self.resource_timer.start(60000)  # Check every minute

    def restart_critical_components(self):
        """Restart critical components to prevent resource bloat"""
        print(f"[MAINTENANCE] Performing scheduled component restart at {datetime.datetime.now()}")

        # Only restart if in monitoring mode to avoid disrupting feeding
        if self.system_mode != "monitoring":
            print("[MAINTENANCE] Skipping restart - system not in monitoring mode")
            return

        try:
            # 1. Stop and restart display timer
            if hasattr(self, 'display_timer') and self.display_timer.isActive():
                self.display_timer.stop()
                self.display_timer.start(1000)
                print("[MAINTENANCE] Display timer restarted")

            # 2. Instead of clearing matplotlib figures completely, just refresh them with existing data
            # Get the data
            timestamps, speeds, variances = self.video_thread.get_speed_data(window_seconds=300)  # Last 5 minutes
            speed_range, variance_range = self.feeding_model.get_satiated_ranges()

            # Only refresh if we have enough data
            if len(timestamps) >= 2 and len(speeds) >= 2 and len(variances) >= 2:
                # Update graphs with current data
                self.speed_graph.update_plot(
                    timestamps.copy() if isinstance(timestamps, list) else list(timestamps),
                    speeds.copy() if isinstance(speeds, list) else list(speeds),
                    speed_range
                )

                self.variance_graph.update_plot(
                    timestamps.copy() if isinstance(timestamps, list) else list(timestamps),
                    variances.copy() if isinstance(variances, list) else list(variances),
                    variance_range
                )

                # Synchronize graph time windows
                self.synchronize_graph_windows()

                print("[MAINTENANCE] Graphs refreshed with existing data")
            else:
                print("[MAINTENANCE] Not enough data to refresh graphs")

            # 3. Force save feeding model to ensure data persistence
            self.feeding_model.save_model()
            print("[MAINTENANCE] Feeding model saved")

            # 4. Force garbage collection
            import gc
            gc.collect()
            print("[MAINTENANCE] Garbage collection performed")

            # Schedule next restart in 4 hours
            QTimer.singleShot(4 * 60 * 60 * 1000, self.restart_critical_components)
            print("[MAINTENANCE] Next component restart scheduled in 4 hours")

        except Exception as e:
            print(f"[ERROR] Component restart failed: {e}")
            traceback.print_exc()

    def update_display(self):
        """Update the UI displays with optimized data handling"""
        # Add post-feeding completion check
        current_time = datetime.datetime.now()

        # IMPORTANT: Add this to prevent too frequent updates
        if hasattr(self, 'last_display_update_time'):
            if time.time() - self.last_display_update_time < 3.0:  # Only update every 3 seconds
                self.updating_graphs = False  # Clear flag immediately
                return
        self.last_display_update_time = time.time()

        self.updating_graphs = True  # Set flag to indicate graph updates are happening

        try:
            # For live graphs, get data from video thread with time window limitation
            timestamps, speeds, variances = self.video_thread.get_speed_data(window_seconds=300)  # Last 5 minutes

            # Get satiated ranges from model
            speed_range, variance_range = self.feeding_model.get_satiated_ranges()

            # Only update graphs if we have enough data
            if len(timestamps) >= 2 and len(speeds) >= 2 and len(variances) >= 2:
                # Update graphs with deep copies to avoid modifying original data
                self.speed_graph.update_plot(
                    timestamps.copy() if isinstance(timestamps, list) else list(timestamps),
                    speeds.copy() if isinstance(speeds, list) else list(speeds),
                    speed_range
                )

                self.variance_graph.update_plot(
                    timestamps.copy() if isinstance(timestamps, list) else list(timestamps),
                    variances.copy() if isinstance(variances, list) else list(variances),
                    variance_range
                )

                # Synchronize graph time windows
                self.synchronize_graph_windows()

            # Update feeding history
            all_feedings = self.feeding_model.get_feeding_history().copy()
            missed_feedings = self.feeding_model.get_missed_feedings()
            if missed_feedings:
                # Add missed feedings with appropriate formatting
                for missed in missed_feedings:
                    feed_record = {
                        'timestamp': missed['timestamp'],
                        'features': missed['features'],
                        'dosage_count': 0,
                        'missed': True
                    }
                    all_feedings.append(feed_record)

                # Sort by timestamp (most recent first)
                all_feedings.sort(key=lambda x: x['timestamp'], reverse=True)

                # Update the feeding history table with all feedings
                self.feeding_history_table.update_history(all_feedings)
            else:
                # Just update with regular feeding history
                self.feeding_history_table.update_history(self.feeding_model.get_feeding_history())

            # Check if new day started
            self.check_day_change()

        except Exception as e:
            print(f"Error in update_display: {e}")
            traceback.print_exc()

        finally:
            # Clear flag when done - VERY IMPORTANT
            self.updating_graphs = False

    def excepthook(self, exc_type, exc_value, exc_tb):
        """Global exception handler for PyQt slots"""
        try:
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print(f"UNHANDLED EXCEPTION:\n{tb}")
            # Continue normal exception processing
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception as e:
            print(f"Error in excepthook: {e}")
            # Basic fallback if traceback formatting fails
            print(f"Original exception: {exc_type.__name__}: {exc_value}")

    def initialize_mode_tracking(self):
        """Initialize mode tracking variables"""
        current_time = datetime.datetime.now()
        self.mode_start_times = {
            "initialising": current_time,
            "monitoring": None,
            "pre_feeding": None,
            "feeding": None,
            "post_feeding": None,
            "cooldown": None
        }
        print("Mode tracking initialized")

    def synchronize_graph_windows(self):
        """Ensure both graphs show the same time window"""
        try:
            # Skip if either graph doesn't have data yet
            if not hasattr(self.speed_graph, 'has_data') or not self.speed_graph.has_data or \
                    not hasattr(self.variance_graph, 'has_data') or not self.variance_graph.has_data:
                return

            # Get current x ranges
            speed_plot = self.speed_graph.plot_widget
            variance_plot = self.variance_graph.plot_widget

            speed_range = speed_plot.getViewBox().viewRange()[0]
            variance_range = variance_plot.getViewBox().viewRange()[0]

            # Calculate the union of both time ranges
            min_x = min(speed_range[0], variance_range[0])
            max_x = max(speed_range[1], variance_range[1])

            # Set the same range for both graphs
            speed_plot.setXRange(min_x, max_x, padding=0)
            variance_plot.setXRange(min_x, max_x, padding=0)

            if DEBUG_MODE:
                print(f"Synchronized graph windows to x range: {min_x:.2f}-{max_x:.2f}")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Error synchronizing graph windows: {e}")

    def load_today_feeding_count(self):
        """Load today's feeding count from disk"""
        try:
            count_file = os.path.join(DATA_FOLDER, "daily_feeding_count.pkl")
            if os.path.exists(count_file):
                with open(count_file, 'rb') as f:
                    data = pickle.load(f)
                    saved_date = data.get('date')
                    saved_count = data.get('count', 0)

                    # Check if the saved date is today
                    current_date = datetime.datetime.now().date()
                    if saved_date == current_date:
                        self.today_feeding_count = saved_count
                        print(f"Loaded today's feeding count: {saved_count}")
                    else:
                        print(f"Saved feeding count is from {saved_date}, resetting for today ({current_date})")
                        self.today_feeding_count = 0
        except Exception as e:
            print(f"Error loading daily feeding count: {e}")
            self.today_feeding_count = 0

        # Update the display
        self.update_feeding_count_display()

    def save_today_feeding_count(self):
        """Save today's feeding count to disk"""
        try:
            count_file = os.path.join(DATA_FOLDER, "daily_feeding_count.pkl")
            data = {
                'date': self.today_date,
                'count': self.today_feeding_count
            }

            with open(count_file, 'wb') as f:
                pickle.dump(data, f)

            print(f"Saved today's feeding count: {self.today_feeding_count}")
        except Exception as e:
            print(f"Error saving daily feeding count: {e}")


# Main entry point
if __name__ == "__main__":
    try:
        # Create application
        app = QApplication(sys.argv)

        # Create and show main window
        window = SmartFishFeederApp()
        window.show()

        # Run application event loop
        sys.exit(app.exec_())
    except Exception as e:
        # Print the error
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()

        # Keep console window open to see the error
        input("Press Enter to exit...")

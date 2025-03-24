import traceback
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
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Check if ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not found. Installing...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Check if filterpy is installed
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    print("FilterPy not found. Installing...")
    os.system("pip install filterpy")
    from filterpy.kalman import KalmanFilter

# Constants
MAX_FEEDING_mode_DURATION = 600  # 10 minutes maximum for any feeding mode
MAX_DURING_mode_NO_DOSAGE = 300  # 5 minutes maximum in "feeding" mode with no dosage
MAX_FEEDING_DURATION = 600  # 10 minutes
MAX_TRAJECTORY_LEN = 30  # Number of frames to keep for trajectory lines
SPEED_HISTORY_LEN = 600  # Number of readings to keep for graphs
MONITORING_WINDOW = 300  # 5 minutes (300 seconds) monitoring window
FEED_DECISION_INTERVAL = 60  # Check for feeding every 60s
MIN_FEED_INTERVAL = 7200  # Minimum seconds between feedings (2 hours)
INITIAL_FEED_DELAY = 300  # Seconds before first feeding analysis (15 minutes)
PRE_FEEDING_DURATION = 300  # 5 minutes pre-feeding analysis
POST_FEEDING_DURATION = 300  # 5 minutes post-feeding analysis
DOSAGE_ASSESSMENT_PERIOD = 30  # 30 seconds to assess each dosage effect
DAILY_OPERATION_START = "07:30"  # Daily start time
DAILY_OPERATION_END = "23:59"  # Daily end time
DATA_FOLDER = "fish_data"  # Folder to store CSV and model data
# YOLOv8 model path with fallback checks
DEFAULT_MODEL_PATH = "runs/detect/train8/weights/best.pt"
MODEL_PATH = DEFAULT_MODEL_PATH
DEBUG_MODE = False  # Set to False to disable debug output
# Check if model exists, if not try to find it elsewhere
if not os.path.exists(MODEL_PATH):
    # Try alternative locations
    alt_paths = [
        "best.pt",
        "weights/best.pt",
        os.path.join(os.getcwd(), "best.pt")
    ]
    for path in alt_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            print(f"Found model at: {MODEL_PATH}")
            break
    # If still not found, we'll use a default YOLO model later

# Create data folder if it doesn't exist
os.makedirs(DATA_FOLDER, exist_ok=True)


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

                if DEBUG_MODE:# Debug output to compare raw vs filtered
                    print(f"Fish speed - Raw: {self.raw_speed:.2f}, Filtered: {self.filtered_speed:.2f} BL/s")

        return np.array([cx, cy])

    def get_speed(self):
        """Return the Kalman-filtered speed value"""
        return self.filtered_speed

    def get_trajectory(self):
        return list(self.positions)


class VideoThread(QThread):
    """Thread for processing video frames with YOLO detection"""
    frame_ready = pyqtSignal(np.ndarray, list, float, float)

    def __init__(self, camera_source=0):
        super().__init__()
        self.camera_source = camera_source
        self.running = False
        self.trackers = {}
        self.next_id = 0
        self.model = None
        self.speed_history = deque(maxlen=600)  # Limit to 10 minutes of data (at 1 sample per second)
        self.variance_history = deque(maxlen=600)
        self.timestamps = deque(maxlen=600)

        # Rolling window for real-time analysis only - constantly overwritten
        self.current_window_speeds = deque(maxlen=300)  # Just 5 minutes (monitoring window)
        self.current_window_variances = deque(maxlen=300)
        self.current_window_timestamps = deque(maxlen=300)

        # Permanent storage only for significant events
        self.feeding_event_data = []  # List of dictionaries containing pre/during/post data
        self.ensure_data_collection()

    def run(self):
        print("VideoThread started. Initialising data collection...")
        try:
            # Check if model path exists
            if not os.path.exists(MODEL_PATH):
                print(f"ERROR: Model file not found at {MODEL_PATH}")
                print("Using YOLO default model for testing purposes.")
                # Use a default YOLO model for testing
                self.model = YOLO("yolov8n.pt")
            else:
                # Load YOLOv8 model
                self.model = YOLO(MODEL_PATH)

            # Open camera with error handling
            try:
                cap = cv2.VideoCapture(self.camera_source)
                if not cap.isOpened():
                    print(f"ERROR: Failed to open camera source {self.camera_source}")
                    print("Using test video or fallback mechanism...")
                    # Try another source or create a black frame for testing
                    cap = cv2.VideoCapture(0)  # Try default camera
                    if not cap.isOpened():
                        # Create dummy frames for testing UI
                        while self.running:
                            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(dummy_frame, "Camera Unavailable", (50, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                            # Use dummy data
                            tracks = []
                            avg_speed = 0.5 + 0.2 * np.sin(time.time())
                            speed_variance = 0.1 + 0.05 * np.cos(time.time())

                            # Record dummy data
                            current_time = datetime.datetime.now()
                            self.speed_history.append(avg_speed)
                            self.variance_history.append(speed_variance)
                            self.timestamps.append(current_time)

                            # Emit frame
                            self.frame_ready.emit(dummy_frame, tracks, avg_speed, speed_variance)
                            time.sleep(0.1)
                        return
            except Exception as e:
                print(f"Camera initialisation error: {e}")
                return

            self.running = True

            while self.running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from camera. Retrying...")
                        # Wait and retry a few times before giving up
                        time.sleep(0.5)
                        continue

                    # Run YOLOv8 detection and tracking
                    try:
                        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.7)
                    except Exception as e:
                        print(f"YOLO tracking error: {e}")
                        # Use empty results
                        tracks = []
                        avg_speed = 0.0 if not self.speed_history else self.speed_history[-1]
                        speed_variance = 0.0 if not self.variance_history else self.variance_history[-1]

                        # Record data
                        current_time = datetime.datetime.now()
                        self.speed_history.append(avg_speed)
                        self.variance_history.append(speed_variance)
                        self.timestamps.append(current_time)

                        if DEBUG_MODE:
                            print(
                                f"Added data point: Speed={avg_speed:.2f}, Variance={speed_variance:.2f}, Total points={len(self.speed_history)}")

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
                        if speeds:
                            current_time = datetime.datetime.now()
                            avg_speed = np.mean(speeds)
                            speed_variance = np.var(speeds) if len(speeds) > 1 else 0
                        else:
                            current_time = datetime.datetime.now()
                            # Use previous values or defaults
                            avg_speed = 0.0 if not self.speed_history else self.speed_history[-1]
                            speed_variance = 0.0 if not self.variance_history else self.variance_history[-1]

                        self.speed_history.append(avg_speed)
                        self.variance_history.append(speed_variance)
                        self.timestamps.append(current_time)

                        if DEBUG_MODE:
                            print(
                                f"Added data point: Speed={avg_speed:.2f}, Variance={speed_variance:.2f}, Total points={len(self.speed_history)}")

                        # Emit frame with tracking data and metrics
                        self.frame_ready.emit(
                            frame,
                            tracks,
                            avg_speed,
                            speed_variance
                        )
                    else:
                        # No results, emit frame with empty tracking
                        current_time = datetime.datetime.now()
                        avg_speed = 0.0 if not self.speed_history else self.speed_history[-1]
                        speed_variance = 0.0 if not self.variance_history else self.variance_history[-1]

                        self.speed_history.append(avg_speed)
                        self.variance_history.append(speed_variance)
                        self.timestamps.append(current_time)

                        if DEBUG_MODE:
                            print(
                                f"Added data point: Speed={avg_speed:.2f}, Variance={speed_variance:.2f}, Total points={len(self.speed_history)}")

                        self.frame_ready.emit(frame, [], avg_speed, speed_variance)

                    # Small delay to reduce CPU usage
                    time.sleep(0.01)

                except Exception as e:
                    print(f"Frame processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.5)  # Delay before trying again

            cap.release()

        except Exception as e:
            print(f"Video thread critical error: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        self.running = False
        self.wait()

    def get_speed_data(self):
        if DEBUG_MODE:
            print(f"get_speed_data called. Data points: {len(self.timestamps)}")
            if len(self.timestamps) > 0:
                print(f"First timestamp: {self.timestamps[0]}, Last timestamp: {self.timestamps[-1]}")
                print(
                    f"Speed range: {min(self.speed_history) if self.speed_history else 0}-{max(self.speed_history) if self.speed_history else 0}")
        return list(self.timestamps), list(self.speed_history), list(self.variance_history)

    def ensure_data_collection(self):
        """Add dummy data if necessary for testing graphs"""
        if len(self.timestamps) < 2:
            if DEBUG_MODE:
                print("Adding initial dummy data points for graph testing")

            current_time = datetime.datetime.now()
            earlier_time = current_time - datetime.timedelta(seconds=10)

            # Add two points 10 seconds apart to establish timeline
            self.timestamps.append(earlier_time)
            self.speed_history.append(0.5)
            self.variance_history.append(0.1)

            self.timestamps.append(current_time)
            self.speed_history.append(0.7)
            self.variance_history.append(0.2)

            if DEBUG_MODE:
                print(f"Added dummy points. Now have {len(self.timestamps)} timestamps")

    # You might also want to add a method to purge old data
    def purge_old_data(self, max_age_seconds=600):
        """Remove data older than max_age_seconds (10 minutes) to prevent memory issues"""
        if not self.timestamps:
            return

        current_time = datetime.datetime.now()
        cutoff_time = current_time - datetime.timedelta(seconds=max_age_seconds)

        # Find index of first item newer than cutoff
        cutoff_index = 0
        for i, timestamp in enumerate(self.timestamps):
            if timestamp >= cutoff_time:
                cutoff_index = i
                break

        # If we found old data, remove it
        if cutoff_index > 0:
            # Convert to list, truncate, and convert back to deque
            timestamps = list(self.timestamps)[cutoff_index:]
            speeds = list(self.speed_history)[cutoff_index:]
            variances = list(self.variance_history)[cutoff_index:]

            self.timestamps = deque(timestamps, maxlen=self.timestamps.maxlen)
            self.speed_history = deque(speeds, maxlen=self.speed_history.maxlen)
            self.variance_history = deque(variances, maxlen=self.variance_history.maxlen)

            print(f"Purged {cutoff_index} old data points to save memory")

    def get_current_window_data(self):
        """Return the current 5-minute window for analysis"""
        return list(self.current_window_timestamps), list(self.current_window_speeds), list(
            self.current_window_variances)

    def get_feeding_event_data(self, event_type=None):
        """Get feeding event data, optionally filtered by event type"""
        if event_type:
            return [data for data in self.feeding_event_data if data.get("event_type") == event_type]
        return self.feeding_event_data

    def add_data_point(self, timestamp, speed, variance, event_type=None):
        """Add a data point with optional event tagging"""
        # Add to rolling window
        self.current_window_speeds.append(speed)
        self.current_window_variances.append(variance)
        self.current_window_timestamps.append(timestamp)

        # If this is a significant event, store it permanently
        if event_type:
            if not hasattr(self, 'feeding_event_data'):
                self.feeding_event_data = []

            self.feeding_event_data.append({
                "timestamp": timestamp,
                "speed": speed,
                "variance": variance,
                "event_type": event_type
            })

class SpeedGraph(FigureCanvasQTAgg):
    """Matplotlib graph for displaying speed metrics"""

    def __init__(self, parent=None, width=5, height=3, dpi=100):
        plt.ioff()  # Turn off interactive mode
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(SpeedGraph, self).__init__(self.fig)
        self.setParent(parent)

        # Initialize plot with empty data
        self.speed_line, = self.axes.plot([], [], 'b-', label='Average Speed', linewidth=1)
        self.axes.set_ylim(0, 2)
        self.axes.set_title('Fish Speed (body lengths/second)')
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Speed')
        self.axes.legend(loc='upper right')
        self.axes.grid(True)
        self.fig.tight_layout()

        # For satiated region and window
        self.satiated_region = None
        self.window_indicator = None

        # For tracking if data has been plotted
        self.has_data = False
        print("SpeedGraph initialized")

    def update_plot(self, timestamps, speeds, satiated_range=None):
        if DEBUG_MODE:
            print(f"SpeedGraph.update_plot called with {len(timestamps)} points")

        if not timestamps or not speeds or len(timestamps) < 2:
            print("Not enough data points to plot speed")
            return

        try:
            # Convert timestamps to matplotlib format - FIXED LINE
            import matplotlib.dates
            dates = matplotlib.dates.date2num(timestamps)
            if DEBUG_MODE:
                print(f"Speed data range: {min(speeds):.2f}-{max(speeds):.2f}")

            # Update data
            self.speed_line.set_data(dates, speeds)

            # Set x limits based on data
            self.axes.set_xlim(min(dates), max(dates))

            # Set y limits based on data (with padding)
            max_speed = max(speeds) if speeds else 2
            self.axes.set_ylim(0, max(2, max_speed * 1.2))

            # Format x-axis
            from matplotlib.dates import DateFormatter
            self.axes.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            for label in self.axes.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

            # Force line to be visible with specific style
            self.speed_line.set_visible(True)
            self.speed_line.set_linewidth(1)
            self.speed_line.set_color('blue')

            # Print the actual plot data for debugging
            if DEBUG_MODE:
                x_data, y_data = self.speed_line.get_data()
                print(f"Plotting speed data: {len(x_data)} points")
                if len(x_data) > 0:
                    print(f"X range: {min(x_data):.2f}-{max(x_data):.2f}, Y range: {min(y_data):.2f}-{max(y_data):.2f}")
                    print(f"Axes limits: x={self.axes.get_xlim()}, y={self.axes.get_ylim()}")

            # Handle satiated region
            if hasattr(self, 'satiated_region') and self.satiated_region is not None:
                try:
                    self.satiated_region.remove()
                    self.satiated_region = None
                except:
                    pass  # Ignore removal errors

            if satiated_range:
                y_min, y_max = satiated_range
                self.satiated_region = self.axes.axhspan(y_min, y_max, alpha=0.2, color='green', label='Satiated Range')

            # Handle window indicator
            if hasattr(self, 'window_indicator') and self.window_indicator is not None:
                try:
                    self.window_indicator.remove()
                    self.window_indicator = None
                except:
                    pass  # Ignore removal errors

            if len(dates) > 2:
                # Calculate window start (5 minutes before last point)
                window_start_date = max(dates) - MONITORING_WINDOW / (24 * 60 * 60)

                # Find index of window start
                window_start_index = 0
                for i, date in enumerate(dates):
                    if date >= window_start_date:
                        window_start_index = i
                        break

                # Add window indicator
                if window_start_index < len(dates):
                    self.window_indicator = self.axes.axvspan(
                        dates[window_start_index], dates[-1],
                        alpha=0.15, color='blue', label='Analysis Window'
                    )

            # Mark that we have data
            self.has_data = True

            # Force redraw
            self.fig.canvas.draw()

            print("Speed graph updated successfully")

        except Exception as e:
            print(f"Error updating speed graph: {e}")
            import traceback
            traceback.print_exc()


class VarianceGraph(FigureCanvasQTAgg):
    """Matplotlib graph for displaying variance metrics"""

    def __init__(self, parent=None, width=5, height=3, dpi=100):
        plt.ioff()  # Turn off interactive mode
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(VarianceGraph, self).__init__(self.fig)
        self.setParent(parent)

        # Initialize plot with empty data
        self.variance_line, = self.axes.plot([], [], 'r-', label='Speed Variance', linewidth=1)
        self.axes.set_ylim(0, 1)
        self.axes.set_title('Speed Variance')
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Variance')
        self.axes.legend(loc='upper right')
        self.axes.grid(True)
        self.fig.tight_layout()

        # For satiated region and window
        self.satiated_region = None
        self.window_indicator = None

        # For tracking if data has been plotted
        self.has_data = False
        print("VarianceGraph initialized")

    def update_plot(self, timestamps, variances, satiated_range=None):
        if DEBUG_MODE:
            print(f"VarianceGraph.update_plot called with {len(timestamps)} points")

        if not timestamps or not variances or len(timestamps) < 2:
            print("Not enough data points to plot variance")
            return

        try:
            # Convert timestamps to matplotlib format - FIXED LINE
            import matplotlib.dates
            dates = matplotlib.dates.date2num(timestamps)
            if DEBUG_MODE:
                print(f"Variance data range: {min(variances):.2f}-{max(variances):.2f}")

            # Update data
            self.variance_line.set_data(dates, variances)

            # Set x limits based on data
            self.axes.set_xlim(min(dates), max(dates))

            # Set y limits based on data (with padding)
            max_variance = max(variances) if variances else 1
            self.axes.set_ylim(0, max(1, max_variance * 1.2))

            # Format x-axis
            from matplotlib.dates import DateFormatter
            self.axes.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            for label in self.axes.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

            # Force line to be visible with specific style
            self.variance_line.set_visible(True)
            self.variance_line.set_linewidth(1)
            self.variance_line.set_color('red')

            # Print the actual plot data for debugging
            x_data, y_data = self.variance_line.get_data()
            if DEBUG_MODE:
                print(f"Plotting variance data: {len(x_data)} points")
                if len(x_data) > 0:
                    print(f"X range: {min(x_data):.2f}-{max(x_data):.2f}, Y range: {min(y_data):.2f}-{max(y_data):.2f}")
                    print(f"Axes limits: x={self.axes.get_xlim()}, y={self.axes.get_ylim()}")

            # Handle satiated region
            if hasattr(self, 'satiated_region') and self.satiated_region is not None:
                try:
                    self.satiated_region.remove()
                    self.satiated_region = None
                except:
                    pass  # Ignore removal errors

            if satiated_range:
                y_min, y_max = satiated_range
                self.satiated_region = self.axes.axhspan(y_min, y_max, alpha=0.2, color='green', label='Satiated Range')

            # Handle window indicator
            if hasattr(self, 'window_indicator') and self.window_indicator is not None:
                try:
                    self.window_indicator.remove()
                    self.window_indicator = None
                except:
                    pass  # Ignore removal errors

            if len(dates) > 2:
                # Calculate window start (5 minutes before last point)
                window_start_date = max(dates) - MONITORING_WINDOW / (24 * 60 * 60)

                # Find index of window start
                window_start_index = 0
                for i, date in enumerate(dates):
                    if date >= window_start_date:
                        window_start_index = i
                        break

                # Add window indicator
                if window_start_index < len(dates):
                    self.window_indicator = self.axes.axvspan(
                        dates[window_start_index], dates[-1],
                        alpha=0.15, color='blue', label='Analysis Window'
                    )

            # Mark that we have data
            self.has_data = True

            # Force redraw
            self.fig.canvas.draw()

            print("Variance graph updated successfully")

        except Exception as e:
            print(f"Error updating variance graph: {e}")
            import traceback
            traceback.print_exc()


class FeedingHistoryTable(QWidget):
    """Widget for displaying feeding history using a proper table widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Initialize max visible rows
        self.max_visible_rows = 10  # Default value
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
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Time column
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Dosages column

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

        # Configure row height
        self.table.verticalHeader().setVisible(False)  # Hide row numbers
        self.table.verticalHeader().setDefaultSectionSize(24)  # Compact row height

        self.layout.addWidget(self.table)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def update_history(self, feeding_history):
        # Store the full feeding history
        self.full_history = feeding_history

        # Clear table
        self.table.setRowCount(0)

        # Calculate how many rows to show (use max_visible_rows if defined)
        if hasattr(self, 'max_visible_rows'):
            rows_to_show = min(len(feeding_history), self.max_visible_rows)
        else:
            rows_to_show = min(len(feeding_history), 10)  # Default to 10 rows

        # Add rows for each feeding event (most recent first)
        for i, feed in enumerate(feeding_history[:rows_to_show]):
            self.table.insertRow(i)

            # Format time to be more readable with %m/%d %H:%M
            time_str = feed['timestamp'].strftime("%m/%d %H:%M")
            time_item = QTableWidgetItem(time_str)
            self.table.setItem(i, 0, time_item)

            # Display dosage count (or "Missed" if it was a missed feeding)
            is_missed = feed.get('missed', False)

            if is_missed:
                dosage_item = QTableWidgetItem("Missed")
                dosage_item.setForeground(QColor(100, 100, 255))  # Blue for missed
            else:
                dosage_count = feed.get('dosage_count', 1)
                dosage_item = QTableWidgetItem(str(dosage_count))

            dosage_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 1, dosage_item)

    # Add to your SmartFishFeederApp class
    def resizeEvent(self, event):
        """Handle window resize event"""
        # Call parent implementation if it exists
        if hasattr(super(), 'resizeEvent'):
            super().resizeEvent(event)

        # Adjust table height based on available space
        if hasattr(self, 'forecast_table'):
            total_height = self.height()
            if total_height < 800:  # Compact mode for smaller windows
                self.forecast_table.setMaximumHeight(80)  # Show fewer rows
                if hasattr(self, 'forecast_canvas_widget'):
                    self.forecast_canvas_widget.setMinimumHeight(150)
            else:  # More space in larger windows
                self.forecast_table.setMaximumHeight(120)
                if hasattr(self, 'forecast_canvas_widget'):
                    self.forecast_canvas_widget.setMinimumHeight(180)

    def update_display_rows(self):
        """Update how many rows are displayed based on available space"""
        # Store current data
        current_data = []
        for row in range(self.table.rowCount()):
            row_data = []
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    row_data.append((item.text(), item.background()))
                else:
                    row_data.append(("", None))
            current_data.append(row_data)

        # Determine how many rows to show
        rows_to_show = min(len(current_data), self.max_visible_rows)

        # Reset table
        self.table.setRowCount(rows_to_show)

        # Refill with data
        for row in range(rows_to_show):
            for col in range(self.table.columnCount()):
                text, bg = current_data[row][col]
                item = QTableWidgetItem(text)
                if bg:
                    item.setBackground(bg)
                self.table.setItem(row, col, item)

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
        print("Started direct post-feeding check timer")

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

            self.feeding_active = False
            self.feed_start_time = None
            self.pre_feeding_speeds = []
            self.pre_feeding_variances = []
            self.during_feeding_speeds = []
            self.during_feeding_variances = []
            self.post_feeding_speeds = []
            self.post_feeding_variances = []
            # Replace with:
            self.system_mode = "initialising"  # initialising, monitoring, pre_feeding, feeding, post_feeding, cooldown
            self.last_feed_check = datetime.datetime.now()
            self.last_dosage_time = None
            self.dosage_count = 0
            self.cooldown_active = False
            self.cooldown_end_time = None

            # Check if we should be in cooldown based on last feeding time
            if self.feeding_model.feeding_history:
                last_feed = self.feeding_model.feeding_history[-1]['timestamp']
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
            self.today_feeding_count = 0
            self.today_date = datetime.datetime.now().date()

            # UI setup
            self.setup_ui()

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
            self.daily_timer.start(60000)  # Check every minute

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

            # Start button state diagnostic timer
            QTimer.singleShot(5000, self.check_button_state)

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
        """Update the displayed video frame with tracking info"""
        # Log data
        current_time = datetime.datetime.now()
        self.data_logger.log_data(current_time, avg_speed, speed_variance,
                                  feeding_event=(self.system_mode == "feeding"))

        # Add data to 1-second display buffers
        self.speed_buffer.append(avg_speed)
        self.variance_buffer.append(speed_variance)

        # Add data to appropriate mode arrays
        if self.system_mode == "monitoring" or self.system_mode == "initialising":
            # Add data to 5-minute window analysis
            self.feeding_model.add_data_point(current_time, avg_speed, speed_variance)
        elif self.system_mode == "pre_feeding":
            self.pre_feeding_speeds.append(avg_speed)
            self.pre_feeding_variances.append(speed_variance)
            # Also track in feeding model with event type
            self.feeding_model.add_data_point(current_time, avg_speed, speed_variance, event_type="pre_feeding")
        elif self.system_mode == "feeding":
            self.during_feeding_speeds.append(avg_speed)
            self.during_feeding_variances.append(speed_variance)
            # Also track in feeding model with event type
            self.feeding_model.add_data_point(current_time, avg_speed, speed_variance, event_type="during_feeding")
        elif self.system_mode == "post_feeding":
            self.post_feeding_speeds.append(avg_speed)
            self.post_feeding_variances.append(speed_variance)
            # Also track in feeding model with event type
            self.feeding_model.add_data_point(current_time, avg_speed, speed_variance, event_type="post_feeding")
            # Check if post-feeding mode is complete (30 seconds) and needs to be completed
            if len(self.post_feeding_speeds) >= 30:
                if not hasattr(self, 'post_feeding_timer_set'):
                    # Set attribute to prevent multiple timers
                    self.post_feeding_timer_set = True
                    # Start a timer to complete the feeding cycle
                    self.post_feeding_timer = QTimer()
                    self.post_feeding_timer.setSingleShot(True)
                    self.post_feeding_timer.timeout.connect(self.complete_feeding_cycle)
                    self.post_feeding_timer.start(
                        POST_FEEDING_DURATION * 1000 - 30000)  # Adjusted for the 30 seconds we already have
                    print(f"[DEBUG] Post-feeding timer set to complete in {POST_FEEDING_DURATION - 30} seconds")




        # Draw bounding boxes and trajectories
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        # Create a copy of the frame for drawing
        draw_frame = frame.copy()

        # Draw tracks
        for track_id, box, trajectory, color in tracks:
            x1, y1, x2, y2 = box
            # Draw bounding box
            cv2.rectangle(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw trajectory
            points = trajectory[-30:]
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(draw_frame, points[i], points[i + 1], color, 2)
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

    def update_feeding_count_display(self):
        """Update the feeding count display"""
        self.feeding_count_label.setText(f"Today's feedings: {self.today_feeding_count}")

        # Add warning if approaching max daily feedings
        if self.today_feeding_count >= 4:
            self.feeding_count_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.feeding_count_label.setStyleSheet("")

    @staticmethod
    def excepthook(exc_type, exc_value, exc_tb):
        """Global exception handler for PyQt slots"""
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"UNHANDLED EXCEPTION:\n{tb}")
        # Continue normal exception processing
        sys.__excepthook__(exc_type, exc_value, exc_tb)

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

        speed_box = QGroupBox("Current Speed")
        speed_box_layout = QVBoxLayout()
        self.speed_value = QLabel("0.00")
        self.speed_value.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.speed_value.setAlignment(Qt.AlignCenter)
        speed_box_layout.addWidget(self.speed_value)
        speed_box.setLayout(speed_box_layout)

        variance_box = QGroupBox("Current Variance")
        variance_box_layout = QVBoxLayout()
        self.variance_value = QLabel("0.00")
        self.variance_value.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.variance_value.setAlignment(Qt.AlignCenter)
        variance_box_layout.addWidget(self.variance_value)
        variance_box.setLayout(variance_box_layout)

        metrics_layout.addWidget(speed_box)
        metrics_layout.addWidget(variance_box)

        # Add to control layout
        control_layout.addWidget(self.feed_button)
        control_layout.addWidget(self.feeding_count_label)
        control_layout.addLayout(metrics_layout)
        control_layout.addStretch()

        # Graphs
        self.speed_graph = SpeedGraph(self, width=5, height=3)
        self.variance_graph = VarianceGraph(self, width=5, height=3)

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

        # Add these widgets to your layout
        # For example, if you have a control_layout:
        control_layout.addWidget(status_box)
        control_layout.addWidget(self.feed_button)
        control_layout.addWidget(self.status_label)

    def update_display(self):
        """Update the UI displays using just the data we need"""
        # Add post-feeding completion check
        current_time = datetime.datetime.now()

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

        # For live graphs, we only need the current monitoring window
        timestamps, speeds, variances = self.video_thread.get_current_window_data()

        # For feeding history display, use the significant stored events
        feeding_events = self.video_thread.get_feeding_event_data("complete_feeding")
        try:
            # Get data from video thread
            timestamps, speeds, variances = self.video_thread.get_speed_data()

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

                # Mark missed feeding events on graphs with blue vertical lines
                if hasattr(self.feeding_model, 'missed_feedings'):
                    for missed in self.feeding_model.missed_feedings:
                        # Convert timestamp to matplotlib format
                        import matplotlib.dates
                        event_time = matplotlib.dates.date2num(missed['timestamp'])

                        # Add blue dotted line on both graphs
                        if hasattr(self, 'speed_graph') and self.speed_graph.axes:
                            self.speed_graph.axes.axvline(x=event_time, color='blue', linestyle='--', alpha=0.5)
                        if hasattr(self, 'variance_graph') and self.variance_graph.axes:
                            self.variance_graph.axes.axvline(x=event_time, color='blue', linestyle='--', alpha=0.5)

                    # Force redraw of graphs
                    if hasattr(self, 'speed_graph'):
                        self.speed_graph.fig.canvas.draw()
                    if hasattr(self, 'variance_graph'):
                        self.variance_graph.fig.canvas.draw()

            # Update feeding history - include both regular and missed feedings
            if hasattr(self.feeding_model, 'missed_feedings'):
                # Get all feedings to display
                all_feedings = self.feeding_model.feeding_history.copy()

                # Add missed feedings with appropriate formatting
                for missed in self.feeding_model.missed_feedings:
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
                self.feeding_history_table.update_history(self.feeding_model.feeding_history)

            # Check if new day started
            self.check_day_change()

        except Exception as e:
            print(f"Error in update_display: {e}")
            import traceback
            traceback.print_exc()

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

    def analyze_hunger_patterns(self):
        """Display hunger pattern analysis from missed and confirmed feedings"""
        if not hasattr(self.feeding_model, 'hunger_patterns') or not self.feeding_model.hunger_patterns:
            QMessageBox.information(self, "Hunger Pattern Analysis",
                                    "Not enough data to analyze hunger patterns yet.")
            return

        patterns = self.feeding_model.hunger_patterns

        # Find the most consistent patterns
        pattern_scores = []

        for hour, pattern in patterns.items():
            if pattern['count'] < 3:
                continue

            # Calculate consistency score based on range vs average
            speed_range = pattern['speed_range'][1] - pattern['speed_range'][0]
            speed_avg = pattern['avg_speed']
            var_range = pattern['variance_range'][1] - pattern['variance_range'][0]
            var_avg = pattern['avg_variance']

            # Lower score is better (more consistent)
            score = (speed_range / speed_avg if speed_avg > 0 else 999) + \
                    (var_range / var_avg if var_avg > 0 else 999)

            pattern_scores.append((hour, pattern['count'], score, pattern))

        # Sort by count (descending) then score (ascending)
        pattern_scores.sort(key=lambda x: (-x[1], x[2]))

        # Build message
        if not pattern_scores:
            QMessageBox.information(self, "Hunger Pattern Analysis",
                                    "Not enough consistent data to analyze hunger patterns yet.")
            return

        message = "Hunger Pattern Analysis:\n\n"

        for hour, count, score, pattern in pattern_scores[:5]:  # Show top 5
            message += f"Hour {hour}:00 ({count} samples):\n"
            message += f"  Avg Speed: {pattern['avg_speed']:.2f} BL/s\n"
            message += f"  Speed Range: {pattern['speed_range'][0]:.2f}-{pattern['speed_range'][1]:.2f} BL/s\n"
            message += f"  Consistency: {100 / (score if score > 0 else 999):.1f}%\n\n"

        message += "These patterns will be used to improve hunger detection."

        QMessageBox.information(self, "Hunger Pattern Analysis", message)

    def update_status_display(self):
        # Near the beginning, add this check
        if self.cooldown_active and self.cooldown_end_time:
            # Make sure any "FEEDING RECOMMENDED" message is cleared during cooldown
            if "FEEDING RECOMMENDED" in self.status_label.text():
                remaining_minutes = int((self.cooldown_end_time - datetime.datetime.now()).total_seconds() / 60)
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
                progress = int((elapsed_time / MONITORING_WINDOW) * 100)
                self.time_label.setText(f"Initialising: {minutes:02d}:{seconds:02d} remaining ({progress}%)")
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
                f"Pre-feeding: {minutes:02d}:{seconds:02d} (Remaining: {rem_min:02d}:{rem_sec:02d})")

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
            self.time_label.setText(f"Post-feeding: Remaining: {rem_min:02d}:{rem_sec:02d}")

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

    def check_button_state(self):
        """Diagnostic method to print button state"""
        print(f"Button state: Enabled={self.feed_button.isEnabled()}, Text={self.feed_button.text()}")
        print(f"Current mode: {self.system_mode}")

        # Force button to be enabled if we're in during mode with no dosages
        if self.system_mode == "feeding" and self.dosage_count == 0:
            print("Forcing button to enabled state")
            self.feed_button.setEnabled(True)
            self.feed_button.setText("CONFIRM FEED")
            self.feed_button.setStyleSheet(
                "background-color: #FF4500; color: white; font-size: 18px; font-weight: bold;")

        # Schedule next check
        QTimer.singleShot(5000, self.check_button_state)  # Check again in 5 seconds

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

            # Reset mode and clear data
            self.system_mode = "monitoring"
            self.mode_start_times["monitoring"] = current_time

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

        # Skip if feeding is already active
        if self.system_mode != "monitoring":
            return

        # Skip if in cooldown period
        if self.cooldown_active:
            if current_time < self.cooldown_end_time:
                remaining_minutes = (self.cooldown_end_time - current_time).total_seconds() / 60
                self.status_label.setText(
                    f"Cooldown active. Next feeding available in {int(remaining_minutes)} minutes")
                return
            else:
                self.cooldown_active = False

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
        if self.system_mode == "monitoring" and not self.cooldown_active:
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
        # Stop video thread
        self.video_thread.stop()
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

        # Skip if we're in monitoring mode
        if self.system_mode == "monitoring":
            return

        if self.system_mode == "cooldown":
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
        if hasattr(self.feeding_model, 'feeding_history') and self.feeding_model.feeding_history:
            last_feed = self.feeding_model.feeding_history[-1]['timestamp']
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
            has_history = len(self.feeding_model.feeding_history) > 0
            print(f"Feeding history found: {has_history} ({len(self.feeding_model.feeding_history)} records)")
        else:
            print("No feeding_history attribute found in model")

        # Single decision point for what to do next
        if not has_history:
            print("NO FEEDING HISTORY DETECTED - FORCING FIRST FEEDING CYCLE")
            # Use a short delay to allow UI to update first
            QTimer.singleShot(2000, lambda: self.start_feeding_cycle(current_time))
        else:
            # Continue with normal operation using Prophet
            print(f"Using existing feeding history with {len(self.feeding_model.feeding_history)} records")

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

    def check_button_state(self):
        """Diagnostic method to monitor and fix button state"""
        current_time = datetime.datetime.now()
        print(f"\n----- Button State Check ({current_time}) -----")
        print(f"Button state: Enabled={self.feed_button.isEnabled()}, Text={self.feed_button.text()}")
        print(f"Current mode: {self.system_mode}")
        print(f"Status text: {self.status_label.text()}")

        # First, ensure button is disabled during cooldown regardless of other conditions
        if self.system_mode == "cooldown" or self.cooldown_active:
            if self.feed_button.isEnabled():
                print("FIXING: Disabling feed button during cooldown period")
                self.feed_button.setEnabled(False)
                self.feed_button.setText("COOLDOWN")
                self.feed_button.setStyleSheet(
                    "background-color: #808080; color: white; font-size: 18px; font-weight: bold;")
            return  # Exit early to prevent other conditions from enabling the button

        # Fix common state issues:

        # Case 1: We're in "feeding" mode but button is disabled or incorrectly connected
        if self.system_mode == "feeding" and self.dosage_count == 0:
            if not self.feed_button.isEnabled():
                print("FIXING: Enabling button in during mode with no dosages")
                self.feed_button.setEnabled(True)
                self.feed_button.setText("CONFIRM FEED")
                self.feed_button.setStyleSheet(
                    "background-color: #FF4500; color: white; font-size: 18px; font-weight: bold;")
                # Fix connection
                self.fix_button_connections()

        # Case 2: Feeding is recommended but button is disabled
        if "FEEDING RECOMMENDED" in self.status_label.text() and not self.feed_button.isEnabled():
            print("FIXING: Enabling button for feeding recommendation")
            self.feed_button.setEnabled(True)
            self.feed_button.setText("CONFIRM FEED")
            self.feed_button.setStyleSheet(
                "background-color: #FF4500; color: white; font-size: 18px; font-weight: bold;")
            # Fix connection
            self.fix_button_connections()

        # Schedule next check
        QTimer.singleShot(2000, self.check_button_state)  # Check more frequently (every 2 seconds)

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
            from matplotlib.figure import Figure
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
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
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
        """Create a standalone dashboard window for Prophet model visualizations"""
        try:
            # Create a new window
            self.prophet_dashboard = QMainWindow()
            self.prophet_dashboard.setWindowTitle("Fish Feeding Prophet Dashboard")
            self.prophet_dashboard.setGeometry(200, 200, 800, 600)

            # Create central widget and layout
            central_widget = QWidget()
            self.prophet_dashboard.setCentralWidget(central_widget)
            main_layout = QVBoxLayout()
            central_widget.setLayout(main_layout)

            # Add header
            header_label = QLabel("Prophet Model Predictions")
            header_label.setStyleSheet("font-size: 18px; font-weight: bold;")
            main_layout.addWidget(header_label)

            # Add forecast graph
            forecast_box = QGroupBox("24-Hour Hunger Forecast")
            forecast_layout = QVBoxLayout()

            # Create matplotlib canvas
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            fig = self.feeding_model.get_forecast_plot()
            if fig:
                canvas = FigureCanvasQTAgg(fig)
                forecast_layout.addWidget(canvas)
            else:
                canvas_placeholder = QLabel("Forecast not available - need more feeding data")
                forecast_layout.addWidget(canvas_placeholder)

            forecast_box.setLayout(forecast_layout)
            main_layout.addWidget(forecast_box)

            # Add hourly predictions table
            hourly_box = QGroupBox("Hourly Feeding Recommendations")
            hourly_layout = QVBoxLayout()

            hourly_table = QTableWidget()
            hourly_table.setColumnCount(4)
            hourly_table.setHorizontalHeaderLabels(["Time", "Hunger Score", "Uncertainty", "Recommendation"])

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

                    # Uncertainty
                    uncertainty = hour_data['upper_bound'] - hour_data['lower_bound']
                    uncertainty_item = QTableWidgetItem(f"{uncertainty:.2f}")
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

            # Add feeding history impact
            history_box = QGroupBox("Feeding History Impact")
            history_layout = QVBoxLayout()

            # Create text summary
            if self.feeding_model.trained and self.feeding_model.prophet_model:
                history_text = QLabel(
                    f"Prophet model trained with {len(self.feeding_model.feeding_history)} feeding events\n"
                    f"and {len(self.feeding_model.missed_feedings)} missed feedings.\n\n"
                    f"The model has identified daily patterns in fish hunger, with peak hunger typically\n"
                    f"occurring at specific times of day based on your feeding history."
                )
            else:
                history_text = QLabel(
                    "Not enough feeding history to analyze patterns yet.\n"
                    "Continue using the system to accumulate more data."
                )

            history_layout.addWidget(history_text)
            history_box.setLayout(history_layout)
            main_layout.addWidget(history_box)

            # Add explanation
            explanation = QLabel(
                "The Prophet model analyzes your feeding history to predict when fish are likely to be hungry.\n"
                "Green areas show predicted times of satiation, while red areas indicate likely hunger periods.\n"
                "The system will automatically recommend feedings based on these predictions and current behavior."
            )
            explanation.setStyleSheet("font-style: italic;")
            main_layout.addWidget(explanation)

            # Add refresh button
            refresh_button = QPushButton("Refresh Data")
            refresh_button.clicked.connect(lambda: self.update_prophet_dashboard())
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

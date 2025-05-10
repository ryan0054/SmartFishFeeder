import pandas as pd
import numpy as np
from prophet import Prophet
import datetime
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import deque
import time


class ProphetFeedingModel:
    """ML model for smart feeding decisions using Facebook Prophet for time series analysis"""

    def __init__(self):
        # Initialize with default values
        self.prophet_model = None
        self.trained = False
        self.feeding_history = []
        self.MIN_SAMPLES_FOR_TRAINING = 5  # Need at least 5 feedings to start learning
        self.model_file = os.path.join("fish_data", "prophet_feeding_model.pkl")

        # For 5-minute window analysis
        self.current_window_speeds = deque(maxlen=300)  # 5 minutes (300 seconds)
        self.current_window_variances = deque(maxlen=300)
        self.current_window_timestamps = deque(maxlen=300)

        # Track missed feeding events
        self.missed_feedings = []
        self.hunger_patterns = {}  # To store identified hunger patterns

        # Load model if available
        self.load_model()

        # After loading, prune to prevent performance issues
        self.prune_feeding_history(max_entries=100)

        self._cache = {
            'feeding_history': None,
            'missed_feedings': None,
            'satiated_ranges': None,
            'daily_forecast': None,
            'forecast_plot': None
        }
        self._cache_timestamps = {k: 0 for k in self._cache.keys()}
        self._cache_lifetimes = {
            'feeding_history': 60,  # 1 minute
            'missed_feedings': 60,  # 1 minute
            'satiated_ranges': 300,  # 5 minutes
            'daily_forecast': 300,  # 5 minutes
            'forecast_plot': 600  # 10 minutes
        }

    def load_model(self):
        """Load model from disk if available with missed feedings"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.prophet_model = data.get('prophet_model')
                    self.feeding_history = data.get('history', [])
                    self.trained = data.get('trained', False)
                    self.missed_feedings = data.get('missed_feedings', [])
                    self.hunger_patterns = data.get('hunger_patterns', {})

                print(f"Loaded Prophet feeding model from disk with {len(self.feeding_history)} feedings")
                print(f"Loaded {len(self.missed_feedings)} missed feeding records")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")

                # Try backup file if available
                backup_file = self.model_file + '.bak'
                if os.path.exists(backup_file):
                    try:
                        print("Attempting to load from backup...")
                        with open(backup_file, 'rb') as f:
                            data = pickle.load(f)
                            self.prophet_model = data.get('prophet_model')
                            self.feeding_history = data.get('history', [])
                            self.trained = data.get('trained', False)
                            self.missed_feedings = data.get('missed_feedings', [])
                            self.hunger_patterns = data.get('hunger_patterns', {})
                        print("Successfully loaded model from backup")
                        return True
                    except Exception as backup_e:
                        print(f"Error loading backup: {backup_e}")

                # Initialize empty if both failed
                print("Starting with empty history")
                self.feeding_history = []
                self.missed_feedings = []

        else:
            print(f"No model file found at {self.model_file}, starting with empty history")
            self.feeding_history = []
            self.missed_feedings = []

        return False

    def save_model(self):
        """Save model to disk including missed feedings"""
        # Prune before saving to keep file size manageable
        self.prune_feeding_history(max_entries=100)
        try:
            # Make sure missed_feedings exists
            if not hasattr(self, 'missed_feedings'):
                self.missed_feedings = []

            data = {
                'prophet_model': self.prophet_model,
                'history': self.feeding_history,
                'trained': self.trained,
                'missed_feedings': self.missed_feedings,
                'hunger_patterns': self.hunger_patterns
            }

            # Create a backup of the existing file first
            if os.path.exists(self.model_file):
                backup_file = self.model_file + '.bak'
                try:
                    import shutil
                    shutil.copy2(self.model_file, backup_file)
                    print(f"Created backup of feeding model at {backup_file}")
                except Exception as e:
                    print(f"Warning: Could not create backup: {e}")

            # Write the file
            with open(self.model_file, 'wb') as f:
                pickle.dump(data, f)

            print(
                f"Successfully saved Prophet model with {len(self.feeding_history)} feedings and {len(self.missed_feedings)} missed feedings")

        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()

    def add_feeding_data(self, timestamp, pre_speeds, pre_variances,
                         during_speeds, during_variances,
                         post_speeds, post_variances, dosage_count=1, manual=False):
        """Add data from a feeding event"""
        # Calculate features from the feeding event
        features = {
            'pre_speed_mean': np.mean(pre_speeds) if pre_speeds else 0,
            'pre_speed_var': np.var(pre_speeds) if pre_speeds else 0,
            'pre_var_mean': np.mean(pre_variances) if pre_variances else 0,
            'pre_var_var': np.var(pre_variances) if pre_variances else 0,
            'during_speed_mean': np.mean(during_speeds) if during_speeds else 0,
            'during_var_mean': np.mean(during_variances) if during_variances else 0,
            'post_speed_mean': np.mean(post_speeds) if post_speeds else 0,
            'post_var_mean': np.mean(post_variances) if post_variances else 0,
            'time_since_last_feed': 0,
            'time_of_day': timestamp.hour + timestamp.minute / 60.0,
            'manual': manual,
            'hunger_score': self._calculate_hunger_score(pre_speeds, pre_variances)
        }

        # Calculate time since last feeding
        if self.feeding_history:
            last_feed = self.feeding_history[-1]['timestamp']
            time_diff = (timestamp - last_feed).total_seconds() / 3600.0  # Hours
            features['time_since_last_feed'] = time_diff

        # Add to history with explicit dosage count
        self.feeding_history.append({
            'timestamp': timestamp,
            'features': features,
            'manual': manual,
            'dosage_count': dosage_count,
            'hunger_score': features['hunger_score']
        })

        self._cache['feeding_history'] = None
        self._cache['satiated_ranges'] = None
        self._cache['daily_forecast'] = None
        self._cache['forecast_plot'] = None

        # Train model if we have enough data
        if len(self.feeding_history) >= self.MIN_SAMPLES_FOR_TRAINING:
            self.train_prophet_model()

        # Save updated model
        self.save_model()

        # Reset 5-minute window analysis
        self.reset_window()

        print(f"Added feeding event at {timestamp} with {dosage_count} dosages")

    def _calculate_hunger_score(self, speeds, variances):
        """Calculate a numeric hunger score based on pre-feeding metrics"""
        if not speeds or not variances:
            return 0.5  # Default moderate hunger if no data

        # Get the key metrics
        avg_speed = np.mean(speeds)
        avg_variance = np.mean(variances)

        # Calculate trend in speed (is it increasing?)
        if len(speeds) >= 10:
            first_half = speeds[:len(speeds) // 2]
            second_half = speeds[len(speeds) // 2:]
            speed_trend = np.mean(second_half) / np.mean(first_half) if np.mean(first_half) > 0 else 1.0
        else:
            speed_trend = 1.0

        # Combine metrics into a hunger score
        # Higher values = more hungry
        base_score = (avg_speed * 0.6) + (avg_variance * 0.4)

        # Adjust for trend - increasing speed is a stronger hunger signal
        trend_factor = 1.0 + max(0, min(1, (speed_trend - 1) * 2))

        # Final score (normalize to typical range)
        normalized_score = base_score * trend_factor / 1.5

        # Ensure reasonable bounds
        return max(0.1, min(3.0, normalized_score))

    def reset_window(self):
        """Reset the rolling window for behavior analysis"""
        self.current_window_speeds = deque(maxlen=300)
        self.current_window_variances = deque(maxlen=300)
        self.current_window_timestamps = deque(maxlen=300)

    def add_data_point(self, timestamp, speed, variance, event_type=None):
        """Add a data point with optional event tagging"""
        # Always add to rolling window
        self.current_window_speeds.append(speed)
        self.current_window_variances.append(variance)
        self.current_window_timestamps.append(timestamp)

    def get_current_window_data(self):
        """Return the current monitoring window data"""
        return list(self.current_window_timestamps), list(self.current_window_speeds), list(
            self.current_window_variances)

    def train_prophet_model(self):
        """Train the Prophet time series model"""
        if len(self.feeding_history) < self.MIN_SAMPLES_FOR_TRAINING:
            print(
                f"Not enough feeding events for training. Need {self.MIN_SAMPLES_FOR_TRAINING}, have {len(self.feeding_history)}")
            return False

        try:
            # Prepare data for Prophet (needs 'ds' for dates and 'y' for values)
            feed_data = []

            for feed in self.feeding_history:
                # Extract timestamp and hunger score
                feed_data.append({
                    'ds': feed['timestamp'],
                    'y': feed['hunger_score'],
                    'hour': feed['timestamp'].hour,
                    'minute': feed['timestamp'].minute,
                    'day_of_week': feed['timestamp'].weekday(),
                    'dosage_count': feed.get('dosage_count', 1)
                })

            # Create DataFrame for Prophet
            df = pd.DataFrame(feed_data)

            # Add hour as a regressor since time of day affects fish hunger
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.5,  # Increase this from 0.1 to 0.5
                seasonality_mode='multiplicative',
                uncertainty_samples=2000,  # Add this parameter - default is 1000
                interval_width=0.95  # Add this parameter - default is 0.8
            )

            # Add regressors
            model.add_regressor('hour')

            # Fit the model
            model.fit(df)

            # Save the trained model
            self.prophet_model = model
            self.trained = True

            print(f"Successfully trained Prophet model with {len(feed_data)} feeding events")

            # Create and save a forecast visualization for debugging/validation
            self._create_forecast_visualization()

            if self.trained:
                self._cache['daily_forecast'] = None
                self._cache['forecast_plot'] = None

            return True

        except Exception as e:
            print(f"Error training Prophet model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_forecast_visualization(self):
        """Create and save a visualization of the model's predictions"""
        if not self.trained or not self.prophet_model:
            return

        try:
            # Create future dataframe for the next week
            future = self.prophet_model.make_future_dataframe(
                periods=24 * 7,  # 7 days ahead
                freq='H'  # Hourly frequency
            )

            # Add hour regressor
            future['hour'] = future['ds'].dt.hour

            # Make prediction
            forecast = self.prophet_model.predict(future)

            # Create figure
            fig = plt.figure(figsize=(12, 8))

            # Plot the forecast on first subplot
            ax1 = fig.add_subplot(211)
            ax1.plot(forecast['ds'], forecast['yhat'], label='Prediction')
            ax1.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                             color='gray', alpha=0.2, label='Uncertainty')

            # Plot actual data points
            feed_times = [feed['timestamp'] for feed in self.feeding_history]
            feed_hunger = [feed['hunger_score'] for feed in self.feeding_history]
            ax1.scatter(feed_times, feed_hunger, color='red', label='Actual Feedings', s=50)

            ax1.set_title('Hunger Score Forecast')
            ax1.set_ylabel('Hunger Score')
            ax1.legend()

            # For components, we need to create a new figure since plot_components doesn't support ax parameter
            # Save the original figure
            plt.tight_layout()
            plt.savefig(os.path.join("fish_data", "prophet_forecast.png"))
            plt.close(fig)

            # Create a separate figure for components
            self.prophet_model.plot_components(forecast)
            plt.tight_layout()
            plt.savefig(os.path.join("fish_data", "prophet_components.png"))
            plt.close()

            print("Saved Prophet forecast visualization")

        except Exception as e:
            print(f"Error creating forecast visualization: {e}")
            import traceback
            traceback.print_exc()

    def should_feed(self, current_time, min_hours_since_last_feed=2.0):
        """Determine if feeding is recommended based on Prophet model and current behavior"""
        try:
            # Check if enough time has passed since last feeding
            if self.feeding_history:
                last_feed = self.feeding_history[-1]['timestamp']
                hours_since_last_feed = (current_time - last_feed).total_seconds() / 3600.0
                if hours_since_last_feed < min_hours_since_last_feed:
                    print(
                        f"Too soon since last feeding ({hours_since_last_feed:.1f} hrs < {min_hours_since_last_feed} hrs)")
                    return False

            # Get current window data
            if len(self.current_window_speeds) < 30:  # Need at least 30 seconds of data
                print("Not enough current window data for decision")
                return False

            # Calculate current hunger metrics
            current_speed_mean = np.mean(self.current_window_speeds)
            current_var_mean = np.mean(self.current_window_variances)

            # First, check if we have a trained Prophet model
            if self.trained and self.prophet_model:
                # Create a dataframe for prediction
                future = pd.DataFrame([{
                    'ds': current_time,
                    'hour': current_time.hour,
                    'missed': 0  # Add this line - 0 means not a missed feeding
                }])

                # Make prediction
                forecast = self.prophet_model.predict(future)
                predicted_hunger = forecast.iloc[0]['yhat']
                upper_bound = forecast.iloc[0]['yhat_upper']
                lower_bound = forecast.iloc[0]['yhat_lower']
                uncertainty = upper_bound - lower_bound

                print(f"Prophet prediction: Hunger={predicted_hunger:.2f}, Uncertainty={uncertainty:.2f}")

                # Get typical satiated ranges for comparison
                speed_range, var_range = self.get_satiated_ranges()

                # Combine Prophet prediction with current behavior

                # 1. Check time-based prediction from Prophet
                # Higher predicted hunger score makes us more likely to feed
                prophet_feed_signal = predicted_hunger > 1.3  # Threshold can be tuned

                # 2. Check current behavior metrics
                # Compare current values with satiated ranges
                speed_above_satiated = current_speed_mean / speed_range[1] if speed_range[1] > 0 else 1.0
                var_above_satiated = current_var_mean / var_range[1] if var_range[1] > 0 else 1.0

                # Behavior-based hunger signal
                behavior_feed_signal = (speed_above_satiated > 1.3 and var_above_satiated > 1.3)

                # 3. Check time since last feed as additional factor
                time_feed_signal = False
                if hours_since_last_feed > 4.0:  # Been a while since feeding
                    time_feed_signal = (speed_above_satiated > 1.2 or predicted_hunger > 1.2)

                # 4. Calculate final decision - weighted combination of signals
                # Adjust weights based on uncertainty of Prophet prediction
                if uncertainty > 0.8:  # High uncertainty
                    # Rely more on behavior
                    prophet_weight = 0.3
                    behavior_weight = 0.7
                else:  # Low uncertainty
                    # Rely more on Prophet
                    prophet_weight = 0.6
                    behavior_weight = 0.4

                # Calculate weighted decision
                feed_score = (prophet_weight * (1 if prophet_feed_signal else 0) +
                              behavior_weight * (1 if behavior_feed_signal else 0))

                # Include time factor to break ties
                if time_feed_signal:
                    feed_score += 0.2

                # Apply threshold
                hunger_detected = feed_score >= 0.5

                print(f"Feed decision: Prophet={prophet_feed_signal}, Behavior={behavior_feed_signal}, " +
                      f"Time={time_feed_signal}, Score={feed_score:.2f}, Decision={'Feed' if hunger_detected else 'Dont Feed'}")

                return hunger_detected

            else:
                # Fall back to basic behavior analysis if no trained model
                print("No trained Prophet model - using behavior-based decision")

                # Get satiated ranges for comparison from previous feeding data
                speed_range, var_range = self.get_satiated_ranges()

                # Check if current values exceed thresholds for hunger
                speed_above_satiated = current_speed_mean / speed_range[1] if speed_range[1] > 0 else 1.0
                var_above_satiated = current_var_mean / var_range[1] if var_range[1] > 0 else 1.0

                # Hunger detected if activity is significantly above satiated levels
                hunger_detected = (speed_above_satiated > 1.3 and var_above_satiated > 1.3)

                # Also check time since last feed as a factor
                if hours_since_last_feed > 4.0:  # Been a while since feeding
                    hunger_detected = hunger_detected or (speed_above_satiated > 1.2)

                print(f"Behavior-based decision: Speed ratio={speed_above_satiated:.2f}, " +
                      f"Variance ratio={var_above_satiated:.2f}, Hours since feed={hours_since_last_feed:.1f}, " +
                      f"Decision={'Feed' if hunger_detected else 'Dont Feed'}")

                if not self.trained and hours_since_last_feed >= 4.0:
                    if not hunger_detected:
                        print(f"SAFETY: Forcing feeding after {hours_since_last_feed:.1f} hours despite behavior")
                        hunger_detected = True

                return hunger_detected

        except Exception as e:
            print(f"Error in should_feed: {e}")
            import traceback
            traceback.print_exc()
            return False  # Default to not feeding on error

    def get_satiated_ranges(self):
        """Get cached satiated ranges"""

        def calculate_ranges():
            if not hasattr(self, 'feeding_history') or not self.feeding_history:
                return (0, 0.5), (0, 0.1)  # Default ranges

            # Extract post-feeding speeds and variances
            post_speeds = [feed['features']['post_speed_mean'] for feed in self.feeding_history
                           if 'features' in feed and feed['features']['post_speed_mean'] > 0]
            post_vars = [feed['features']['post_var_mean'] for feed in self.feeding_history
                         if 'features' in feed and feed['features']['post_var_mean'] > 0]

            if not post_speeds or not post_vars:
                return (0, 0.5), (0, 0.1)  # Default ranges

            # Calculate range with some buffer
            speed_mean = np.mean(post_speeds)
            speed_std = np.std(post_speeds) if len(post_speeds) > 1 else 0.1 * speed_mean
            var_mean = np.mean(post_vars)
            var_std = np.std(post_vars) if len(post_vars) > 1 else 0.1 * var_mean

            speed_range = (max(0, speed_mean - speed_std), speed_mean + speed_std)
            var_range = (max(0, var_mean - var_std), var_mean + var_std)

            return speed_range, var_range

        return self._get_cached('satiated_ranges', calculate_ranges)

    def add_missed_feeding(self, timestamp, pre_speeds, pre_variances):
        """Record a missed feeding (user unavailable) as valid hunger data with persistence"""
        # Calculate hunger score
        hunger_score = self._calculate_hunger_score(pre_speeds, pre_variances)

        # Extract basic features from the pre-feeding data
        features = {
            'pre_speed_mean': np.mean(pre_speeds) if pre_speeds else 0,
            'pre_speed_var': np.var(pre_speeds) if pre_speeds else 0,
            'pre_var_mean': np.mean(pre_variances) if pre_variances else 0,
            'pre_var_var': np.var(pre_variances) if pre_variances else 0,
            'time_of_day': timestamp.hour + timestamp.minute / 60.0,
            'missed': True,  # Flag to indicate this was a missed feeding
            'hunger_score': hunger_score
        }

        # Calculate time since last feeding
        if self.feeding_history:
            last_feed = self.feeding_history[-1]['timestamp']
            time_diff = (timestamp - last_feed).total_seconds() / 3600.0  # Hours
            features['time_since_last_feed'] = time_diff
        else:
            features['time_since_last_feed'] = 24.0  # Default to 24 hours if no history

        # Add to missed feedings list
        self.missed_feedings.append({
            'timestamp': timestamp,
            'features': features,
            'pre_speeds': pre_speeds.copy() if isinstance(pre_speeds, list) else list(pre_speeds),
            'pre_variances': pre_variances.copy() if isinstance(pre_variances, list) else list(pre_variances),
            'hunger_score': hunger_score
        })

        self._cache['missed_feedings'] = None
        self._cache['daily_forecast'] = None
        self._cache['forecast_plot'] = None

        # Update the hunger patterns database
        self.update_hunger_patterns(features)

        # Save the model to disk immediately
        self.save_model()

        print(f"Recorded missed feeding at {timestamp} - Data saved to disk for future use")

        # If enough missed feedings, also incorporate them into the Prophet model
        if self.trained and self.prophet_model and len(self.missed_feedings) >= 3:
            self.incorporate_missed_feedings()

    def incorporate_missed_feedings(self):
        """Incorporate missed feeding data into the Prophet model"""
        if not self.missed_feedings:
            return

        try:
            # Create dataframe with all feeding data (real and missed)
            all_feed_data = []

            # Add regular feedings
            for feed in self.feeding_history:
                all_feed_data.append({
                    'ds': feed['timestamp'],
                    'y': feed.get('hunger_score', self._calculate_hunger_score([], [])),
                    'hour': feed['timestamp'].hour,
                    'missed': 0
                })

            # Add missed feedings
            for feed in self.missed_feedings:
                all_feed_data.append({
                    'ds': feed['timestamp'],
                    'y': feed.get('hunger_score', 1.5),  # Default to higher hunger if not calculated
                    'hour': feed['timestamp'].hour,
                    'missed': 1
                })

            # Create DataFrame
            df = pd.DataFrame(all_feed_data)

            # Sort by timestamp
            df = df.sort_values('ds')

            # Retrain Prophet model with combined data
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.1,
                seasonality_mode='multiplicative'
            )

            # Add regressors
            model.add_regressor('hour')
            model.add_regressor('missed')

            # Fit the model
            model.fit(df)

            # Update the Prophet model
            self.prophet_model = model
            self.trained = True

            print(f"Retrained Prophet model incorporating {len(self.missed_feedings)} missed feedings")

            # Update visualization
            self._create_forecast_visualization()

        except Exception as e:
            print(f"Error incorporating missed feedings: {e}")
            import traceback
            traceback.print_exc()

    def update_hunger_patterns(self, features):
        """Update database of known hunger patterns based on confirmed or missed feedings"""
        # Round time of day to the nearest hour for pattern tracking
        hour = int(features['time_of_day'])

        # Get the existing pattern for this hour or create a new one
        if hour not in self.hunger_patterns:
            self.hunger_patterns[hour] = {
                'speed_samples': [],
                'variance_samples': [],
                'count': 0
            }

        # Add this sample to the pattern
        pattern = self.hunger_patterns[hour]
        pattern['speed_samples'].append(features['pre_speed_mean'])
        pattern['variance_samples'].append(features['pre_var_mean'])
        pattern['count'] += 1

        # Calculate updated averages and ranges
        if len(pattern['speed_samples']) > 0:
            pattern['avg_speed'] = np.mean(pattern['speed_samples'])
            pattern['avg_variance'] = np.mean(pattern['variance_samples'])
            pattern['speed_range'] = (np.min(pattern['speed_samples']), np.max(pattern['speed_samples']))
            pattern['variance_range'] = (np.min(pattern['variance_samples']), np.max(pattern['variance_samples']))

        print(
            f"Updated hunger pattern for hour {hour}: {pattern['count']} samples, avg speed: {pattern['avg_speed']:.2f}")

    def get_prediction_for_time(self, target_time):
        """Get Prophet's prediction for a specific time"""
        if not self.trained or not self.prophet_model:
            return None

        try:
            # Create a dataframe for prediction
            future = pd.DataFrame([{
                'ds': target_time,
                'hour': target_time.hour,
                'missed': 0  # Add missed regressor
            }])

            # Make prediction
            forecast = self.prophet_model.predict(future)

            result = {
                'hunger_score': forecast.iloc[0]['yhat'],
                'lower_bound': forecast.iloc[0]['yhat_lower'],
                'upper_bound': forecast.iloc[0]['yhat_upper'],
                'uncertainty': forecast.iloc[0]['yhat_upper'] - forecast.iloc[0]['yhat_lower'],
                'trend': forecast.iloc[0]['trend'],
                'time': target_time
            }

            return result

        except Exception as e:
            print(f"Error in get_prediction_for_time: {e}")
            return None

    def get_daily_forecast(self):
        """Get cached daily forecast"""

        def calculate_forecast():
            if not self.trained or not self.prophet_model:
                return None

            try:
                # Create a DataFrame for the next 24 hours
                now = datetime.datetime.now()
                start = now.replace(minute=0, second=0, microsecond=0)
                hours = [start + datetime.timedelta(hours=i) for i in range(24)]

                future = pd.DataFrame([{
                    'ds': hour,
                    'hour': hour.hour,
                    'missed': 0  # Add missed regressor
                } for hour in hours])

                # Make prediction
                forecast = self.prophet_model.predict(future)

                # Debug - print raw forecast values for first few hours
                print("Raw Prophet forecast (first 3 hours):")
                print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(3))

                # Format results
                results = []
                for i in range(len(forecast)):
                    # Apply bounds to all values
                    hunger_score = max(0.1, min(3.0, forecast.iloc[i]['yhat']))
                    lower_bound = max(0.1, min(3.0, forecast.iloc[i]['yhat_lower']))
                    upper_bound = max(0.1, min(3.0, forecast.iloc[i]['yhat_upper']))

                    # Ensure upper is actually higher than lower
                    if upper_bound <= lower_bound:
                        # If they're equal, add at least a small uncertainty
                        upper_bound = lower_bound + 0.2

                    # Debug uncertainty value
                    uncertainty = upper_bound - lower_bound
                    if i < 5:  # Only log first 5 hours
                        print(f"Hour {hours[i].hour}: score={hunger_score:.4f}, " +
                              f"lower={lower_bound:.4f}, upper={upper_bound:.4f}, " +
                              f"uncertainty={uncertainty:.4f}")

                    results.append({
                        'time': hours[i],
                        'hour': hours[i].hour,
                        'hunger_score': hunger_score,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'uncertainty': uncertainty,  # Store uncertainty directly
                        'recommended': hunger_score > 1.3  # Threshold for recommendation
                    })

                return results

            except Exception as e:
                print(f"Error in get_daily_forecast: {e}")
                import traceback
                traceback.print_exc()
                return None

        return self._get_cached('daily_forecast', calculate_forecast)

    def get_forecast_plot(self):
        """Get cached forecast plot"""

        def generate_plot():
            if not self.trained or not self.prophet_model:
                return None

            try:
                # Create future dataframe for the next 36 hours
                now = datetime.datetime.now()
                future = pd.DataFrame([{
                    'ds': now + datetime.timedelta(hours=i),
                    'hour': (now.hour + i) % 24,
                    'missed': 0  # Include the missed regressor
                } for i in range(36)])

                # Make prediction
                forecast = self.prophet_model.predict(future)

                # Enforce valid hunger score range (0.1 to 3.0) for visualization
                forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0.1, min(3.0, x)))
                forecast['yhat_lower'] = forecast['yhat_lower'].apply(lambda x: max(0.1, x))
                forecast['yhat_upper'] = forecast['yhat_upper'].apply(lambda x: min(3.0, x))

                # Create a matplotlib figure
                fig = Figure(figsize=(8, 4), dpi=100)
                ax = fig.add_subplot(111)

                # Plot prediction
                times = [pd.to_datetime(time) for time in forecast['ds']]
                ax.plot(times, forecast['yhat'], 'b-', label='Predicted Hunger')

                # Plot uncertainty
                ax.fill_between(times, forecast['yhat_lower'], forecast['yhat_upper'],
                                color='blue', alpha=0.2, label='Uncertainty')

                # Add threshold line for feeding
                ax.axhline(y=1.3, color='r', linestyle='--', label='Feeding Threshold')

                # Set y-axis limits to enforce valid range and prevent negative values
                ax.set_ylim(bottom=0, top=3.5)  # Show full range with a bit of padding

                # Format x-axis to show hours
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

                # Add labels
                ax.set_xlabel('Time')
                ax.set_ylabel('Hunger Score')
                ax.set_title('Forecasted Fish Hunger')
                ax.legend()

                # Rotate x labels
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')

                fig.tight_layout()
                return fig

            except Exception as e:
                print(f"Error generating forecast plot: {e}")
                import traceback
                traceback.print_exc()
                return None

        return self._get_cached('forecast_plot', generate_plot)

    def analyze_hunger_from_missed_feedings(self):
        """Analyze hunger patterns from missed feedings"""
        print("Analyzing hunger patterns from missed feedings")
        if not hasattr(self, 'missed_feedings') or not self.missed_feedings:
            print("No missed feedings to analyze")
            return

        print(f"Analyzing {len(self.missed_feedings)} missed feeding events")

        # Extract key data from missed feedings
        for missed in self.missed_feedings:
            # Update hunger patterns database
            if 'features' in missed:
                self.update_hunger_patterns(missed['features'])

        # If we have enough missed feedings, incorporate them into the model
        if len(self.missed_feedings) >= 3:
            self.incorporate_missed_feedings()

    def force_complete_retrain(self):
        print("===== PROPHET DIAGNOSIS =====")
        print(f"Regular feedings: {len(self.feeding_model.feeding_history)}")
        print(f"Missed feedings: {len(self.feeding_model.missed_feedings)}")

        # Force incorporate missed feedings
        self.feeding_model.incorporate_missed_feedings()

        # Train with complete data
        success = self.feeding_model.train_prophet_model(force=True)
        print(f"Training result: {success}")

        # Check forecast
        forecast = self.feeding_model.get_daily_forecast()
        if forecast:
            print(f"Forecast available with {len(forecast)} entries")
            print(f"First forecast entry: {forecast[0]}")
        else:
            print("No forecast available after training")

        # Force save
        self.feeding_model.save_model()
        print("Model saved to disk")

        # Update UI
        self.update_prophet_forecast()

    def check_prophet_data(self):
        if not hasattr(self.feeding_model, 'prophet_model') or self.feeding_model.prophet_model is None:
            print("No Prophet model exists yet")
            return

        # Check Prophet model internals
        model = self.feeding_model.prophet_model
        if hasattr(model, 'history') and model.history is not None:
            print(f"Prophet model contains {len(model.history)} training points")
            print(f"First few training points:")
            print(model.history.head())
        else:
            print("Prophet model exists but has no history data")

        # Force a new forecast and print details
        forecast = self.feeding_model.get_daily_forecast()
        if forecast:
            print(f"Generated forecast with {len(forecast)} hours")
            for i, f in enumerate(forecast[:3]):  # Print first 3 hours
                print(f"Hour {i}: score={f['hunger_score']:.2f}, recommended={f['recommended']}")
        else:
            print("Failed to generate forecast")

    def _get_cached(self, key, generator_func):
        """Generic caching method with time-based invalidation"""
        current_time = time.time()
        lifetime = self._cache_lifetimes.get(key, 60)  # Default 60s

        # Check if cache needs refresh
        if (self._cache[key] is None or
                current_time - self._cache_timestamps[key] > lifetime):
            # Generate fresh data
            self._cache[key] = generator_func()
            self._cache_timestamps[key] = current_time

        return self._cache[key]

    def get_feeding_history(self):
        """Get cached feeding history"""
        return self._get_cached('feeding_history',
                                lambda: self.feeding_history.copy() if hasattr(self, 'feeding_history') else [])

    def get_missed_feedings(self):
        """Get cached missed feedings"""
        return self._get_cached('missed_feedings',
                                lambda: self.missed_feedings.copy() if hasattr(self, 'missed_feedings') else [])

    def prune_feeding_history(self, max_entries=100):
        """Limit the size of feeding history to prevent performance issues"""
        if hasattr(self, 'feeding_history') and len(self.feeding_history) > max_entries:
            # Sort by timestamp (newest first)
            sorted_history = sorted(self.feeding_history,
                                    key=lambda x: x['timestamp'],
                                    reverse=True)
            # Keep only the most recent entries
            self.feeding_history = sorted_history[:max_entries]
            print(f"Pruned feeding history to {max_entries} entries")

            # Invalidate cache
            self._cache['feeding_history'] = None

        # Also prune missed feedings
        if hasattr(self, 'missed_feedings') and len(self.missed_feedings) > max_entries // 2:
            sorted_missed = sorted(self.missed_feedings,
                                   key=lambda x: x['timestamp'],
                                   reverse=True)
            self.missed_feedings = sorted_missed[:max_entries // 2]
            print(f"Pruned missed feedings to {max_entries // 2} entries")

            # Invalidate cache
            self._cache['missed_feedings'] = None

"""
Wearable Echomyography System for Muscle Activity Monitoring

This system implements ultrasound-based muscle monitoring using a single transducer,

The system can monitor:
- Diaphragm activity for breathing pattern recognition
- Forearm muscle activity for hand gesture tracking

Author: Pradosh P. Dash
Date: 2025
inspired by: Gao, Xiaoxiang, et al. "A wearable echomyography system based on a single transducer." Nature Electronics 7.11 (2024): 1035-1046.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import ndimage
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class UltrasoundSignalProcessor:
    """
    Core signal processing class for echomyography signals
    
    Handles filtering, envelope detection, and feature extraction
    from raw ultrasound radio-frequency (RF) signals.
    """
    
    def __init__(self, sampling_rate: int = 12_000_000, frame_rate: int = 50):
        """
        Initialize the signal processor
        
        Args:
            sampling_rate: ADC sampling rate in Hz (12 MHz default)
            frame_rate: Frame acquisition rate in Hz (50 Hz default)
        """
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.samples_per_frame = 1024  # Fixed frame size
        
        # Pre-calculate time axis for efficiency
        self.time_axis = np.linspace(0, self.samples_per_frame / sampling_rate * 1e6, 
                                   self.samples_per_frame)  # Time in microseconds
        
    def apply_bandpass_filter(self, rf_signal: np.ndarray, 
                             low_freq: float = 2e6, high_freq: float = 6e6) -> np.ndarray:
        """
        Apply bandpass filter to RF signal
        
        Args:
            rf_signal: Raw RF signal
            low_freq: Lower cutoff frequency
            high_freq: Higher cutoff frequency
            
        Returns:
            Filtered RF signal
        """
        nyquist = self.sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        filtered_signal = signal.sosfilt(sos, rf_signal)
        
        return filtered_signal
    
    def extract_envelope(self, rf_signal: np.ndarray) -> np.ndarray:
        """
        Extract signal envelope using Hilbert transform
        
        Args:
            rf_signal: Filtered RF signal
            
        Returns:
            Signal envelope
        """
        analytic_signal = signal.hilbert(rf_signal)
        envelope = np.abs(analytic_signal)
        return envelope
    
    def detect_tissue_boundaries(self, envelope: np.ndarray, 
                                prominence: float = 0.1) -> np.ndarray:
        """
        Detect tissue boundaries from envelope signal
        
        Args:
            envelope: Signal envelope
            prominence: Minimum peak prominence
            
        Returns:
            Array of peak indices representing tissue boundaries
        """
        # Normalize envelope
        normalized_envelope = envelope / np.max(envelope)
        
        # Find peaks representing tissue interfaces
        peaks, _ = signal.find_peaks(normalized_envelope, 
                                   prominence=prominence,
                                   distance=20)  # Minimum distance between peaks
        
        return peaks
    
    def calculate_tissue_thickness(self, boundary_indices: np.ndarray) -> float:
        """
        Calculate tissue thickness from boundary detection
        
        Args:
            boundary_indices: Indices of detected boundaries
            
        Returns:
            Tissue thickness in millimeters
        """
        if len(boundary_indices) < 2:
            return 0.0
        
        # Speed of sound in tissue (approximately 1540 m/s)
        sound_speed = 1540  # m/s
        
        # Calculate depth difference between boundaries
        depth_samples = boundary_indices[-1] - boundary_indices[0]
        depth_time = depth_samples / self.sampling_rate  # seconds
        
        # Distance = (speed × time) / 2 (round trip)
        thickness_meters = (sound_speed * depth_time) / 2
        thickness_mm = thickness_meters * 1000  # Convert to mm
        
        return thickness_mm


class DiaphragmMonitor:
    """
    Specialized class for diaphragm monitoring and breathing pattern analysis
    
    Analyzes diaphragm thickness changes to classify breathing modes:
    - Abdominal (diaphragmatic) breathing
    - Thoracic (shallow) breathing
    """
    
    def __init__(self, signal_processor: UltrasoundSignalProcessor):
        """
        Initialize diaphragm monitor
        
        Args:
            signal_processor: Instance of UltrasoundSignalProcessor
        """
        self.processor = signal_processor
        self.thickness_history = []
        self.breathing_cycles = []
        
    def analyze_breathing_cycle(self, thickness_sequence: List[float], 
                              window_size: int = 300) -> Dict[str, float]:
        """
        Analyze a breathing cycle to extract respiratory parameters
        
        Args:
            thickness_sequence: Sequence of diaphragm thickness measurements
            window_size: Analysis window size in frames
            
        Returns:
            Dictionary containing breathing parameters
        """
        if len(thickness_sequence) < window_size:
            return {"dtf": 0.0, "respiratory_rate": 0.0, "breathing_mode": "unknown"}
        
        # Use last window_size measurements
        recent_thickness = np.array(thickness_sequence[-window_size:])
        
        # Find peaks (inspiration) and troughs (expiration)
        peaks, _ = signal.find_peaks(recent_thickness, distance=50)
        troughs, _ = signal.find_peaks(-recent_thickness, distance=50)
        
        if len(peaks) == 0 or len(troughs) == 0:
            return {"dtf": 0.0, "respiratory_rate": 0.0, "breathing_mode": "unknown"}
        
        # Calculate Diaphragm Thickening Fraction (DTF)
        max_thickness = np.mean(recent_thickness[peaks])
        min_thickness = np.mean(recent_thickness[troughs])
        
        if min_thickness > 0:
            dtf = (max_thickness - min_thickness) / min_thickness
        else:
            dtf = 0.0
        
        # Calculate respiratory rate
        time_window = window_size / self.processor.frame_rate  # seconds
        respiratory_rate = len(peaks) / time_window * 60  # breaths per minute
        
        # Classify breathing mode based on DTF
        if dtf > 0.25:
            breathing_mode = "abdominal"
        elif dtf > 0.10:
            breathing_mode = "mixed"
        else:
            breathing_mode = "thoracic"
        
        return {
            "dtf": dtf,
            "respiratory_rate": respiratory_rate,
            "breathing_mode": breathing_mode,
            "max_thickness": max_thickness,
            "min_thickness": min_thickness
        }
    
    def process_rf_frame(self, rf_signal: np.ndarray) -> Dict[str, float]:
        """
        Process a single RF frame for diaphragm monitoring
        
        Args:
            rf_signal: Raw RF signal frame
            
        Returns:
            Processing results including thickness and breathing parameters
        """
        # Apply signal processing pipeline
        filtered_signal = self.processor.apply_bandpass_filter(rf_signal)
        envelope = self.processor.extract_envelope(filtered_signal)
        boundaries = self.processor.detect_tissue_boundaries(envelope)
        thickness = self.processor.calculate_tissue_thickness(boundaries)
        
        # Add to history
        self.thickness_history.append(thickness)
        
        # Analyze breathing if we have enough data
        breathing_params = self.analyze_breathing_cycle(self.thickness_history)
        
        return {
            "thickness": thickness,
            "num_boundaries": len(boundaries),
            **breathing_params
        }


class HandGestureClassifier:
    """
    Deep learning-based hand gesture classifier using ultrasound signals
    
    Uses a 1D CNN to classify hand gestures from forearm muscle ultrasound signals.
    Capable of tracking 13 degrees of freedom (10 finger joints + 3 wrist rotations).
    """
    
    def __init__(self, input_length: int = 1024, num_joints: int = 13):
        """
        Initialize the gesture classifier
        
        Args:
            input_length: Length of input RF signal
            num_joints: Number of tracked joint angles
        """
        self.input_length = input_length
        self.num_joints = num_joints
        self.model = self._build_network()
        self.training_history = None
        
    def _build_network(self) -> keras.Model:
        """
        Build the 1D CNN architecture for gesture classification
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=(self.input_length, 1), name='rf_input')
        
        # Feature extraction layers
        x = inputs
        
        # Stack of 1D convolutional layers with increasing filters
        conv_configs = [
            (32, 11, 2),   # (filters, kernel_size, strides)
            (64, 9, 2),
            (128, 7, 2),
            (256, 5, 2),
            (512, 3, 2),
            (512, 3, 1),
            (256, 3, 1),
            (128, 3, 1),
        ]
        
        for i, (filters, kernel_size, strides) in enumerate(conv_configs):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                activation='relu',
                kernel_initializer='he_normal',
                name=f'conv1d_{i+1}'
            )(x)
            
            # Add batch normalization for deeper layers
            if filters >= 256:
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            # Add dropout for regularization
            if filters >= 512:
                x = layers.Dropout(0.2, name=f'dropout_{i+1}')(x)
        
        # Global feature pooling
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers for regression
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.3, name='final_dropout')(x)
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        
        # Output layer for joint angles
        outputs = layers.Dense(
            self.num_joints, 
            activation='linear',  # Linear for regression
            name='joint_angles'
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='GestureClassifier')
        
        # Compile with appropriate loss for regression
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def preprocess_signal(self, rf_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess RF signal for model input
        
        Args:
            rf_signal: Raw RF signal
            
        Returns:
            Preprocessed signal ready for model
        """
        # Normalize to [-1, 1] range
        signal_min, signal_max = rf_signal.min(), rf_signal.max()
        if signal_max > signal_min:
            normalized = 2 * (rf_signal - signal_min) / (signal_max - signal_min) - 1
        else:
            normalized = rf_signal
        
        # Ensure correct shape
        if len(normalized.shape) == 1:
            normalized = normalized.reshape(1, -1, 1)
        elif len(normalized.shape) == 2:
            normalized = normalized.reshape(normalized.shape[0], -1, 1)
        
        return normalized
    
    def predict_gesture(self, rf_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict hand gesture from RF signal
        
        Args:
            rf_signal: Raw RF signal
            
        Returns:
            Dictionary containing predicted joint angles
        """
        # Preprocess input
        processed_signal = self.preprocess_signal(rf_signal)
        
        # Make prediction
        predictions = self.model.predict(processed_signal, verbose=0)
        
        # Format output
        joint_names = [
            'thumb_ip', 'thumb_mcp',
            'index_pip', 'index_mcp',
            'middle_pip', 'middle_mcp',
            'ring_pip', 'ring_mcp',
            'little_pip', 'little_mcp',
            'wrist_roll', 'wrist_pitch', 'wrist_yaw'
        ]
        
        if len(predictions.shape) == 2:
            predictions = predictions[0]  # Take first sample if batch
        
        result = {
            'joint_angles': predictions,
            'joint_names': joint_names,
            'predictions_dict': dict(zip(joint_names, predictions))
        }
        
        return result
    
    def train_model(self, rf_signals: np.ndarray, joint_angles: np.ndarray,
                   validation_split: float = 0.2, epochs: int = 100) -> keras.callbacks.History:
        """
        Train the gesture classification model
        
        Args:
            rf_signals: Training RF signals
            joint_angles: Ground truth joint angles
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        # Preprocess signals
        processed_signals = self.preprocess_signal(rf_signals)
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            processed_signals,
            joint_angles,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history
        return history


class EchomyographySystem:
    """
    Main system class that integrates all components
    
    Provides a unified interface for both diaphragm monitoring
    and hand gesture tracking applications.
    """
    
    def __init__(self, application_mode: str = "diaphragm"):
        """
        Initialize the echomyography system
        
        Args:
            application_mode: Either "diaphragm" or "gesture"
        """
        self.application_mode = application_mode
        self.signal_processor = UltrasoundSignalProcessor()
        
        if application_mode == "diaphragm":
            self.monitor = DiaphragmMonitor(self.signal_processor)
        elif application_mode == "gesture":
            self.classifier = HandGestureClassifier()
        else:
            raise ValueError("application_mode must be 'diaphragm' or 'gesture'")
        
        self.data_buffer = []
        self.results_history = []
    
    def process_real_time_frame(self, rf_signal: np.ndarray) -> Dict:
        """
        Process a single frame in real-time
        
        Args:
            rf_signal: Raw RF signal frame
            
        Returns:
            Processing results specific to application mode
        """
        if self.application_mode == "diaphragm":
            results = self.monitor.process_rf_frame(rf_signal)
            
        elif self.application_mode == "gesture":
            results = self.classifier.predict_gesture(rf_signal)
            
        # Store results
        self.results_history.append(results)
        
        return results
    
    def generate_report(self, time_window: int = 300) -> Dict:
        """
        Generate analysis report for recent data
        
        Args:
            time_window: Number of recent frames to analyze
            
        Returns:
            Comprehensive analysis report
        """
        if len(self.results_history) == 0:
            return {"error": "No data available"}
        
        recent_results = self.results_history[-time_window:]
        
        if self.application_mode == "diaphragm":
            return self._generate_breathing_report(recent_results)
        elif self.application_mode == "gesture":
            return self._generate_gesture_report(recent_results)
    
    def _generate_breathing_report(self, results: List[Dict]) -> Dict:
        """Generate breathing analysis report"""
        if not results:
            return {"error": "No breathing data"}
        
        # Extract parameters
        dtf_values = [r.get('dtf', 0) for r in results if 'dtf' in r]
        thickness_values = [r.get('thickness', 0) for r in results if 'thickness' in r]
        breathing_modes = [r.get('breathing_mode', 'unknown') for r in results if 'breathing_mode' in r]
        
        # Calculate statistics
        report = {
            "analysis_window": len(results),
            "average_dtf": np.mean(dtf_values) if dtf_values else 0,
            "std_dtf": np.std(dtf_values) if dtf_values else 0,
            "average_thickness": np.mean(thickness_values) if thickness_values else 0,
            "thickness_range": (np.min(thickness_values), np.max(thickness_values)) if thickness_values else (0, 0),
            "dominant_breathing_mode": max(set(breathing_modes), key=breathing_modes.count) if breathing_modes else "unknown",
            "breathing_mode_distribution": {mode: breathing_modes.count(mode) for mode in set(breathing_modes)}
        }
        
        return report
    
    def _generate_gesture_report(self, results: List[Dict]) -> Dict:
        """Generate gesture analysis report"""
        if not results:
            return {"error": "No gesture data"}
        
        # Extract joint angles
        all_angles = [r.get('joint_angles', []) for r in results if 'joint_angles' in r]
        joint_names = results[0].get('joint_names', []) if results else []
        
        if not all_angles or not joint_names:
            return {"error": "No valid gesture data"}
        
        all_angles = np.array(all_angles)
        
        # Calculate movement statistics
        report = {
            "analysis_window": len(results),
            "joint_statistics": {},
            "movement_activity": np.std(all_angles, axis=0).tolist() if len(all_angles) > 1 else [],
            "average_positions": np.mean(all_angles, axis=0).tolist()
        }
        
        # Per-joint statistics
        for i, joint_name in enumerate(joint_names):
            if i < all_angles.shape[1]:
                joint_data = all_angles[:, i]
                report["joint_statistics"][joint_name] = {
                    "mean_angle": float(np.mean(joint_data)),
                    "std_angle": float(np.std(joint_data)),
                    "range": (float(np.min(joint_data)), float(np.max(joint_data))),
                    "activity_level": float(np.std(joint_data))
                }
        
        return report
    
    def visualize_results(self, time_window: int = 200):
        """
        Create visualization of recent results
        
        Args:
            time_window: Number of recent frames to plot
        """
        if len(self.results_history) == 0:
            print("No data to visualize")
            return
        
        recent_results = self.results_history[-time_window:]
        
        if self.application_mode == "diaphragm":
            self._plot_breathing_analysis(recent_results)
        elif self.application_mode == "gesture":
            self._plot_gesture_analysis(recent_results)
    
    def _plot_breathing_analysis(self, results: List[Dict]):
        """Plot breathing analysis"""
        thickness = [r.get('thickness', 0) for r in results]
        dtf = [r.get('dtf', 0) for r in results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot thickness over time
        ax1.plot(thickness, 'b-', linewidth=2, label='Diaphragm Thickness')
        ax1.set_ylabel('Thickness (mm)')
        ax1.set_title('Diaphragm Thickness Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot DTF over time
        ax2.plot(dtf, 'r-', linewidth=2, label='Diaphragm Thickening Fraction')
        ax2.axhline(y=0.25, color='g', linestyle='--', alpha=0.7, label='Abdominal Threshold')
        ax2.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='Mixed Threshold')
        ax2.set_ylabel('DTF')
        ax2.set_xlabel('Frame Number')
        ax2.set_title('Breathing Pattern Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_gesture_analysis(self, results: List[Dict]):
        """Plot gesture analysis"""
        all_angles = [r.get('joint_angles', []) for r in results if 'joint_angles' in r]
        joint_names = results[0].get('joint_names', []) if results else []
        
        if not all_angles or not joint_names:
            print("No valid gesture data to plot")
            return
        
        all_angles = np.array(all_angles)
        
        # Plot subset of most active joints
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Wrist angles
        wrist_indices = [-3, -2, -1]  # roll, pitch, yaw
        wrist_names = joint_names[-3:]
        
        for i, (idx, name) in enumerate(zip(wrist_indices, wrist_names)):
            if idx < all_angles.shape[1]:
                axes[0].plot(all_angles[:, idx], label=name, linewidth=2)
        
        axes[0].set_title('Wrist Joint Angles')
        axes[0].set_ylabel('Angle (degrees)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Finger MCP joints
        mcp_indices = [1, 3, 5, 7, 9]  # MCP joints
        mcp_names = [joint_names[i] for i in mcp_indices if i < len(joint_names)]
        
        for i, (idx, name) in enumerate(zip(mcp_indices, mcp_names)):
            if idx < all_angles.shape[1]:
                axes[1].plot(all_angles[:, idx], label=name, linewidth=2)
        
        axes[1].set_title('Finger MCP Joint Angles')
        axes[1].set_ylabel('Angle (degrees)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Finger PIP joints
        pip_indices = [0, 2, 4, 6, 8]  # PIP joints
        pip_names = [joint_names[i] for i in pip_indices if i < len(joint_names)]
        
        for i, (idx, name) in enumerate(zip(pip_indices, pip_names)):
            if idx < all_angles.shape[1]:
                axes[2].plot(all_angles[:, idx], label=name, linewidth=2)
        
        axes[2].set_title('Finger PIP Joint Angles')
        axes[2].set_ylabel('Angle (degrees)')
        axes[2].set_xlabel('Frame Number')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def demonstrate_system():
    """
    Demonstration function showing how to use the echomyography system
    """
    print("🔬 Wearable Echomyography System Demonstration")
    print("=" * 50)
    
    # Simulate diaphragm monitoring
    print("\n📊 DIAPHRAGM MONITORING MODE")
    print("-" * 30)
    
    diaphragm_system = EchomyographySystem("diaphragm")
    
    # Simulate breathing patterns with synthetic data
    print("Simulating breathing patterns...")
    
    for i in range(100):
        # Create synthetic RF signal with breathing pattern
        time = np.linspace(0, 85.3, 1024)  # 85.3 µs for 1024 samples at 12 MHz
        
        # Simulate diaphragm thickness variation (breathing cycle)
        breathing_freq = 0.2  # Hz (12 breaths per minute)
        thickness_variation = 2 + 0.8 * np.sin(2 * np.pi * breathing_freq * i / 50)
        
        # Create synthetic RF with tissue reflections
        rf_signal = np.zeros(1024)
        rf_signal[200] = 1.0  # Pleura reflection
        rf_signal[int(200 + thickness_variation * 20)] = 0.8  # Peritoneum reflection
        
        # Add noise and realistic RF characteristics
        rf_signal += 0.1 * np.random.randn(1024)
        
        # Process frame
        results = diaphragm_system.process_real_time_frame(rf_signal)
        
        if i % 20 == 0:  # Print every 20th frame
            print(f"Frame {i:3d}: Thickness={results['thickness']:.2f}mm, "
                  f"DTF={results['dtf']:.3f}, Mode={results['breathing_mode']}")
    
    # Generate report
    report = diaphragm_system.generate_report()
    print(f"\n📋 BREATHING ANALYSIS REPORT:")
    print(f"   Average DTF: {report['average_dtf']:.3f}")
    print(f"   Dominant breathing mode: {report['dominant_breathing_mode']}")
    print(f"   Average thickness: {report['average_thickness']:.2f} mm")
    
    # Demonstrate gesture recognition
    print("\n\n🤏 HAND GESTURE TRACKING MODE")
    print("-" * 30)
    
    gesture_system = EchomyographySystem("gesture")
    
    print("Simulating hand gesture tracking...")
    
    # Simulate some gesture data
    for i in range(50):
        # Create synthetic RF signal representing muscle activity
        rf_signal = np.random.randn(1024) * 0.1
        
        # Add simulated muscle reflections
        for muscle_depth in [150, 280, 420, 580]:
            muscle_activity = 0.5 + 0.3 * np.sin(2 * np.pi * 0.1 * i + muscle_depth/100)
            rf_signal[muscle_depth:muscle_depth+10] += muscle_activity
        
        # Process frame
        results = gesture_system.process_real_time_frame(rf_signal)
        
        if i % 10 == 0:  # Print every 10th frame
            angles = results['joint_angles']
            print(f"Frame {i:2d}: Wrist=[{angles[-3]:.1f}°, {angles[-2]:.1f}°, {angles[-1]:.1f}°]")
    
    # Generate gesture report
    gesture_report = gesture_system.generate_report()
    print(f"\n📋 GESTURE ANALYSIS REPORT:")
    print(f"   Analysis window: {gesture_report['analysis_window']} frames")
    print(f"   Most active joint: {max(gesture_report['joint_statistics'].items(), key=lambda x: x[1]['activity_level'])[0]}")
    
    print("\n✅ Demonstration completed!")
    print("The system successfully demonstrated both diaphragm monitoring and gesture tracking.")


if __name__ == "__main__":
    demonstrate_system()

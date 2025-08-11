#!/usr/bin/env python3
"""
Practical Examples for Wearable Echomyography System

This script demonstrates real-world usage scenarios including:
1. Simulated data processing
2. Real-time monitoring setup
3. Clinical analysis workflows
4. Integration with external devices

Run this script to see the system in action!
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
from pathlib import Path

# Import our echomyography system
from echomyography_system import (
    EchomyographySystem, 
    UltrasoundSignalProcessor,
    DiaphragmMonitor,
    HandGestureClassifier
)


class DataSimulator:
    """
    Simulate realistic ultrasound data for testing and demonstration
    """
    
    def __init__(self, mode="diaphragm"):
        self.mode = mode
        self.frame_count = 0
        
    def generate_diaphragm_rf(self, breathing_phase=0, noise_level=0.1):
        """Generate realistic diaphragm RF signal"""
        # Create time axis (1024 samples at 12 MHz)
        samples = 1024
        rf_signal = np.zeros(samples)
        
        # Simulate tissue layers
        pleura_depth = 180  # Pleura reflection
        diaphragm_thickness = 3.5 + 1.2 * np.sin(breathing_phase)  # Breathing variation
        peritoneum_depth = pleura_depth + int(diaphragm_thickness * 25)  # Scale to samples
        
        # Add tissue reflections with realistic amplitudes
        rf_signal[pleura_depth] = 0.8  # Strong pleura reflection
        rf_signal[peritoneum_depth] = 0.6  # Peritoneum reflection
        
        # Add liver reflection (deeper)
        liver_depth = peritoneum_depth + 40
        if liver_depth < samples:
            rf_signal[liver_depth] = 0.4
        
        # Add realistic noise and attenuation
        rf_signal += noise_level * np.random.randn(samples)
        
        # Apply depth-dependent attenuation
        attenuation = np.exp(-np.arange(samples) * 0.001)
        rf_signal *= attenuation
        
        return rf_signal
    
    def generate_forearm_rf(self, gesture_state=None, noise_level=0.1):
        """Generate realistic forearm muscle RF signal"""
        samples = 1024
        rf_signal = np.zeros(samples)
        
        # Default gesture state
        if gesture_state is None:
            gesture_state = {
                'finger_flexion': 0.0,
                'wrist_rotation': 0.0,
                'muscle_tension': 0.2
            }
        
        # Simulate multiple muscle layers
        muscle_depths = [120, 180, 250, 320, 450, 580]  # Different muscle groups
        
        for i, depth in enumerate(muscle_depths):
            # Muscle activity affects reflection amplitude
            base_amplitude = 0.3 + 0.2 * gesture_state['muscle_tension']
            
            # Finger movements affect specific muscles
            if i < 3:  # Superficial muscles
                amplitude = base_amplitude + 0.3 * gesture_state['finger_flexion']
            else:  # Deeper muscles
                amplitude = base_amplitude + 0.2 * gesture_state['wrist_rotation']
            
            # Add muscle reflection
            if depth < samples:
                rf_signal[depth:depth+5] += amplitude * np.random.randn(5) * 0.1
                rf_signal[depth] += amplitude
        
        # Add noise and attenuation
        rf_signal += noise_level * np.random.randn(samples)
        attenuation = np.exp(-np.arange(samples) * 0.0008)
        rf_signal *= attenuation
        
        return rf_signal


def demonstrate_breathing_monitoring():
    """
    Comprehensive demonstration of breathing pattern monitoring
    """
    print("🫁 BREATHING MONITORING DEMONSTRATION")
    print("=" * 50)
    
    # Initialize system
    system = EchomyographySystem("diaphragm")
    simulator = DataSimulator("diaphragm")
    
    # Simulate different breathing scenarios
    scenarios = [
        {"name": "Normal Breathing", "rate": 0.25, "depth": 1.0, "frames": 150},
        {"name": "Deep Breathing", "rate": 0.15, "depth": 1.8, "frames": 100},
        {"name": "Shallow Breathing", "rate": 0.35, "depth": 0.3, "frames": 100},
        {"name": "Exercise Breathing", "rate": 0.45, "depth": 1.5, "frames": 80},
    ]
    
    all_results = []
    scenario_labels = []
    
    print("\n📊 Processing breathing scenarios...")
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        scenario_results = []
        
        for frame in range(scenario['frames']):
            # Calculate breathing phase
            phase = 2 * np.pi * scenario['rate'] * frame / 50  # 50 Hz frame rate
            breathing_amplitude = scenario['depth']
            
            # Generate RF signal
            rf_signal = simulator.generate_diaphragm_rf(
                breathing_phase=phase * breathing_amplitude,
                noise_level=0.08
            )
            
            # Process frame
            results = system.process_real_time_frame(rf_signal)
            scenario_results.append(results)
            
            # Print progress
            if frame % 30 == 0:
                print(f"  Frame {frame:3d}: Thickness={results['thickness']:.2f}mm, "
                      f"DTF={results['dtf']:.3f}, Mode={results['breathing_mode']}")
        
        all_results.extend(scenario_results)
        scenario_labels.extend([scenario['name']] * len(scenario_results))
        
        # Generate report for this scenario
        # Reset system for clean analysis
        temp_system = EchomyographySystem("diaphragm")
        for result in scenario_results:
            temp_system.results_history.append(result)
        
        report = temp_system.generate_report()
        print(f"  📋 Summary: DTF={report['average_dtf']:.3f}, "
              f"Mode={report['dominant_breathing_mode']}")
    
    # Create comprehensive visualization
    plot_breathing_analysis(all_results, scenario_labels)
    
    return all_results


def demonstrate_gesture_tracking():
    """
    Demonstrate hand gesture recognition capabilities
    """
    print("\n\n🤏 HAND GESTURE TRACKING DEMONSTRATION")
    print("=" * 50)
    
    # Initialize system
    system = EchomyographySystem("gesture")
    simulator = DataSimulator("forearm")
    
    # Define gesture sequences
    gestures = [
        {"name": "Rest Position", "finger_flexion": 0.0, "wrist_rotation": 0.0, "frames": 50},
        {"name": "Make Fist", "finger_flexion": 1.0, "wrist_rotation": 0.0, "frames": 60},
        {"name": "Point Index", "finger_flexion": 0.2, "wrist_rotation": 0.0, "frames": 40},
        {"name": "Wrist Flex", "finger_flexion": 0.1, "wrist_rotation": 0.8, "frames": 50},
        {"name": "Combined Motion", "finger_flexion": 0.6, "wrist_rotation": 0.5, "frames": 70},
    ]
    
    all_results = []
    gesture_labels = []
    
    print("\n📊 Processing gesture sequences...")
    
    for gesture in gestures:
        print(f"\n--- {gesture['name']} ---")
        
        for frame in range(gesture['frames']):
            # Create gesture state with smooth transitions
            transition_factor = min(frame / 20, 1.0)  # Smooth ramp-up
            
            gesture_state = {
                'finger_flexion': gesture['finger_flexion'] * transition_factor,
                'wrist_rotation': gesture['wrist_rotation'] * transition_factor,
                'muscle_tension': 0.3 + 0.2 * max(gesture['finger_flexion'], gesture['wrist_rotation'])
            }
            
            # Generate RF signal
            rf_signal = simulator.generate_forearm_rf(gesture_state, noise_level=0.1)
            
            # Process frame
            results = system.process_real_time_frame(rf_signal)
            all_results.append(results)
            gesture_labels.append(gesture['name'])
            
            # Print sample results
            if frame % 15 == 0:
                angles = results['joint_angles']
                print(f"  Frame {frame:2d}: Wrist=[{angles[-3]:.1f}°, {angles[-2]:.1f}°, {angles[-1]:.1f}°]")
    
    # Generate comprehensive report
    report = system.generate_report()
    print(f"\n📋 GESTURE TRACKING SUMMARY:")
    print(f"   Total frames processed: {len(all_results)}")
    print(f"   Most active joint: {max(report['joint_statistics'].items(), key=lambda x: x[1]['activity_level'])[0]}")
    
    # Visualize results
    plot_gesture_analysis(all_results, gesture_labels)
    
    return all_results


def plot_breathing_analysis(results, labels):
    """Create detailed breathing analysis plots"""
    thickness_data = [r['thickness'] for r in results]
    dtf_data = [r['dtf'] for r in results]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Thickness over time with scenario labels
    axes[0].plot(thickness_data, linewidth=2, color='blue', alpha=0.8)
    axes[0].set_ylabel('Diaphragm Thickness (mm)')
    axes[0].set_title('Diaphragm Thickness Variation Across Breathing Scenarios')
    axes[0].grid(True, alpha=0.3)
    
    # Add scenario boundaries
    current_pos = 0
    colors = ['red', 'green', 'orange', 'purple']
    unique_labels = []
    for i, label in enumerate(labels):
        if i == 0 or label != labels[i-1]:
            if len(unique_labels) < len(colors):
                axes[0].axvline(x=i, color=colors[len(unique_labels)], 
                              linestyle='--', alpha=0.7, linewidth=2)
                axes[0].text(i+5, max(thickness_data)*0.9 - len(unique_labels)*0.5, 
                           label, color=colors[len(unique_labels)], fontweight='bold')
                unique_labels.append(label)
    
    # Plot 2: DTF analysis
    axes[1].plot(dtf_data, linewidth=2, color='red', alpha=0.8)
    axes[1].axhline(y=0.25, color='green', linestyle='--', alpha=0.7, 
                   label='Abdominal Threshold (DTF > 0.25)')
    axes[1].axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, 
                   label='Mixed Threshold (DTF > 0.10)')
    axes[1].set_ylabel('Diaphragm Thickening Fraction')
    axes[1].set_title('Breathing Pattern Classification (DTF Analysis)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Breathing mode distribution
    breathing_modes = [r['breathing_mode'] for r in results]
    mode_counts = {}
    for mode in breathing_modes:
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    modes = list(mode_counts.keys())
    counts = list(mode_counts.values())
    colors_pie = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
    
    axes[2].pie(counts, labels=modes, autopct='%1.1f%%', 
               colors=colors_pie[:len(modes)], startangle=90)
    axes[2].set_title('Overall Breathing Mode Distribution')
    
    plt.tight_layout()
    plt.savefig('breathing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_gesture_analysis(results, labels):
    """Create detailed gesture analysis plots"""
    # Extract angle data
    all_angles = np.array([r['joint_angles'] for r in results])
    joint_names = results[0]['joint_names']
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Plot 1: Wrist angles
    wrist_indices = [-3, -2, -1]  # roll, pitch, yaw
    wrist_names = ['Roll', 'Pitch', 'Yaw']
    colors = ['red', 'blue', 'green']
    
    for i, (idx, name, color) in enumerate(zip(wrist_indices, wrist_names, colors)):
        axes[0].plot(all_angles[:, idx], label=f'Wrist {name}', 
                    color=color, linewidth=2, alpha=0.8)
    
    axes[0].set_title('Wrist Joint Angles Over Time')
    axes[0].set_ylabel('Angle (degrees)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Index finger
    index_mcp = all_angles[:, 3]  # Index MCP
    index_pip = all_angles[:, 2]  # Index PIP
    
    axes[1].plot(index_mcp, label='Index MCP', color='darkblue', linewidth=2)
    axes[1].plot(index_pip, label='Index PIP', color='lightblue', linewidth=2)
    axes[1].set_title('Index Finger Joint Angles')
    axes[1].set_ylabel('Angle (degrees)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Overall activity level
    activity_level = np.std(all_angles, axis=1)
    axes[2].plot(activity_level, color='purple', linewidth=2, alpha=0.8)
    axes[2].set_title('Overall Hand Activity Level')
    axes[2].set_ylabel('Activity Score')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Gesture timeline
    unique_labels = []
    label_positions = []
    for i, label in enumerate(labels):
        if i == 0 or label != labels[i-1]:
            unique_labels.append(label)
            label_positions.append(i)
    
    for i, (pos, label) in enumerate(zip(label_positions, unique_labels)):
        color = plt.cm.Set3(i / len(unique_labels))
        axes[3].axvline(x=pos, color=color, linewidth=3, alpha=0.7)
        axes[3].text(pos + 2, 0.8 - (i % 3) * 0.2, label, 
                    color=color, fontweight='bold', fontsize=10)
    
    axes[3].set_title('Gesture Timeline')
    axes[3].set_xlabel('Frame Number')
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('gesture_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def real_time_monitoring_demo():
    """
    Demonstrate real-time monitoring capabilities
    """
    print("\n\n⚡ REAL-TIME MONITORING DEMONSTRATION")
    print("=" * 50)
    
    # Initialize both systems
    diaphragm_system = EchomyographySystem("diaphragm")
    gesture_system = EchomyographySystem("gesture")
    simulators = {
        "diaphragm": DataSimulator("diaphragm"),
        "gesture": DataSimulator("forearm")
    }
    
    print("🔴 Starting real-time monitoring (simulated)...")
    print("Press Ctrl+C to stop\n")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 200:  # Limited demo
            current_time = time.time()
            
            # Simulate 50 Hz frame rate
            if (current_time - start_time) >= frame_count / 50.0:
                
                # Process diaphragm data
                breathing_phase = 2 * np.pi * 0.2 * frame_count / 50
                diaphragm_rf = simulators["diaphragm"].generate_diaphragm_rf(breathing_phase)
                diaphragm_results = diaphragm_system.process_real_time_frame(diaphragm_rf)
                
                # Process gesture data
                gesture_state = {
                    'finger_flexion': 0.5 + 0.3 * np.sin(0.1 * frame_count),
                    'wrist_rotation': 0.2 * np.sin(0.05 * frame_count),
                    'muscle_tension': 0.3
                }
                gesture_rf = simulators["gesture"].generate_forearm_rf(gesture_state)
                gesture_results = gesture_system.process_real_time_frame(gesture_rf)
                
                # Display results every 25 frames (0

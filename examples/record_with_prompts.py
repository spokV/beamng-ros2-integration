#!/usr/bin/env python3
"""
Example script for recording BeamNG driving data with VLA prompts.

This script demonstrates how to:
1. Start a BeamNG scenario with driving prompts
2. Record all sensor data including driving prompts using ros2 bag
3. Create properly annotated trajectory data for VLA model training

Usage:
    python3 record_with_prompts.py --scenario path/to/scenario.json --output_bag trajectory_data
"""

import argparse
import subprocess
import time
import signal
import sys
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from beamng_msgs.srv import StartScenario
from beamng_msgs.msg import DrivingPrompt
from std_srvs.srv import Empty


class VLADataRecorder(Node):
    """Node for recording VLA training data with driving prompts."""
    
    def __init__(self):
        super().__init__('vla_data_recorder')
        
        # Service clients
        self.start_scenario_client = self.create_client(StartScenario, '/beamng_bridge/start_scenario')
        self.stop_prompt_client = self.create_client(Empty, '/vehicles/ego/stop_prompt_publisher')
        
        # Track recording state
        self.recording_process = None
        self.scenario_running = False
        
        # Subscribe to driving prompts to verify they're being published
        self.prompt_subscription = self.create_subscription(
            DrivingPrompt,
            '/beamng_bridge/driving_prompt',
            self.prompt_callback,
            10
        )
        
        self.get_logger().info("VLA Data Recorder initialized")
    
    def prompt_callback(self, msg):
        """Callback for driving prompt messages - logs prompt for verification."""
        self.get_logger().info(f"Received driving prompt: {msg.task_description}")
    
    def start_scenario(self, scenario_path: str) -> bool:
        """Start the BeamNG scenario."""
        if not self.start_scenario_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("BeamNG bridge service not available")
            return False
        
        request = StartScenario.Request()
        request.path_to_scenario_definition = str(scenario_path)
        
        self.get_logger().info(f"Starting scenario: {scenario_path}")
        future = self.start_scenario_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().success:
            self.scenario_running = True
            self.get_logger().info("Scenario started successfully")
            return True
        else:
            self.get_logger().error("Failed to start scenario")
            return False
    
    def start_recording(self, output_bag: str, topics: list = None):
        """Start recording ROS2 bag with all relevant topics."""
        if topics is None:
            # Default topics for VLA training
            topics = [
                '/vehicles/ego/driving_prompt',  # VLA driving prompts
                '/vehicles/ego/sensors/damage',
                '/vehicles/ego/sensors/electrics',
                '/vehicles/ego/sensors/front_cam/colour',
                '/vehicles/ego/sensors/gps',
                '/vehicles/ego/sensors/imu',
                '/vehicles/ego/sensors/mesh',
                '/vehicles/ego/sensors/powertrain',
                '/vehicles/ego/sensors/state',
            ]
        
        # Check which topics are actually available
        import subprocess as sp
        try:
            result = sp.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=5)
            available_topics = result.stdout.strip().split('\n')
            self.get_logger().info(f"Available topics: {available_topics}")
            
            # Filter to only record topics that actually exist
            valid_topics = [topic for topic in topics if topic in available_topics]
            self.get_logger().info(f"Valid topics to record: {valid_topics}")
            
            if not valid_topics:
                self.get_logger().warn("No valid topics found to record!")
                return None
                
        except Exception as e:
            self.get_logger().warn(f"Could not check topics: {e}, proceeding with all topics")
            valid_topics = topics
        
        # Build ros2 bag record command with explicit sourcing
        setup_cmd = "source /opt/ros/jazzy/setup.bash && source /home/spok/ros2_ws/install/setup.bash"
        record_cmd = f"ros2 bag record -o {output_bag} --max-bag-size 0 {' '.join(valid_topics)}"
        full_cmd = f"{setup_cmd} && {record_cmd}"
        
        self.get_logger().info(f"Starting bag recording: {record_cmd}")
        self.get_logger().info("Using shell to source ROS2 workspace properly")
        self.recording_process = subprocess.Popen(
            full_cmd,
            shell=True,
            executable='/bin/bash'
        )
        
        return self.recording_process
    
    def stop_recording(self):
        """Stop the ROS2 bag recording."""
        if self.recording_process:
            self.recording_process.send_signal(signal.SIGINT)
            self.recording_process.wait()
            self.get_logger().info("Recording stopped")
    
    def stop_prompt_publisher(self):
        """Stop the prompt publisher and its background threads."""
        if not self.scenario_running:
            return
            
        if self.stop_prompt_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Stopping prompt publisher...")
            request = Empty.Request()
            future = self.stop_prompt_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            self.get_logger().info("Prompt publisher stopped")
        else:
            self.get_logger().warn("Prompt publisher stop service not available")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_recording()
        self.stop_prompt_publisher()
        if self.scenario_running:
            self.scenario_running = False


def signal_handler(_, __, recorder):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down gracefully...")
    recorder.cleanup()
    # Don't call sys.exit() in IDE environment - just return to allow clean shutdown
    return


def main():
    parser = argparse.ArgumentParser(description="Record BeamNG driving data with VLA prompts")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument("--output_bag", required=True, help="Output bag file name")
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds")
    parser.add_argument("--topics", nargs="*", help="Additional topics to record")
    
    args = parser.parse_args()
    
    # Validate scenario file
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1
    
    # Initialize ROS2
    rclpy.init()
    recorder = VLADataRecorder()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, recorder))
    
    try:
        # Start the scenario
        if not recorder.start_scenario(scenario_path.absolute()):
            return 1
        
        # Wait a moment for everything to initialize
        time.sleep(2)
        
        # Create timestamped output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_output = f"{args.output_bag}_{timestamp}"
        
        # Start recording
        recorder.start_recording(timestamped_output, args.topics)
        
        # Record for specified duration
        recorder.get_logger().info(f"Recording for {args.duration} seconds...")
        recorder.get_logger().info("Drive the vehicle manually or let AI control it")
        recorder.get_logger().info("Press Ctrl+C to stop recording early")
        
        start_time = time.time()
        while time.time() - start_time < args.duration:
            rclpy.spin_once(recorder, timeout_sec=1.0)
        
        recorder.get_logger().info("Recording completed")
        
    except KeyboardInterrupt:
        recorder.get_logger().info("Recording interrupted by user")
    finally:
        recorder.cleanup()
        rclpy.shutdown()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        print(f"Program completed with exit code: {exit_code}")
    except Exception as e:
        print(f"Program failed with exception: {e}")
        exit_code = 1
    # Don't call sys.exit() in IDE - just let it end naturally